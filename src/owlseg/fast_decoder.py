"""SAM's mask decoder for many boxes on one image, without the work that is the same for every box.

Same maths as transformers 4.44's SamMaskDecoder with multimask_output=False, but:
  * the image embedding is not copied once per box; it is broadcast,
  * in the first transformer layer the image side has not seen any box yet, so its projections (keys and values
    for token->image, queries for image->token) are computed once instead of once per box,
  * only the single-mask output is computed (the eager decoder computes all 4 mask tokens and drops 3).
Results match the eager decoder to float32 rounding.
"""
from __future__ import annotations

import torch


def _heads(x: torch.Tensor, h: int) -> torch.Tensor:  # (B, n, c) -> (B, h, n, c/h)
    b, n, c = x.shape
    return x.reshape(b, n, h, c // h).transpose(1, 2)


def _merge(x: torch.Tensor) -> torch.Tensor:  # (B, h, n, d) -> (B, n, h*d)
    b, h, n, d = x.shape
    return x.transpose(1, 2).reshape(b, n, h * d)


def _attend(q, k, v):
    attn = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    return torch.softmax(attn, dim=-1) @ v


def _attention(m, q, k, v):
    """SamAttention, everything per box: q, k, v (N, n, C)."""
    h = m.num_attention_heads
    return m.out_proj(_merge(_attend(_heads(m.q_proj(q), h), _heads(m.k_proj(k), h), _heads(m.v_proj(v), h))))


def _attention_shared_kv(m, q, k, v):
    """Box tokens q (N, t, C) attend to the same image k, v (1, n, C): all boxes' tokens in one matrix."""
    n, t, _ = q.shape
    h = m.num_attention_heads
    out = _attend(_heads(m.q_proj(q).reshape(1, n * t, -1), h), _heads(m.k_proj(k), h), _heads(m.v_proj(v), h))
    return m.out_proj(_merge(out).reshape(n, t, -1))


def _attention_shared_q(m, q, k, v):
    """The same image q (1, n, C) attends to each box's tokens k, v (N, t, C)."""
    n, t, _ = k.shape
    h = m.num_attention_heads
    qh = _heads(m.q_proj(q), h)[0]  # (h, n_img, d)
    kh, vh = _heads(m.k_proj(k), h), _heads(m.v_proj(v), h)  # (N, h, t, d)
    d = qh.shape[-1]
    attn = qh @ kh.permute(1, 3, 0, 2).reshape(h, d, n * t) / (d ** 0.5)  # (h, n_img, N*t)
    attn = attn.reshape(h, -1, n, t).permute(2, 0, 1, 3)  # (N, h, n_img, t)
    return m.out_proj(_merge(torch.softmax(attn, dim=-1) @ vh))


class BoxDecoder(torch.nn.Module):
    """(image embedding (1, C, H, W), positional embedding (1, C, H, W), box embeddings (1, N, 2, C), dense
    embedding (1, C, H, W)) -> (low-res mask logits (N, 1, 4H, 4W), predicted IoU (N,))."""

    def __init__(self, decoder):
        super().__init__()
        self.dec = decoder

    def forward(self, image_embeddings, image_pe, sparse, dense):
        dec, tr = self.dec, self.dec.transformer
        n = sparse.shape[1]
        _, c, hh, ww = image_embeddings.shape
        tokens = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], dim=0)
        tokens = torch.cat([tokens.expand(n, -1, -1), sparse[0]], dim=1)  # (N, 7, C); box prompts are never all 0
        keys = (image_embeddings + dense).flatten(2).transpose(1, 2)  # (1, H*W, C), per box after layer 1
        pos = image_pe.flatten(2).transpose(1, 2)
        queries = tokens
        for i, layer in enumerate(tr.layers):
            shared = i == 0  # the image side has not been updated by any box yet
            if layer.skip_first_layer_pe:
                queries = _attention(layer.self_attn, queries, queries, queries)
            else:
                q = queries + tokens
                queries = queries + _attention(layer.self_attn, q, q, queries)
            queries = layer.layer_norm1(queries)
            q, k = queries + tokens, keys + pos
            attn = (_attention_shared_kv if shared else _attention)(layer.cross_attn_token_to_image, q, k, keys)
            queries = layer.layer_norm2(queries + attn)
            queries = layer.layer_norm3(queries + layer.mlp(queries))
            q, k = queries + tokens, keys + pos
            attn = (_attention_shared_q if shared else _attention)(layer.cross_attn_image_to_token, k, q, queries)
            keys = layer.layer_norm4(keys + attn)
        q, k = queries + tokens, keys + pos
        queries = tr.layer_norm_final_attn(queries + _attention(tr.final_attn_token_to_image, q, k, keys))

        img = keys.transpose(1, 2).reshape(n, c, hh, ww)
        up = dec.activation(dec.upscale_layer_norm(dec.upscale_conv1(img)))
        up = dec.activation(dec.upscale_conv2(up))  # (N, C/8, 4H, 4W)
        hyper = dec.output_hypernetworks_mlps[0](queries[:, 1])  # first mask token = the single-mask output
        masks = (hyper[:, None] @ up.flatten(2)).reshape(n, 1, *up.shape[-2:])
        iou = dec.iou_prediction_head(queries[:, 0])[:, 0]
        return masks, iou
