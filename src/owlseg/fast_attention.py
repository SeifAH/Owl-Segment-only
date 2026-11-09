"""Fused attention for the CUDA backend: the same fp32 maths as transformers 4.44's eager attention, in one kernel.

The eager code writes the full (tokens x tokens) attention matrix to GPU memory several times (scores, softmax,
dropout); torch's scaled_dot_product_attention computes it in tiles instead. Only the vision attention layers
that the pipeline runs are swapped, on the loaded model instances (the classes stay untouched).
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _owl_forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
    """Owlv2Attention.forward for the vision tower (no masks, no attention output)."""
    b, n, d = hidden_states.shape
    q, k, v = (p(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
               for p in (self.q_proj, self.k_proj, self.v_proj))
    out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    return self.out_proj(out.transpose(1, 2).reshape(b, n, d)), None


def _sam_forward(self, hidden_states, output_attentions=False):
    """SamVisionAttention.forward: decomposed relative positions become an additive attention bias."""
    b, h, w, _ = hidden_states.shape
    qkv = self.qkv(hidden_states).reshape(b, h * w, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # (b, heads, h*w, c)
    bias = None
    if self.use_rel_pos:
        rel_h = self.get_rel_pos(h, h, self.rel_pos_h)
        rel_w = self.get_rel_pos(w, w, self.rel_pos_w)
        rq = q.reshape(b * self.num_attention_heads, h, w, -1)
        rh = torch.einsum("bhwc,hkc->bhwk", rq, rel_h)
        rw = torch.einsum("bhwc,wkc->bhwk", rq, rel_w)
        bias = (rh[:, :, :, :, None] + rw[:, :, :, None, :]).reshape(b, self.num_attention_heads, h * w, h * w)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
    out = out.reshape(b, self.num_attention_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
    return self.proj(out), None


def use_fused_attention(owl_vision: torch.nn.Module, sam_vision: torch.nn.Module) -> None:
    from transformers.models.owlv2.modeling_owlv2 import Owlv2Attention
    from transformers.models.sam.modeling_sam import SamVisionAttention

    for m in owl_vision.modules():
        if isinstance(m, Owlv2Attention):
            m.forward = types.MethodType(_owl_forward, m)
    for m in sam_vision.modules():
        if isinstance(m, SamVisionAttention):
            m.forward = types.MethodType(_sam_forward, m)
