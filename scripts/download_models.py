"""Pre-download OWLv2 (~620 MB) and SAM ViT-B (~375 MB) into the Hugging Face cache (~/.cache/huggingface)."""
from huggingface_hub import snapshot_download

from owlseg.models import OWL_MODEL, SAM_MODEL

# max_workers=1: on Windows without symlink privileges, huggingface_hub's symlink check
# races across download threads and crashes with WinError 1314.
for name, repo in (("OWLv2", OWL_MODEL), ("SAM", SAM_MODEL)):
    print(f"{name} ({repo})...")
    print("  ->", snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors"], max_workers=1))
print("Done.")
