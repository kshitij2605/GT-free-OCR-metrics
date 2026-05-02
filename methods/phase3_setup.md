# Phase 3 one-time model setup

Phase 3 of autoresearch covers techniques that require external GitHub checkouts and pre-downloaded weights (Hi-SAM, CRAFT, FontCLIP, dino.txt). Autoresearch's Experiment Runner cannot reliably clone large repos and fetch multi-GB weights inside its 3-attempt self-heal window, so the operator runs this preparation **once** before launching the Phase 3 outer loop.

Run every command **from** `/home/mac/test/r1-p2` on iitgpu11.

## 1. Hi-SAM (hierarchical text segmentation)

```bash
cd models/
git clone https://github.com/ymy-k/Hi-SAM.git
cd Hi-SAM
mkdir -p pretrained_checkpoint
# SAM ViT-B weights (smaller, for quick experiments)
wget -O pretrained_checkpoint/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# SAM ViT-L (recommended for production — 1.2GB)
wget -O pretrained_checkpoint/sam_vit_l_0b3195.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# Hi-SAM adapter weights (see README for the current release link)
# Check https://github.com/ymy-k/Hi-SAM#release for the latest hisam_vit_b.pth / hisam_vit_l.pth
```

Verify:
```bash
ls -lh models/Hi-SAM/pretrained_checkpoint/
```

## 2. CRAFT (text detection / perceptual features)

CRAFT-pytorch is on PyPI but the official Clova weights live on Google Drive. Quickest path:
```bash
pip install craft-text-detection  # provided the autoresearch scope.allowed_packages includes this
```
First invocation downloads `craft_mlt_25k.pth` automatically into `~/.craft/`. No manual step needed, but a dry run keeps it off the autoresearch budget:
```bash
uv run python -c "from craft_text_detector import Craft; Craft(output_dir='/tmp/craft-warm', cuda=False); print('craft ready')"
```

## 3. FontCLIP (typography embeddings)

```bash
cd models/
git clone https://github.com/yukistavailable/FontCLIP.git
cd FontCLIP
# Follow README — typically requires base CLIP + fine-tuned adapter checkpoint
# Download instructions are in their README.md. Place under models/FontCLIP/checkpoints/
```

## 4. MinerU-Diffusion (generative OCR backend; Phase 2 actually, but big)

```bash
# Pre-cache the HF weights to avoid download during experiments (5GB).
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='opendatalab/MinerU-Diffusion-V1-0320-2.5B',
    cache_dir='models/hf_cache',
)
print('minerU diffusion cached')
"
```

## 5. DINOv3 (for Phase 2 / Phase 3 patch metrics)

```bash
uv run python -c "
from huggingface_hub import snapshot_download
for repo in ['facebook/dinov3-vitb16-pretrain-lvd1689m',
             'facebook/dinov3-vitl16-pretrain-lvd1689m']:
    snapshot_download(repo_id=repo, cache_dir='models/hf_cache')
print('dinov3 cached')
"
```

## 6. dino.txt (text-queried patch localisation)

dino.txt code is inside `facebookresearch/dinov3` after the Jan 2026 update. Text-head checkpoints are downloaded on first use of `dinov3.load('dinov3_vitl16', text=True)` — run that once:
```bash
uv run python -c "
import sys; sys.path.insert(0, 'models/Hi-SAM')   # anywhere with torch hub
import torch
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16', source='github', pretrained=True)
print('dinov3 hub ok')
"
```

## 7. ColPali (document retrieval, Phase 2)

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='vidore/colpali-v1.3-hf', cache_dir='models/hf_cache')
print('colpali cached')
"
```

## Verification

All downloaded paths visible in one summary:
```bash
du -sh models/Hi-SAM models/FontCLIP models/hf_cache 2>/dev/null
ls models/hf_cache/models--* 2>/dev/null | head
```

## After preparation

Generate the Phase 3 program and launch autoresearch:
```bash
uv run python scripts/build_autoresearch_program.py --phase 3
bash /home/mac/test/autoresearch/outer_loop.sh \
    --project /home/mac/test/r1-p2 \
    --tag phase3 \
    --max-time 168h \
    --max-experiments 60
```

If disk pressure becomes an issue, the `models/hf_cache/` entries for a finished technique can be pruned after its results.json is archived — autoresearch does not re-download unless `cache_dirs` is cleared.
