#!/usr/bin/env python3
"""train_docsim.py — fine-tune a DocSim head via LoRA on edit-distance triplets.

D60 step 3 of 4. Companion to scripts/build_docsim_triplets.py.

Architecture:
    OpenCLIP(ViT-B/32, laion2b)  → CLS  (512-dim)  ┐
    DINOv2(vitb14)               → CLS  (768-dim)  ├── concat → 1280-dim
                                                    ┘
    LoRA-adapted MLP head    (1280 → 512 → 256-dim normalized embedding)

Loss:
    Triplet margin loss with label smoothing — matches DreamSim's approach.
        L = max(0, m + cos(anchor, neg) - cos(anchor, pos))
    The backbones stay frozen; only the LoRA-wrapped head trains.

Why these backbones (not ColPali / DiT / LayoutLMv3):
    - DINOv2 + OpenCLIP are already proven in our pipeline (H4.e configuration).
    - Both are NOT task-trained for OCR/layout, so they don't carry the
      trained-invariance trap that killed D104 TrOCR / D193 DocLayout / D10
      ColPali. The LoRA head learns the OCR-fidelity-preserving projection.
    - Light: 1280-dim concat fits in 1 GPU comfortably; full training run
      ~16-24h on RTX 6000 Ada.
    - The LoRA-on-MLP-head approach matches the original DreamSim recipe
      (Fu et al. NeurIPS 2023).

Usage:
    python scripts/train_docsim.py \\
        --triplets data/omnidocbench/docsim_triplets/manifest.jsonl \\
        --data-root data/omnidocbench \\
        --output models/docsim_lora/ \\
        --epochs 3 --batch-size 16 --lr 1e-4

Output:
    models/docsim_lora/
        adapter_weights.safetensors    LoRA weights
        config.json                    backbone names + LoRA config
        head_state.safetensors         MLP head weights
        train_log.jsonl                per-step metrics
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Setup ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_docsim")


# ── Triplet dataset ───────────────────────────────────────────────────────────


class TripletDataset(Dataset):
    """Reads a JSONL manifest produced by build_docsim_triplets.py."""

    def __init__(self, manifest_path: Path, transform_clip, transform_dino, project_root: Path):
        self.records = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
        self.t_clip = transform_clip
        self.t_dino = transform_dino
        self.root = project_root

    def __len__(self):
        return len(self.records)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.root / path)

    def _load(self, p: str) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self._resolve(p)).convert("RGB")
        return self.t_clip(img), self.t_dino(img)

    def __getitem__(self, i):
        r = self.records[i]
        a_c, a_d = self._load(r["anchor_path"])
        p_c, p_d = self._load(r["positive_path"])
        n_c, n_d = self._load(r["negative_path"])
        return {
            "a_c": a_c, "a_d": a_d,
            "p_c": p_c, "p_d": p_d,
            "n_c": n_c, "n_d": n_d,
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Model ─────────────────────────────────────────────────────────────────────


@dataclass
class DocSimConfig:
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    dino_model: str = "dinov2_vitb14"
    embed_dim: int = 256
    hidden_dim: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    margin: float = 0.10


class FrozenBackbones(nn.Module):
    """Loads OpenCLIP + DINOv2, both in eval mode with frozen weights.

    Returns a dict of {clip_preprocess, dino_transform, encode(images_clip, images_dino)}.
    """

    def __init__(self, cfg: DocSimConfig, device: str):
        super().__init__()
        import open_clip

        self.device = device

        log.info("Loading OpenCLIP %s (%s)…", cfg.clip_model, cfg.clip_pretrained)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model, pretrained=cfg.clip_pretrained
        )
        self.clip_model.eval().to(device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_dim = self.clip_model.visual.output_dim  # 512 for ViT-B/32

        log.info("Loading DINOv2 %s…", cfg.dino_model)
        self.dino_model = torch.hub.load("facebookresearch/dinov2", cfg.dino_model, trust_repo=True)
        self.dino_model.eval().to(device)
        for p in self.dino_model.parameters():
            p.requires_grad = False
        self.dino_dim = 768  # vitb14 CLS dim

        # DINOv2 transform — match published recipe (224×224, ImageNet stats)
        self.dino_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode(self, x_clip: torch.Tensor, x_dino: torch.Tensor) -> torch.Tensor:
        x_clip = x_clip.to(self.device)
        x_dino = x_dino.to(self.device)
        f_clip = self.clip_model.encode_image(x_clip).float()
        f_dino = self.dino_model(x_dino).float()  # CLS token
        f_clip = F.normalize(f_clip, dim=-1)
        f_dino = F.normalize(f_dino, dim=-1)
        return torch.cat([f_clip, f_dino], dim=-1)  # (B, 1280)


class DocSimHead(nn.Module):
    """MLP head wrapped with LoRA. Trains on the concatenated frozen backbones' output."""

    def __init__(self, in_dim: int, cfg: DocSimConfig):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        # Plain MLP head
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )
        # Wrap the proj's Linear layers with LoRA. Targeting the linear layers' weight matrices
        # gives a small, efficient adapter while still letting the head learn the projection.
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["0", "2"],  # the two Linear layers in the Sequential
            bias="none",
        )
        self.proj = get_peft_model(self.proj, lora_cfg)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


# ── Training loop ─────────────────────────────────────────────────────────────


def train(cfg: DocSimConfig, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)
    if device == "cpu":
        log.warning("Training on CPU will be extremely slow.")

    # 1) Build backbones
    backbones = FrozenBackbones(cfg, device)
    log.info("Backbones loaded. clip_dim=%d dino_dim=%d", backbones.clip_dim, backbones.dino_dim)

    # 2) Build head + LoRA
    head = DocSimHead(in_dim=backbones.clip_dim + backbones.dino_dim, cfg=cfg).to(device)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in head.parameters())
    log.info("Head params: trainable=%d total=%d (LoRA fraction: %.2f%%)",
             n_trainable, n_total, 100 * n_trainable / max(1, n_total))

    # 3) Dataset / loader
    data_root = args.data_root if args.data_root.is_absolute() else Path.cwd() / args.data_root
    ds = TripletDataset(args.triplets, backbones.clip_preprocess, backbones.dino_transform, data_root)
    log.info("Loaded %d triplets from %s", len(ds), args.triplets)

    n_train = int(0.95 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate, pin_memory=True,
    )

    # 4) Optimizer + scheduler
    opt = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # 5) Output dir + log
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    def encode_triplet(b):
        a = backbones.encode(b["a_c"], b["a_d"])  # (B, 1280)
        p = backbones.encode(b["p_c"], b["p_d"])
        n = backbones.encode(b["n_c"], b["n_d"])
        return head(a), head(p), head(n)  # each (B, 256), L2-normed

    # 6) Train
    step = 0
    t0 = time.time()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        head.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for i, batch in enumerate(train_loader):
            a, p, n = encode_triplet(batch)
            d_pos = 1.0 - (a * p).sum(dim=-1)  # cosine distance
            d_neg = 1.0 - (a * n).sum(dim=-1)
            loss = F.relu(cfg.margin + d_pos - d_neg).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            with torch.no_grad():
                running_correct += (d_pos < d_neg).sum().item()
                running_total += a.size(0)
                running_loss += loss.item() * a.size(0)

            step += 1
            if step % args.log_every == 0:
                avg_loss = running_loss / max(1, running_total)
                acc = running_correct / max(1, running_total)
                lr = sched.get_last_lr()[0]
                rec = {
                    "epoch": epoch, "step": step, "loss": avg_loss, "train_acc": acc,
                    "lr": lr, "elapsed_s": time.time() - t0,
                }
                log.info("ep=%d step=%d loss=%.4f acc=%.3f lr=%.2e elapsed=%.0fs",
                         epoch, step, avg_loss, acc, lr, rec["elapsed_s"])
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                running_loss = running_correct = running_total = 0

        # validation after each epoch
        head.eval()
        with torch.no_grad():
            v_correct, v_total, v_loss = 0, 0, 0.0
            for batch in val_loader:
                a, p, n = encode_triplet(batch)
                d_pos = 1.0 - (a * p).sum(dim=-1)
                d_neg = 1.0 - (a * n).sum(dim=-1)
                loss = F.relu(cfg.margin + d_pos - d_neg).mean()
                v_correct += (d_pos < d_neg).sum().item()
                v_total += a.size(0)
                v_loss += loss.item() * a.size(0)
            v_acc = v_correct / max(1, v_total)
            v_avg = v_loss / max(1, v_total)
            log.info("[VAL ep=%d] loss=%.4f acc=%.3f", epoch, v_avg, v_acc)
            log_f.write(json.dumps({"epoch": epoch, "val_loss": v_avg, "val_acc": v_acc}) + "\n")
            log_f.flush()

        # checkpoint
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            head.proj.save_pretrained(str(args.output / "lora_adapter_best"))
            torch.save(head.state_dict(), args.output / "head_state_best.pt")
            log.info("  ✓ checkpointed best (val_acc=%.3f)", v_acc)

    # final save
    head.proj.save_pretrained(str(args.output / "lora_adapter_final"))
    torch.save(head.state_dict(), args.output / "head_state_final.pt")
    cfg_dict = {**cfg.__dict__, "best_val_acc": best_val_acc, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr,
                "n_train": len(train_ds), "n_val": len(val_ds)}
    (args.output / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    log_f.close()
    log.info("Training complete. Best val_acc=%.3f. Output: %s", best_val_acc, args.output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", type=Path, default=Path("data/omnidocbench/docsim_triplets/manifest.jsonl"))
    p.add_argument("--data-root", type=Path, default=Path("data/omnidocbench"),
                   help="Root prepended to relative image paths in the manifest "
                        "(use '.' if paths are already absolute or repo-root-relative)")
    p.add_argument("--output", type=Path, default=Path("models/docsim_lora"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.triplets.exists():
        log.error("Triplets manifest not found: %s. Run scripts/build_docsim_triplets.py first.",
                  args.triplets)
        return 1

    cfg = DocSimConfig()
    train(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
