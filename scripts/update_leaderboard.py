#!/usr/bin/env python3
"""Update results/leaderboard.json from a technique's correlation results.

Usage:
    uv run python scripts/update_leaderboard.py --technique-yaml methods/baseline.yaml
"""
import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: uv add pyyaml")

PROJECT = Path(__file__).parent.parent
LEADERBOARD = PROJECT / "results" / "leaderboard.json"

RANKING_CONFIG = {
    "ocr_text":         {"primary_dim": "text_accuracy", "rank_by": "spearman"},
    "ocr_formula":      {"primary_dim": "formula_edit",  "rank_by": "spearman"},
    "ocr_table":        {"primary_dim": "table",         "rank_by": "spearman"},
    "ocr_all":          {"primary_dim": "end_to_end",    "rank_by": "spearman"},
    "ocr_all_no_mask":  {"primary_dim": "end_to_end",    "rank_by": "spearman"},
}
VARIANTS = list(RANKING_CONFIG.keys())
METRICS = ["multi_composite", "clip_cosine"]


def _nan_to_none(obj):
    """Recursively replace float NaN with None for JSON serialisation."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _load_leaderboard() -> dict:
    if not LEADERBOARD.exists():
        return {
            "last_updated": None,
            "ranking_config": RANKING_CONFIG,
            "variants": {v: {"best": {}, "history": []} for v in VARIANTS},
        }
    with open(LEADERBOARD) as f:
        return json.load(f)


def _save_leaderboard(lb: dict) -> None:
    lb["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = LEADERBOARD.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(lb, f, indent=2)
    os.replace(tmp, LEADERBOARD)


def _code_hash() -> str:
    """Short string identifying current code state for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT, timeout=5,
        )
        if result.returncode == 0:
            h = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "diff", "--quiet", "HEAD"],
                capture_output=True, cwd=PROJECT, timeout=5,
            ).returncode != 0
            return f"{h}-dirty" if dirty else h
    except Exception:
        pass
    return "unknown"


def _snapshot_leaderboard() -> None:
    """Write a timestamped copy of the leaderboard to results/snapshots/."""
    if not LEADERBOARD.exists():
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    snap_dir = LEADERBOARD.parent / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    shutil.copy2(LEADERBOARD, snap_dir / f"leaderboard-{ts}.json")


def _get_pearson(corr_data: dict, metric: str, dim: str):
    try:
        val = corr_data["correlations"][metric][dim]["pearson"]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return val
    except (KeyError, TypeError):
        return None


def _get_spearman(corr_data: dict, metric: str, dim: str):
    try:
        val = corr_data["correlations"][metric][dim]["spearman"]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return val
    except (KeyError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--technique-yaml", required=True, type=Path)
    args = parser.parse_args()

    spec = yaml.safe_load(args.technique_yaml.read_text())
    technique_id = spec["id"]
    date_str = spec.get("date") or datetime.now().strftime("%Y-%m-%d")
    if hasattr(date_str, "isoformat"):
        date_str = date_str.isoformat()

    code_hash = _code_hash()

    lb = _load_leaderboard()
    updated_variants = []
    new_bests = []

    for var_key in VARIANTS:
        variant_short = var_key.replace("ocr_", "")
        corr_path_template = spec.get("correlations_path", "")
        corr_path = PROJECT / corr_path_template.replace("{variant}", variant_short)

        if not corr_path.exists():
            print(f"  SKIP {var_key}: {corr_path} not found")
            continue

        with open(corr_path) as f:
            corr_data = _nan_to_none(json.load(f))

        primary_dim = RANKING_CONFIG[var_key]["primary_dim"]

        # Build stripped correlations blob (exclude lm_composite)
        stripped = {}
        for m in METRICS:
            if m in corr_data.get("correlations", {}):
                stripped[m] = corr_data["correlations"][m]

        # Strip per-category breakdown the same way (keep only METRICS)
        stripped_by_cat = {}
        for cat, cat_blob in (corr_data.get("by_category") or {}).items():
            if not isinstance(cat_blob, dict):
                continue
            cat_corrs = cat_blob.get("correlations") or {}
            cat_stripped = {m: cat_corrs[m] for m in METRICS if m in cat_corrs}
            stripped_by_cat[cat] = {
                "n_pages": cat_blob.get("n_pages"),
                "correlations": cat_stripped,
            }

        history_entry = {
            "technique_id": technique_id,
            "date": str(date_str),
            "code_hash": code_hash,
            "correlations": stripped,
            "by_category": stripped_by_cat,
        }

        var_state = lb["variants"].setdefault(var_key, {"best": {}, "history": []})
        var_state["history"].append(history_entry)

        # Update best_pearson and best_spearman per metric
        for metric in METRICS:
            pearson = _get_pearson({"correlations": stripped}, metric, primary_dim)
            spearman = _get_spearman({"correlations": stripped}, metric, primary_dim)
            n_samples = 0
            try:
                n_samples = stripped[metric][primary_dim]["n_samples"]
            except (KeyError, TypeError):
                pass

            if pearson is not None:
                current_best = var_state["best"].get(metric, {})
                current_pearson = current_best.get("pearson")
                # >= (not >) so a later technique that ties the current best
                # displaces the attribution. Reason: when a refinement reuses an
                # earlier method's signal for some variants (e.g. P2_060b reuses
                # P2_060's DocSim cc for {all, all_no_mask}), the newer/KEEP
                # technique should claim the credit, not the older/REVERT one.
                if current_pearson is None or pearson >= current_pearson:
                    old_str = f"{current_pearson:+.4f}" if current_pearson is not None else "none"
                    new_bests.append(
                        f"  NEW BEST Pearson {var_key} / {metric} / {primary_dim}: "
                        f"{old_str} → {pearson:+.4f} (technique={technique_id})"
                    )
                    var_state["best"][metric] = {
                        "technique_id": technique_id,
                        "pearson": pearson,
                        "spearman": spearman,
                        "n_samples": n_samples,
                    }

            if spearman is not None:
                current_best_s = var_state.setdefault("best_spearman", {}).get(metric, {})
                current_spearman = current_best_s.get("spearman")
                # >= (see Pearson block above) — newer ties displace older.
                if current_spearman is None or spearman >= current_spearman:
                    old_str = f"{current_spearman:+.4f}" if current_spearman is not None else "none"
                    new_bests.append(
                        f"  NEW BEST Spearman {var_key} / {metric} / {primary_dim}: "
                        f"{old_str} → {spearman:+.4f} (technique={technique_id})"
                    )
                    var_state["best_spearman"][metric] = {
                        "technique_id": technique_id,
                        "pearson": pearson,
                        "spearman": spearman,
                        "n_samples": n_samples,
                    }

        # Recompute overall best (Pearson) across all metrics
        best_p, best_metric = None, None
        for m in METRICS:
            p = var_state["best"].get(m, {}).get("pearson")
            if p is not None and (best_p is None or p > best_p):
                best_p, best_metric = p, m
        if best_metric:
            entry = var_state["best"][best_metric]
            var_state["best"]["overall"] = {
                "technique_id": entry["technique_id"],
                "metric": best_metric,
                "pearson": entry["pearson"],
                "spearman": entry["spearman"],
                "n_samples": entry["n_samples"],
            }

        # Recompute overall best (Spearman) across all metrics
        best_s, best_s_metric = None, None
        for m in METRICS:
            s = var_state.get("best_spearman", {}).get(m, {}).get("spearman")
            if s is not None and (best_s is None or s > best_s):
                best_s, best_s_metric = s, m
        if best_s_metric:
            entry = var_state["best_spearman"][best_s_metric]
            var_state["best_spearman"]["overall"] = {
                "technique_id": entry["technique_id"],
                "metric": best_s_metric,
                "pearson": entry["pearson"],
                "spearman": entry["spearman"],
                "n_samples": entry["n_samples"],
            }

        updated_variants.append(var_key)

    _save_leaderboard(lb)
    _snapshot_leaderboard()

    print(f"Updated leaderboard for: {', '.join(updated_variants) if updated_variants else 'none'}")
    if new_bests:
        for line in new_bests:
            print(line)
    else:
        print("  No new bests (scores not strictly higher than existing records).")
    print(f"Leaderboard saved to {LEADERBOARD}")


if __name__ == "__main__":
    main()
