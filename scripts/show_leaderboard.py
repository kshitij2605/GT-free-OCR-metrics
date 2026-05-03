#!/usr/bin/env python3
"""Display results/leaderboard.json in a readable terminal table.

Usage:
    uv run python scripts/show_leaderboard.py
    uv run python scripts/show_leaderboard.py --variant formula
    uv run python scripts/show_leaderboard.py --history
    uv run python scripts/show_leaderboard.py --top 5
    uv run python scripts/show_leaderboard.py --history --rank-by pearson
    uv run python scripts/show_leaderboard.py --history --rank-by divergence
    uv run python scripts/show_leaderboard.py --summary
    uv run python scripts/show_leaderboard.py --history --category metric_new
    uv run python scripts/show_leaderboard.py --history --min-spearman 0.35
    uv run python scripts/show_leaderboard.py --history --since 2026-04-24
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LEADERBOARD = PROJECT / "results" / "leaderboard.json"
METHODS_DIR = PROJECT / "methods"

RANKING_CONFIG = {
    "ocr_text":    {"primary_dim": "text_accuracy"},
    "ocr_formula": {"primary_dim": "formula_edit"},
    "ocr_table":   {"primary_dim": "table"},
    "ocr_all":     {"primary_dim": "end_to_end"},
}
METRICS = ["multi_composite", "clip_cosine"]


def _fmt(val) -> str:
    if val is None:
        return "  n/a  "
    return f"{val:+.4f}"


def _load_category_map() -> dict:
    """Return {technique_id: category} from all YAMLs in methods/."""
    cat_map = {}
    try:
        import yaml
        for f in METHODS_DIR.glob("*.yaml"):
            try:
                spec = yaml.safe_load(f.read_text())
                if spec and "id" in spec:
                    cat_map[spec["id"]] = spec.get("category", "unknown")
            except Exception:
                pass
    except ImportError:
        pass
    return cat_map


def _rows_for_variant(
    var_state: dict,
    var_key: str,
    top_n: int,
    rank_by: str = "spearman",
    category: str = None,
    category_map: dict = None,
    min_spearman: float = None,
    min_pearson: float = None,
    since: str = None,
) -> list:
    """Return deduped, sorted, filtered history rows for the variant.

    Dedup: keep only the best-scoring entry per (technique, metric) pair.
    rank_by: "pearson", "spearman", or "divergence" (largest |P-S|).
    """
    primary_dim = RANKING_CONFIG[var_key]["primary_dim"]
    best: dict = {}  # (technique, metric) -> row
    for entry in var_state.get("history", []):
        for metric in METRICS:
            try:
                dim_data = entry["correlations"][metric][primary_dim]
                p = dim_data.get("pearson")
                s = dim_data.get("spearman")
                n = dim_data.get("n_samples", 0)
                sp = dim_data.get("spearman_pvalue")
            except (KeyError, TypeError):
                continue
            key = (entry["technique_id"], metric)
            existing = best.get(key)
            if rank_by == "pearson":
                score = p
                existing_score = existing["pearson"] if existing else None
            elif rank_by == "spearman":
                score = s
                existing_score = existing["spearman"] if existing else None
            else:  # divergence
                score = abs(p - s) if (p is not None and s is not None) else None
                ep = existing["pearson"] if existing else None
                es = existing["spearman"] if existing else None
                existing_score = abs(ep - es) if (ep is not None and es is not None) else None
            if existing is None or (score is not None and (existing_score is None or score > existing_score)):
                best[key] = {
                    "technique": entry["technique_id"],
                    "metric": metric,
                    "pearson": p,
                    "spearman": s,
                    "n": n,
                    "spearman_pvalue": sp,
                    "date": entry.get("date", ""),
                }

    rows = list(best.values())

    # Apply filters
    if category is not None and category_map is not None:
        rows = [r for r in rows if category_map.get(r["technique"], "unknown") == category]
    if min_spearman is not None:
        rows = [r for r in rows if r["spearman"] is not None and r["spearman"] >= min_spearman]
    if min_pearson is not None:
        rows = [r for r in rows if r["pearson"] is not None and r["pearson"] >= min_pearson]
    if since is not None:
        rows = [r for r in rows if r["date"] >= since]

    if rank_by == "pearson":
        rows.sort(key=lambda r: (r["pearson"] is None, -(r["pearson"] or 0)))
    elif rank_by == "spearman":
        rows.sort(key=lambda r: (r["spearman"] is None, -(r["spearman"] or 0)))
    else:  # divergence
        def _div_key(r):
            p, s = r["pearson"], r["spearman"]
            if p is None or s is None:
                return (True, 0.0)
            return (False, -(abs(p - s)))
        rows.sort(key=_div_key)

    return rows[:top_n] if top_n else rows


def _print_table(rows: list, rank_label: str) -> None:
    """Print a single ranked table with Pearson, Spearman, and delta columns."""
    print(f"  --- Ranked by {rank_label} ---")
    print(f"  {'#':<3}  {'technique':<32}  {'metric':<18}  {'Pearson':>8}  "
          f"{'Spearman':>9}  {'p-val':>7}  {'D(P-S)':>8}  {'!':>1}  {'N':>6}  {'date'}")
    print(f"  {'-'*3}  {'-'*32}  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*1}  {'-'*6}  {'-'*10}")
    for i, r in enumerate(rows, 1):
        p, s = r["pearson"], r["spearman"]
        if p is not None and s is not None:
            delta = p - s
            delta_str = f"{delta:+.4f}"
            flag = "!" if abs(delta) > 0.10 else " "
        else:
            delta_str = "  n/a  "
            flag = " "
        sp_p = r.get("spearman_pvalue")
        pval_str = f"{sp_p:.4f}" if sp_p is not None else "  n/a "
        print(f"  {i:<3}  {r['technique']:<32}  {r['metric']:<18}  "
              f"{_fmt(p):>8}  {_fmt(s):>9}  {pval_str:>7}  {delta_str:>8}  {flag:>1}  "
              f"{r['n']:>6}  {r['date']}")


def print_variant(
    var_key: str,
    var_state: dict,
    show_history: bool,
    top_n: int,
    rank_by: str = "spearman",
    category: str = None,
    category_map: dict = None,
    min_spearman: float = None,
    min_pearson: float = None,
    since: str = None,
) -> None:
    primary_dim = RANKING_CONFIG[var_key]["primary_dim"]
    print()
    print("=" * 88)
    print(f"  {var_key.upper()}   (primary ref: {primary_dim})")
    print("=" * 88)

    best_p = var_state.get("best", {})
    best_s = var_state.get("best_spearman", {})
    ov_p = best_p.get("overall", {})
    ov_s = best_s.get("overall", {})
    print(f"  BEST Pearson:  technique={ov_p.get('technique_id', 'none'):<28} "
          f"metric={ov_p.get('metric', '-'):<18} Pearson={_fmt(ov_p.get('pearson'))}")
    print(f"  BEST Spearman: technique={ov_s.get('technique_id', 'none'):<28} "
          f"metric={ov_s.get('metric', '-'):<18} Spearman={_fmt(ov_s.get('spearman'))}")
    for m in METRICS:
        bp = best_p.get(m, {})
        bs = best_s.get(m, {})
        print(f"  best_pearson/{m:<14} technique={bp.get('technique_id', 'none'):<22} "
              f"Pearson={_fmt(bp.get('pearson'))}  Spearman={_fmt(bp.get('spearman'))}")
        print(f"  best_spearman/{m:<13} technique={bs.get('technique_id', 'none'):<22} "
              f"Pearson={_fmt(bs.get('pearson'))}  Spearman={_fmt(bs.get('spearman'))}")

    if show_history:
        rows = _rows_for_variant(
            var_state, var_key, top_n, rank_by=rank_by,
            category=category, category_map=category_map,
            min_spearman=min_spearman, min_pearson=min_pearson, since=since,
        )
        if not rows:
            print("  (no matching history)")
            return
        print()
        _print_table(rows, rank_by)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["text", "formula", "table", "all"])
    parser.add_argument("--history", action="store_true", help="Show full history table")
    parser.add_argument("--summary", action="store_true",
                        help="Show top-3 history per variant (shorthand for --history --top 3)")
    parser.add_argument("--top", type=int, default=0, help="Limit history rows per variant")
    parser.add_argument("--rank-by", choices=["pearson", "spearman", "divergence"], default="spearman",
                        help="Sort history table by this coefficient (default: spearman)")
    parser.add_argument("--category", help="Filter history to methods with this YAML category")
    parser.add_argument("--min-spearman", type=float, help="Hide rows with Spearman below this threshold")
    parser.add_argument("--min-pearson", type=float, help="Hide rows with Pearson below this threshold")
    parser.add_argument("--since", help="Hide rows with date before YYYY-MM-DD")
    args = parser.parse_args()

    # --summary is shorthand for --history --top 3
    show_history = args.history or args.summary
    top_n = args.top if args.top else (3 if args.summary and not args.top else 0)

    if not LEADERBOARD.exists():
        print(f"No leaderboard found at {LEADERBOARD}")
        print("Run: uv run python scripts/update_leaderboard.py --technique-yaml methods/baseline.yaml")
        return

    with open(LEADERBOARD) as f:
        lb = json.load(f)

    category_map = _load_category_map() if args.category else {}

    print(f"Leaderboard — last updated: {lb.get('last_updated', 'unknown')}")

    target_vars = [f"ocr_{args.variant}"] if args.variant else list(RANKING_CONFIG.keys())
    for var_key in target_vars:
        var_state = lb.get("variants", {}).get(var_key, {"best": {}, "history": []})
        print_variant(
            var_key, var_state, show_history, top_n,
            rank_by=args.rank_by,
            category=args.category,
            category_map=category_map,
            min_spearman=args.min_spearman,
            min_pearson=args.min_pearson,
            since=args.since,
        )

    print()


if __name__ == "__main__":
    main()
