from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ignisca.evaluation.aggregate import aggregate_cell, collect_runs
from ignisca.evaluation.failure import rank_failures
from ignisca.evaluation.reporting import render_headline_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation runs and render headline table")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/ablation.md"))
    parser.add_argument("--also-failures", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--failures-out", type=Path, default=Path("reports/failures/"))
    args = parser.parse_args()

    results = collect_runs(args.runs_root)
    if not results:
        raise SystemExit(f"no eval.json files found under {args.runs_root}")

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for r in results:
        grouped[(r.fire_id, r.cell)].append(r)

    rows = []
    for (fire_id, cell), seed_results in sorted(grouped.items()):
        rows.append(aggregate_cell(cell=cell, fire_id=fire_id, results=seed_results))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_headline_table(rows))
    print(f"Wrote {args.out} with {len(rows)} rows")

    if args.also_failures:
        from ignisca.evaluation.failure import render_failure_case

        args.failures_out.mkdir(parents=True, exist_ok=True)
        for r in results:
            run_dir = args.runs_root / r.run_name
            sample_jsonl = run_dir / f"sample_metrics_{r.fire_id}.jsonl"
            npz_path = run_dir / f"predictions_{r.fire_id}.npz"
            if not sample_jsonl.exists() or not npz_path.exists():
                continue
            worst = rank_failures(sample_jsonl, k=args.top_k, metric="iou", mode="worst")
            for rank, row in enumerate(worst):
                out_png = args.failures_out / f"{r.run_name}_{r.fire_id}_worst{rank:02d}.png"
                render_failure_case(
                    npz_path=npz_path,
                    sample_idx=int(row["sample_idx"]),
                    out_path=out_png,
                )
        print(f"Wrote failure PNGs under {args.failures_out}")


if __name__ == "__main__":
    main()
