"""Compare Dreamer variants on their training logs, VRU safety first.

Each variant writes a per-episode CSV (training.logger.Logger). Point this
script at those CSVs to get a side-by-side KPI table and, optionally, a
grouped-bar chart of the headline scores.

Usage:
    # explicit label=path pairs
    python -m scripts.compare_dreamers \
        baseline=logs/baseline.csv \
        dreamer=logs/dreamer.csv \
        sdbs=logs/sdbs.csv

    # or bare paths (label inferred from filename)
    python -m scripts.compare_dreamers logs/baseline.csv logs/sdbs.csv

    # judge the whole run instead of the converged tail, and plot:
    python -m scripts.compare_dreamers --tail 1.0 --plot logs/plots/kpi.png \
        baseline=logs/baseline.csv sdbs=logs/sdbs.csv
"""
import argparse
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.kpi import compare_variants, format_comparison_table  # noqa: E402


def _parse_logs(items):
    """Turn ``label=path`` / bare ``path`` args into an ordered label->path map."""
    named = OrderedDict()
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = os.path.splitext(os.path.basename(path))[0]
        named[label] = path
    return named


def _plot(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    metrics = [
        ("VRU safety", "vru_safety_score"),
        ("Vehicle safety", "vehicle_safety_score"),
        ("Comfort", "comfort_score"),
        ("Composite", "composite_score"),
    ]
    n_groups = len(metrics)
    width = 0.8 / max(1, len(labels))
    plt.figure(figsize=(9, 5))
    for i, label in enumerate(labels):
        xs = [g + i * width for g in range(n_groups)]
        ys = [results[label][key] for _, key in metrics]
        plt.bar(xs, ys, width=width, label=label)
    centers = [g + width * (len(labels) - 1) / 2 for g in range(n_groups)]
    plt.xticks(centers, [name for name, _ in metrics])
    plt.ylabel("Score (0-100, higher = better)")
    plt.title("Dreamer variants — headline KPIs (VRU safety prioritised)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Dreamer variants by KPI")
    parser.add_argument("logs", nargs="+",
                        help="CSV logs as 'label=path' or bare 'path'")
    parser.add_argument("--tail", type=float, default=0.25,
                        help="tail fraction (0-1) or episode count to evaluate; "
                             "use 1.0 for the whole run (default: 0.25)")
    parser.add_argument("--tau-ttc", type=float, default=2.0,
                        help="time-to-collision horizon for the TTC penalty")
    parser.add_argument("--plot", default=None,
                        help="optional PNG path for a headline-score bar chart")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    named = _parse_logs(args.logs)
    results = compare_variants(named, tail=args.tail, tau_ttc=args.tau_ttc)
    print(format_comparison_table(results))
    if args.plot:
        _plot(results, args.plot)


if __name__ == "__main__":
    main()
