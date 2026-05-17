# conda-env: mtech-env
"""Generate a cohort flowchart from pipeline cohort_counts.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_counts(counts_path: Path):
    with open(counts_path, 'r') as f:
        return json.load(f)


def render_flowchart(counts, output_path: Path) -> None:
    if not counts:
        raise ValueError("No cohort count steps found")

    fig_height = max(4, len(counts) * 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis('off')

    y_positions = list(reversed(range(len(counts))))
    for idx, (step, y) in enumerate(zip(counts, y_positions)):
        label = (
            f"{step['step_name']}\n"
            f"Rows in: {step['rows_in']} | Rows out: {step['rows_out']}\n"
            f"Dropped: {step['dropped']}"
        )
        ax.text(
            0.5,
            y,
            label,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef5ff', edgecolor='#336699'),
        )
        if idx < len(counts) - 1:
            ax.annotate(
                '',
                xy=(0.5, y - 0.4),
                xytext=(0.5, y - 0.9),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(counts))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render cohort flowchart")
    parser.add_argument(
        "counts_json",
        help="Path to aggregated_results/cohort_counts.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: cohort_flowchart.png next to counts JSON)",
    )
    args = parser.parse_args()

    counts_path = Path(args.counts_json)
    output = Path(args.output) if args.output else counts_path.with_name("cohort_flowchart.png")
    render_flowchart(load_counts(counts_path), output)
