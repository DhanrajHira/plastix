#!/usr/bin/env python3
"""Plot per-phase runtime cost from bench_phase_isolated.

Reads build/benchmarks/_outputs/phase_isolated.json (Google Benchmark
JSON output from bench_phase_isolated). One subplot per (In, Hidden)
shape; one line per phase (Forward, Loss, Backward, UpdateConn,
PruneConn, AddConn, CompactConn, Reset) plotted against density.

CompactConn rows carry an extra `dead_frac` counter and are expanded to
one series per dead fraction (e.g. "CompactConn@50%").
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PHASE_STYLE = {
    "Forward":     {"color": "#1f77b4", "marker": "o"},
    "Loss":        {"color": "#ff7f0e", "marker": "s"},
    "Backward":    {"color": "#2ca02c", "marker": "^"},
    "UpdateConn":  {"color": "#d62728", "marker": "D"},
    "PruneConn":   {"color": "#9467bd", "marker": "v"},
    "AddConn":     {"color": "#8c564b", "marker": "P"},
    "CompactConn": {"color": "#e377c2", "marker": "X"},
    "Reset":       {"color": "#7f7f7f", "marker": "*"},
}


def to_microseconds(t: float, unit: str) -> float:
    if unit == "ns":
        return t / 1000.0
    if unit == "us":
        return t
    if unit == "ms":
        return t * 1000.0
    if unit == "s":
        return t * 1_000_000.0
    raise ValueError(f"unexpected time_unit: {unit}")


def parse_row(b: dict) -> tuple[str, int, int, float, float] | None:
    """Returns (series_label, In, Hidden, density, time_us) or None."""
    phase = b["name"].split("/", 1)[0]
    if phase not in PHASE_STYLE:
        return None
    in_dim = int(b["In"])
    hidden = int(b["Hidden"])
    density = float(b["density"])
    t = to_microseconds(float(b["real_time"]), b.get("time_unit", "ns"))
    label = phase
    if phase == "CompactConn" and "dead_frac" in b:
        label = f"CompactConn@{int(round(float(b['dead_frac']) * 100))}%"
    return label, in_dim, hidden, density, t


def load(path: Path):
    """{(In, Hidden): {label: [(density, time_us), ...]}}."""
    with path.open() as f:
        data = json.load(f)
    by_shape: dict = defaultdict(lambda: defaultdict(list))
    for b in data["benchmarks"]:
        row = parse_row(b)
        if row is None:
            continue
        label, in_dim, hidden, density, t = row
        by_shape[(in_dim, hidden)][label].append((density, t))
    for shape, series in by_shape.items():
        for label, pts in series.items():
            pts.sort(key=lambda p: p[0])
    return by_shape


def style_for(label: str) -> dict:
    if label in PHASE_STYLE:
        return PHASE_STYLE[label]
    # CompactConn variants share the CompactConn color but vary the marker.
    base = label.split("@", 1)[0]
    return PHASE_STYLE.get(base, {"color": "#000000", "marker": "o"})


# Floor below which a series is treated as below-resolution noise and
# dropped from the plot (in microseconds). 1e-3 µs == 1 ns.
NOISE_FLOOR_US = 1e-3


def plot(by_shape, out_png: Path, out_svg: Path | None = None,
         keep_noise: bool = False) -> None:
    shapes = sorted(by_shape.keys(), key=lambda s: (s[0] * s[1], s[0], s[1]))
    n = len(shapes)
    ncols = min(2, n) if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.0 * ncols, 5.0 * nrows),
                             sharey=True, squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, shape in zip(flat, shapes):
        in_dim, hidden = shape
        series_dict = by_shape[shape]
        for label in sorted(series_dict.keys()):
            pts = series_dict[label]
            if not keep_noise and max(p[1] for p in pts) < NOISE_FLOOR_US:
                continue
            xs = [p[0] * 100.0 for p in pts]  # density as percent
            ys = [p[1] for p in pts]
            style = style_for(label)
            ax.plot(xs, ys, label=label, linewidth=1.8,
                    marker=style["marker"], markersize=6.0,
                    color=style["color"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{in_dim} × {hidden}", fontsize=12)
        ax.set_xlabel("density (%)", fontsize=10)
        ax.set_ylabel("time (µs)", fontsize=10)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=9)

    for ax in flat[n:]:
        ax.set_visible(False)

    # Build one shared legend across all subplots.
    handles, labels = [], []
    for ax in flat[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(labels), 6), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Plastix per-phase runtime vs density (log–log)",
                 fontsize=14, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_png}")
    if out_svg is not None:
        print(f"wrote {out_svg}")


def default_input() -> Path:
    repo = Path(__file__).resolve().parent.parent
    return repo / "build" / "benchmarks" / "_outputs" / "phase_isolated.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=default_input(),
                    help="Path to phase_isolated.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: next to input)")
    ap.add_argument("--svg", action="store_true",
                    help="Also write an SVG next to the PNG")
    ap.add_argument("--keep-noise", action="store_true",
                    help="Keep sub-resolution series (e.g. Reset) that would "
                         "otherwise be filtered out")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"input not found: {args.input}\n"
            f"run bench_phase_isolated first")
    out_png = args.out or args.input.with_name("phase_isolated.png")
    out_svg = out_png.with_suffix(".svg") if args.svg else None

    by_shape = load(args.input)
    if not by_shape:
        raise SystemExit("no matching benchmark rows found in JSON")
    plot(by_shape, out_png, out_svg, keep_noise=args.keep_noise)


if __name__ == "__main__":
    main()
