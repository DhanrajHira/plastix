#!/usr/bin/env python3
"""Plot Plastix vs TorchDense vs TorchSparseCsr runtimes as a function
of sparsity.

Reads build/benchmarks/_outputs/spmv_crossover_torch.json (Google
Benchmark JSON output from bench_spmv_crossover_torch). One subplot
per (In, Out) shape; three lines per subplot, one per kernel.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


KERNEL_STYLE = {
    "Plastix":        {"color": "#1f77b4", "marker": "o"},
    "TorchDense":     {"color": "#d62728", "marker": "s"},
    "TorchSparseCsr": {"color": "#2ca02c", "marker": "^"},
}


def parse_row(b: dict) -> tuple[str, int, int, float, float] | None:
    """Returns (kernel, In, Out, sparsity_pct, time_us) or None on miss."""
    parts = b["name"].split("/")
    kernel = parts[0]
    if kernel not in KERNEL_STYLE:
        return None
    in_dim = int(b["In"])
    out_dim = int(b["Out"])
    density = float(b["density"])
    sparsity_pct = 100.0 * (1.0 - density)
    # Google Benchmark may report ns/us/ms; normalize to microseconds.
    unit = b.get("time_unit", "ns")
    t = float(b["real_time"])
    if unit == "ns":
        t /= 1000.0
    elif unit == "ms":
        t *= 1000.0
    elif unit != "us":
        raise ValueError(f"unexpected time_unit: {unit}")
    return kernel, in_dim, out_dim, sparsity_pct, t


def load(path: Path) -> dict[tuple[int, int], dict[str, list[tuple[float, float]]]]:
    """Returns {(In, Out): {kernel: [(sparsity_pct, time_us), ...]}}."""
    with path.open() as f:
        data = json.load(f)
    by_shape: dict = defaultdict(lambda: defaultdict(list))
    for b in data["benchmarks"]:
        row = parse_row(b)
        if row is None:
            continue
        kernel, in_dim, out_dim, sparsity, t = row
        by_shape[(in_dim, out_dim)][kernel].append((sparsity, t))
    # Sort each kernel's series by sparsity so the line connects in order.
    for shape, kernels in by_shape.items():
        for kernel, series in kernels.items():
            series.sort(key=lambda p: p[0])
    return by_shape


def plot(by_shape, out_png: Path, out_svg: Path | None = None) -> None:
    shapes = sorted(by_shape.keys(), key=lambda s: (s[0] * s[1], s[0], s[1]))
    n = len(shapes)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows),
                             sharey=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, shape in zip(axes, shapes):
        in_dim, out_dim = shape
        kernels = by_shape[shape]
        for kernel, series in kernels.items():
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            style = KERNEL_STYLE[kernel]
            ax.plot(xs, ys, label=kernel, linewidth=1.6,
                    marker=style["marker"], markersize=4.5,
                    color=style["color"])
        ax.set_yscale("log")
        ax.set_title(f"{in_dim} × {out_dim}", fontsize=10)
        ax.set_xlabel("sparsity (%)", fontsize=9)
        ax.set_ylabel("time (µs)", fontsize=9)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=8)

    # Hide unused axes.
    for ax in axes[n:]:
        ax.set_visible(False)

    # Single shared legend, top of the figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("SpMV/GEMV: runtime vs sparsity (per shape, log y)",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_png}")
    if out_svg is not None:
        print(f"wrote {out_svg}")


def default_input() -> Path:
    # benchmarks/plot_spmv_crossover.py  ->  build/benchmarks/_outputs/...
    repo = Path(__file__).resolve().parent.parent
    return repo / "build" / "benchmarks" / "_outputs" / "spmv_crossover_torch.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=default_input(),
                    help="Path to spmv_crossover_torch.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: next to input)")
    ap.add_argument("--svg", action="store_true",
                    help="Also write an SVG next to the PNG")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"input not found: {args.input}\n"
            f"run bench_spmv_crossover_torch first")
    out_png = args.out or args.input.with_name("spmv_crossover_torch.png")
    out_svg = out_png.with_suffix(".svg") if args.svg else None

    by_shape = load(args.input)
    if not by_shape:
        raise SystemExit("no matching benchmark rows found in JSON")
    plot(by_shape, out_png, out_svg)


if __name__ == "__main__":
    main()
