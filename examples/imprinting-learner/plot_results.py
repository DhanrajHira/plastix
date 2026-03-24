#!/usr/bin/env python3
"""Plot learning curves from imprinting learner test CSV output."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TESTS = [
    ("test_one_step_linear.csv", "Test 1: One-step linear (lag=1, gamma=0)"),
    ("test_one_step_pattern.csv", "Test 3: One-step pattern+memory (XOR lag=2, gamma=0)"),
    ("test_one_step_memory.csv", "Test 5: One-step memory (lag=4, gamma=0)"),
]


def bin_errors(errors: np.ndarray, n_bins: int = 100) -> np.ndarray:
    bin_size = len(errors) // n_bins
    return errors[: bin_size * n_bins].reshape(n_bins, bin_size).mean(axis=1)


def main():
    csv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir = csv_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    # --- Error learning curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Imprinting Learner — Return MSE Learning Curves", fontsize=14)

    for ax, (csv_name, title) in zip(axes, TESTS):
        path = csv_dir / csv_name
        if not path.exists():
            ax.set_title(f"{title}\n(not found)")
            continue

        df = pd.read_csv(path)
        n_bins = 100
        binned = bin_errors(df["error"].values, n_bins)
        x = np.linspace(0, len(df), n_bins)

        ax.plot(x, binned, color="#2196F3", linewidth=1.2)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE")
        ax.set_xlim(0, len(df))
        ax.set_ylim(bottom=-0.01)

    plt.tight_layout()
    fig.savefig(out_dir / "error_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'error_curves.png'}")

    # --- Prediction vs Return (late phase) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Predictions vs Returns (last 250 steps)", fontsize=14)

    for ax, (csv_name, title) in zip(axes, TESTS):
        path = csv_dir / csv_name
        if not path.exists():
            continue

        df = pd.read_csv(path)
        tail = df.iloc[-250:]

        ax.plot(tail["step"], tail["return"], color="#4CAF50", alpha=0.7,
                linewidth=1.2, label="True Return")
        ax.plot(tail["step"], tail["prediction"], color="#F44336", alpha=0.7,
                linewidth=1.0, label="Prediction")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "predictions_late.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'predictions_late.png'}")

    # --- Prediction vs Return (early phase) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Predictions vs Returns (first 250 steps)", fontsize=14)

    for ax, (csv_name, title) in zip(axes, TESTS):
        path = csv_dir / csv_name
        if not path.exists():
            continue

        df = pd.read_csv(path)
        head = df.iloc[:250]

        ax.plot(head["step"], head["return"], color="#4CAF50", alpha=0.7,
                linewidth=1.2, label="True Return")
        ax.plot(head["step"], head["prediction"], color="#F44336", alpha=0.7,
                linewidth=1.0, label="Prediction")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "predictions_early.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'predictions_early.png'}")


if __name__ == "__main__":
    main()
