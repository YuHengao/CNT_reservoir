"""Analyse measured states.

Input NPZ: u and bias_1 ... bias_10. Each bias array has 1000 cycles
, already aligned to the UV input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BIASES_MV = tuple(range(1, 11))
N_CYCLES, N_NODES = 1000, 6
TRAIN = slice(50, 750)
TEST = slice(750, 999)
RIDGE_ALPHA = 1000.0


def narma_target(u: np.ndarray, order: int) -> np.ndarray:
    """Target at cycle k is the one-step-ahead value y(k+1)."""
    if order not in (2, 5, 10):
        raise ValueError("Supported orders are 2, 5 and 10")
    y = np.zeros(len(u), dtype=float)
    for k in range(max(1, order - 1), len(u) - 1):
        if order == 2:
            y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * u[k] ** 3 + 0.1
        else:
            y[k + 1] = (
                0.3 * y[k]
                + 0.05 * y[k] * np.sum(y[k - order + 1 : k + 1])
                + 1.5 * u[k - order + 1] * u[k]
                + 0.1
            )
    return np.r_[y[1:], np.nan]


def reservoir_state(curves_by_bias: dict[int, np.ndarray]) -> np.ndarray:
    """Combine measured current and within-cycle relative change."""
    blocks = []
    for bias in BIASES_MV:
        current = np.asarray(curves_by_bias[bias], dtype=float)
        if current.shape != (N_CYCLES, N_NODES) or not np.all(np.isfinite(current)):
            raise ValueError(f"bias_{bias} must be a finite {(N_CYCLES, N_NODES)} array")
        relative = (current - current[:, [0]]) / (np.abs(current[:, [0]]) + 1e-18)
        blocks.append(np.hstack((current, relative)))
    return np.hstack(blocks)


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.var(target))
    if variance <= 0:
        raise ValueError("Target variance must be positive")
    return float(np.mean((target - prediction) ** 2) / variance)


def evaluate(u: np.ndarray, curves_by_bias: dict[int, np.ndarray]) -> dict[int, dict]:
    u = np.asarray(u, dtype=float)
    if u.shape != (N_CYCLES,) or not np.all(np.isfinite(u)) or np.any((u < 0) | (u > 0.5)):
        raise ValueError("u must contain 1000 finite values in [0, 0.5]")
    state = reservoir_state(curves_by_bias)
    mean, scale = state[TRAIN].mean(axis=0), state[TRAIN].std(axis=0)
    scale[scale == 0] = 1.0
    train_state = (state[TRAIN] - mean) / scale
    test_state = (state[TEST] - mean) / scale
    results = {}
    for order in (2, 5, 10):
        target = narma_target(u, order)
        train_target = target[TRAIN]
        target_mean = float(train_target.mean())
        gram = train_state.T @ train_state + RIDGE_ALPHA * np.eye(train_state.shape[1])
        weights = np.linalg.solve(gram, train_state.T @ (train_target - target_mean))
        prediction = target_mean + test_state @ weights
        results[order] = {
            "test_nmse": nmse(target[TEST], prediction),
            "test_target": target[TEST],
            "test_prediction": prediction,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-npz", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.states_npz, allow_pickle=False) as archive:
        u = archive["u"]
        curves = {bias: archive[f"bias_{bias}"] for bias in BIASES_MV}
    results = evaluate(u, curves)
    print(json.dumps({f"NARMA{order}": {"test_nmse": result["test_nmse"]}
                      for order, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
