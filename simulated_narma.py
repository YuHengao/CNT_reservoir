"""Model-based NARMA2/5/10 benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


SEED = 20260824
BIASES_MV = tuple(range(1, 11))
SOURCE_NODES = 96
NODES_PER_CHANNEL = 6
WASHOUT, LEARNING, VALIDATION, TEST = 1000, 3000, 1000, 1000
TOTAL = WASHOUT + LEARNING + VALIDATION + TEST
ALPHAS = np.logspace(-9, 4, 14)


def make_input() -> tuple[np.ndarray, np.ndarray]:
    """Return u(k) in [0, 0.5] and its UV-power encoding (percent)."""
    u = np.random.default_rng(SEED).uniform(0.0, 0.5, TOTAL)
    return u, 10.0 + 180.0 * u


def narma_target(u: np.ndarray, order: int) -> np.ndarray:
    """Return y[0..TOTAL], where features at k predict y[k+1]."""
    u = np.asarray(u, dtype=float)
    if order not in (2, 5, 10):
        raise ValueError("Supported NARMA orders are 2, 5 and 10")
    y = np.zeros(len(u) + 1, dtype=float)
    if order == 2:
        for k in range(1, len(u)):
            y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * u[k] ** 3 + 0.1
    else:
        for k in range(order - 1, len(u)):
            y[k + 1] = (
                0.3 * y[k]
                + 0.05 * y[k] * np.sum(y[k - order + 1 : k + 1])
                + 1.5 * u[k - order + 1] * u[k]
                + 0.1
            )
    return y


def sampled_state(curves_by_bias: dict[int, np.ndarray]) -> np.ndarray:
    """Peak plus five within-cycle samples for each of ten bias channels."""
    positions = (np.arange(NODES_PER_CHANNEL - 1) + 0.5) * SOURCE_NODES / (NODES_PER_CHANNEL - 1)
    indices = np.floor(positions).astype(int)
    blocks = []
    for bias in BIASES_MV:
        curves = np.asarray(curves_by_bias[bias], dtype=float)
        if curves.shape != (TOTAL, SOURCE_NODES):
            raise ValueError(f"Bias {bias} mV must have shape {(TOTAL, SOURCE_NODES)}")
        blocks.append(np.column_stack((np.max(curves, axis=1), curves[:, indices])))
    return np.hstack(blocks)


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target - prediction) ** 2) / np.var(target))


def fit_readout(state: np.ndarray, target: np.ndarray, alpha: float, *, selection: bool = False):
    mean, scale = state.mean(axis=0), state.std(axis=0)
    keep = scale > 1e-13
    normalised = (state[:, keep] - mean[keep]) / scale[keep]
    target_mean = float(target.mean())
    ridge = Ridge(
        alpha=float(alpha),
        fit_intercept=False,
        solver="lsqr",
        tol=1e-6 if selection else 1e-7,
        max_iter=1200 if selection else 2500,
    )
    ridge.fit(normalised, target - target_mean)
    return mean, scale, keep, target_mean, ridge


def predict_readout(model, state: np.ndarray) -> np.ndarray:
    mean, scale, keep, target_mean, ridge = model
    normalised = (state[:, keep] - mean[keep]) / scale[keep]
    return target_mean + ridge.predict(normalised)


def evaluate(curves_by_bias: dict[int, np.ndarray]) -> dict[int, dict]:
    u, _ = make_input()
    state = sampled_state(curves_by_bias)
    train = slice(WASHOUT, WASHOUT + LEARNING)
    validation = slice(WASHOUT + LEARNING, WASHOUT + LEARNING + VALIDATION)
    development = slice(WASHOUT, WASHOUT + LEARNING + VALIDATION)
    test = slice(WASHOUT + LEARNING + VALIDATION, TOTAL)
    results = {}
    for order in (2, 5, 10):
        target = narma_target(u, order)
        best = None
        for alpha in ALPHAS:
            readout = fit_readout(state[train], target[WASHOUT + 1 : WASHOUT + LEARNING + 1], alpha, selection=True)
            predicted = predict_readout(readout, state[validation])
            actual = target[WASHOUT + LEARNING + 1 : WASHOUT + LEARNING + VALIDATION + 1]
            candidate = (nmse(actual, predicted), -float(alpha))
            if best is None or candidate < best[0]:
                best = (candidate, float(alpha))
        alpha = best[1]
        readout = fit_readout(state[development], target[WASHOUT + 1 : WASHOUT + LEARNING + VALIDATION + 1], alpha)
        actual = target[WASHOUT + LEARNING + VALIDATION + 1 : TOTAL + 1]
        predicted = predict_readout(readout, state[test])
        results[order] = {
            "selected_ridge_alpha": alpha,
            "validation_nmse": best[0][0],
            "test_nmse": nmse(actual, predicted),
            "test_target": actual,
            "test_prediction": predicted,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves-npz", required=True, type=Path, help="NPZ with bias_1 ... bias_10 arrays, each (6000, 96)")
    args = parser.parse_args()
    with np.load(args.curves_npz, allow_pickle=False) as archive:
        curves = {bias: archive[f"bias_{bias}"] for bias in BIASES_MV}
    results = evaluate(curves)
    print(json.dumps({f"NARMA{order}": {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
                      for order, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
