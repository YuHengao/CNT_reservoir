"""Partially observed, closed-loop Lorenz-x task.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from sklearn.linear_model import Ridge


class ReservoirBank(Protocol):
    def step(self, uv_power_percent: float) -> np.ndarray: ...


SEED = 20260810
WASHOUT, LEARNING, VALIDATION, TEST = 500, 5000, 1000, 2000
TRAIN_END = WASHOUT + LEARNING
CLOSURE = TRAIN_END + VALIDATION
ALPHAS = np.logspace(-9, 4, 14)
THRESHOLD = 0.4


def lorenz_rhs(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    return np.array((10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z))


def lorenz_reference(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """RK4 integration dt=0.01, discard 2000 steps, sample every 0.02."""
    rng = np.random.default_rng(seed)
    initial = np.ones(3, dtype=float) + rng.normal(0.0, 0.015, 3)
    discard, stride = 2000, 2
    steps = discard + (CLOSURE + TEST + 2) * stride
    trajectory = np.empty((steps + 1, 3), dtype=float)
    trajectory[0] = initial
    dt = 0.01
    for k in range(steps):
        value = trajectory[k]
        a = lorenz_rhs(value)
        b = lorenz_rhs(value + 0.5 * dt * a)
        c = lorenz_rhs(value + 0.5 * dt * b)
        d = lorenz_rhs(value + dt * c)
        trajectory[k + 1] = value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
    raw_x = trajectory[discard::stride, 0][: CLOSURE + TEST + 1]
    scaled_x = (raw_x - raw_x.min()) / (raw_x.max() - raw_x.min())
    return raw_x, scaled_x


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target - prediction) ** 2) / np.var(target))


def fit_readout(state: np.ndarray, target: np.ndarray, alpha: float, *, selection: bool = False):
    mean, scale = state.mean(axis=0), state.std(axis=0)
    keep = scale > 1e-13
    z = (state[:, keep] - mean[keep]) / scale[keep]
    y_mean = float(target.mean())
    ridge = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr",
                  tol=1e-6 if selection else 1e-7,
                  max_iter=1200 if selection else 2500)
    ridge.fit(z, target - y_mean)
    return mean, scale, keep, y_mean, ridge


def predict_readout(model, state: np.ndarray) -> np.ndarray:
    mean, scale, keep, y_mean, ridge = model
    z = (state[:, keep] - mean[keep]) / scale[keep]
    return y_mean + ridge.predict(z)


def cumulative_nrmse(target: np.ndarray, prediction: np.ndarray, full_reference: np.ndarray) -> np.ndarray:
    variance = float(np.var(full_reference))
    return np.sqrt(np.cumsum((prediction - target) ** 2) / np.arange(1, len(target) + 1) / variance)


def evaluate(bank_factory: Callable[[], ReservoirBank], seed: int = SEED) -> dict:
    """Train on observed x, then recursively feed back predicted x values."""
    _, reference = lorenz_reference(seed)

    def encode(value: float) -> float:
        return 10.0 + 90.0 * float(np.clip(value, 0.0, 1.0))

    bank = bank_factory()
    base = np.array([bank.step(encode(value)) for value in reference[: CLOSURE - 1]], dtype=float)
    if base.shape != (CLOSURE - 1, 64):
        raise ValueError("The bank must return a 64-element M=8,N=8 state per input")
    features = base
    best = None
    for alpha in ALPHAS:
        readout = fit_readout(features[WASHOUT:TRAIN_END], reference[WASHOUT + 1 : TRAIN_END + 1], alpha, selection=True)
        predicted = predict_readout(readout, features[TRAIN_END:CLOSURE - 1])
        actual = reference[TRAIN_END + 1 : CLOSURE]
        candidate = (nmse(actual, predicted), -float(alpha))
        if best is None or candidate < best[0]:
            best = (candidate, float(alpha))
    alpha = best[1]
    readout = fit_readout(features[WASHOUT:CLOSURE - 1], reference[WASHOUT + 1 : CLOSURE], alpha)

    prediction = np.empty(TEST, dtype=float)
    input_value = float(reference[CLOSURE - 1])
    for step in range(TEST):
        feature = np.asarray(bank.step(encode(input_value)), dtype=float)[None, :]
        prediction[step] = float(np.clip(predict_readout(readout, feature)[0], 0.0, 1.0))
        input_value = prediction[step]

    target = reference[CLOSURE : CLOSURE + TEST]
    error = cumulative_nrmse(target, prediction, reference)
    crossed = np.flatnonzero(error > THRESHOLD)
    return {
        "target": target,
        "prediction": prediction,
        "cumulative_nrmse": error,
        "vpt_steps": int(crossed[0] + 1) if len(crossed) else None,
        "selected_ridge_alpha": alpha,
        "validation_nmse": best[0][0],
        "reference_variance": float(np.var(reference)),
    }


if __name__ == "__main__":
    raise SystemExit("Import this module and call evaluate(bank_factory) with the calibrated reservoir bank.")
