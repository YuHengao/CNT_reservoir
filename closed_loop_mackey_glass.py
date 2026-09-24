"""Closed-loop Mackey–Glass.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from sklearn.linear_model import Ridge


class ReservoirBank(Protocol):
    def step(self, uv_power_percent: float) -> np.ndarray: ...


WASHOUT, LEARNING, VALIDATION, TEST = 1000, 3000, 1000, 1000
CLOSURE = WASHOUT + LEARNING + VALIDATION
TOTAL = CLOSURE + TEST
ALPHAS = np.logspace(-9, 4, 14)
THRESHOLD = 0.4


def mackey_glass_reference(n_samples: int = TOTAL + 1) -> np.ndarray:
    """Historical reference: forward Euler dt=0.1, sampled every 1.0 system unit."""
    beta, gamma, delay, exponent, dt, burn = 0.2, 0.1, 17.0, 10, 0.1, 2000
    stride, delay_steps = int(round(1.0 / dt)), int(round(delay / dt))
    total_steps = (n_samples + burn) * stride + delay_steps + 1
    x = np.full(total_steps, 1.2, dtype=float)
    for k in range(delay_steps, total_steps - 1):
        delayed = x[k - delay_steps]
        x[k + 1] = x[k] + dt * (beta * delayed / (1.0 + delayed**exponent) - gamma * x[k])
    start = delay_steps + burn * stride
    return x[start : start + n_samples * stride : stride]


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
    """Numerator: first n closed-loop errors; denominator: full-reference variance."""
    variance = float(np.var(full_reference))
    return np.sqrt(np.cumsum((prediction - target) ** 2) / np.arange(1, len(target) + 1) / variance)


def evaluate(bank_factory: Callable[[], ReservoirBank]) -> dict:
    """Train a Ridge readout, roll out autonomous predictions and evaluate VPT.

    Selection and refit targets end at reference[CLOSURE - 1]; the first
    closed-loop target is reference[CLOSURE]. Keeping the test interval fixed
    leaves VALIDATION - 1 supervised validation pairs. The final observed state
    seeds the rollout but is excluded from selection and refitting.
    """
    reference = mackey_glass_reference()
    lower, upper = float(np.min(reference[WASHOUT:CLOSURE])), float(np.max(reference[WASHOUT:CLOSURE]))

    def encode(value: float) -> float:
        return float(np.clip(10.0 + 90.0 * (value - lower) / (upper - lower), 10.0, 100.0))

    bank = bank_factory()
    base = np.array([bank.step(encode(value)) for value in reference[:CLOSURE]], dtype=float)
    if base.shape != (CLOSURE, 60):
        raise ValueError("The bank must return a 60-element M=10,N=6 state per input")
    features = base
    train = slice(WASHOUT, WASHOUT + LEARNING)
    validation = slice(WASHOUT + LEARNING, CLOSURE - 1)
    development = slice(WASHOUT, CLOSURE - 1)
    best = None
    for alpha in ALPHAS:
        model = fit_readout(features[train], reference[WASHOUT + 1 : WASHOUT + LEARNING + 1], alpha, selection=True)
        predicted = predict_readout(model, features[validation])
        actual = reference[WASHOUT + LEARNING + 1 : CLOSURE]
        candidate = (nmse(actual, predicted), -float(alpha))
        if best is None or candidate < best[0]:
            best = (candidate, float(alpha))
    alpha = best[1]
    readout = fit_readout(features[development], reference[WASHOUT + 1 : CLOSURE], alpha)

    prediction = np.empty(TEST, dtype=float)
    prediction[0] = float(predict_readout(readout, features[CLOSURE - 1 : CLOSURE])[0])
    for step in range(1, TEST):
        state = np.asarray(bank.step(encode(prediction[step - 1])), dtype=float)
        prediction[step] = float(predict_readout(readout, state[None, :])[0])

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
