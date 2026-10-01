"""Run MG and Lorenz."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from sklearn.linear_model import Ridge

MODEL_TAU_ON_S = 0.6334197817312851
MODEL_TAU_S = np.array([0.7880779417522618, 9.027968384936257], dtype=float)
MODEL_POWERS = np.arange(10.0, 101.0, 10.0)
MODEL_SOURCE_NODES = 96
MODEL_UV_ON_S, MODEL_RECOVERY_S = 0.6, 0.9


class CNTReservoir:

    def __init__(self, biases=tuple(range(1, 11)), nodes=6):
        self.biases = tuple(biases)
        if not self.biases or any(b not in range(1, 11) for b in self.biases):
            raise ValueError("Biases must be integers from 1 to 10 mV")
        if not isinstance(nodes, int) or not 1 <= nodes <= 96:
            raise ValueError("nodes must be an integer from 1 to 96")
        self.nodes = nodes
        self.state = {b: np.zeros(2, dtype=float) for b in self.biases}
        self.indices = (np.floor((np.arange(nodes - 1) + 0.5) * 96 / (nodes - 1)).astype(int)
                        if nodes > 1 else np.empty(0, dtype=int))
        self.phases = np.arange(96, dtype=float) * 1.5 / 96
        self.on_mask = self.phases <= MODEL_UV_ON_S
        self.denominator = 1.0 - np.exp(-1.0 / MODEL_TAU_ON_S)
        self.on_decay = np.exp(-MODEL_UV_ON_S / MODEL_TAU_S)
        self.off_decay = np.exp(-MODEL_RECOVERY_S / MODEL_TAU_S)
        self.phase_decay = np.exp(-self.phases[self.on_mask, None] / MODEL_TAU_S[None, :])
        self.rise = (1.0 - np.exp(-self.phases[self.on_mask] / MODEL_TAU_ON_S)) / self.denominator
        self.tail_decay = np.exp(-(self.phases[~self.on_mask, None] - MODEL_UV_ON_S) / MODEL_TAU_S[None, :])

    def curves(self, uv_power_percent: float) -> dict[int, np.ndarray]:
        power = float(uv_power_percent)
        if not np.isfinite(power) or not 10.0 <= power <= 100.0:
            raise ValueError("UV power must be finite and in [10, 100] percent")
        result = {}
        for bias in self.biases:
            amplitude = float(np.interp(power, MODEL_POWERS, MODEL_AMPLITUDE_MA[bias - 1]))
            wf = float(np.clip(np.interp(power, MODEL_POWERS, MODEL_FAST_WEIGHT[bias - 1]), 0.0, 1.0))
            weight = np.array([wf, 1.0 - wf])
            terminal = amplitude * (1.0 - np.exp(-MODEL_UV_ON_S / MODEL_TAU_ON_S)) / self.denominator
            state_at_off = self.state[bias] * self.on_decay + weight * terminal
            current = np.empty(96, dtype=float)
            current[self.on_mask] = np.sum(self.state[bias][None, :] * self.phase_decay
                + (amplitude * self.rise)[:, None] * weight[None, :], axis=1)
            current[~self.on_mask] = np.sum(state_at_off[None, :] * self.tail_decay, axis=1)
            self.state[bias] = state_at_off * self.off_decay
            result[bias] = current
        return result

    def step(self, uv_power_percent: float) -> np.ndarray:
        curves = self.curves(uv_power_percent)
        return np.concatenate([np.r_[np.max(curves[b]), curves[b][self.indices]] for b in self.biases])


def runtime_versions() -> dict:
    import platform
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__}


def print_summary(summary: dict) -> None:
    import json
    print(json.dumps(summary, indent=2, allow_nan=False))



class ReservoirBank(Protocol):
    def step(self, uv_power_percent: float) -> np.ndarray: ...


WASHOUT, LEARNING, VALIDATION = 500, 5000, 1000
TRAIN_END = WASHOUT + LEARNING
CLOSURE = TRAIN_END + VALIDATION
SEED = 20260810
ALPHAS = np.logspace(-9, 4, 14)
THRESHOLD = 0.4


@dataclass(frozen=True)
class TaskConfig:
    task: str
    label: str
    biases: tuple[int, ...]
    nodes: int
    test_steps: int
    integration_dt: float
    sample_interval: float
    integrator: str
    state_spacing: int = 1
    burn_time: float = 2000.0
    discard_steps: int = 2000
    seed: int | None = None


def task_config(task: str, *, mg_dt: float = 0.1,
                mg_sample_interval: float = 0.1, lorenz_seed: int = SEED,
                mg_burn_time: float = 2000.0, lorenz_discard_steps: int = 2000) -> TaskConfig:
    if task == "mg":
        return TaskConfig("mg", "Mackey-Glass", tuple(range(1, 11)), 6, 1000,
                          mg_dt, mg_sample_interval, "RK4 with linear delayed-state interpolation",
                          state_spacing=10, burn_time=mg_burn_time)
    if task == "lorenz":
        return TaskConfig("lorenz", "Lorenz-x", tuple(range(3, 11)), 8, 2000,
                          0.01, 0.02, "RK4", discard_steps=lorenz_discard_steps,
                          seed=lorenz_seed)
    raise ValueError("task must be 'mg' or 'lorenz'")


def mackey_glass_reference(n_samples: int = 7501, *, dt: float = 0.1,
                          sample_interval: float = 0.1, burn_time: float = 2000.0) -> np.ndarray:
    beta, gamma, delay, exponent = 0.2, 0.1, 17.0, 10
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not np.isfinite([dt, sample_interval, burn_time]).all() or dt <= 0 or sample_interval <= 0 or burn_time < 0:
        raise ValueError("Positive finite dt/sample_interval and nonnegative burn_time required")
    def grid_steps(duration):
        steps = int(round(duration / dt))
        if not np.isclose(steps * dt, duration, rtol=1e-12, atol=1e-12):
            raise ValueError("dt must divide delay, sample_interval and burn_time")
        return steps
    lag, stride, burn = grid_steps(delay), grid_steps(sample_interval), grid_steps(burn_time)
    if lag < 2 or stride < 1:
        raise ValueError("dt must be <= sample_interval and <= delay/2")
    steps = burn + (n_samples - 1) * stride
    x = np.full(lag + steps + 1, 1.2, dtype=float)
    def rhs(value, delayed):
        return beta * delayed / (1.0 + delayed ** exponent) - gamma * value
    for k in range(lag, lag + steps):
        j = k - lag
        delayed_half = 0.5 * (x[j] + x[j + 1])
        a = rhs(x[k], x[j])
        b = rhs(x[k] + dt * a / 2, delayed_half)
        c = rhs(x[k] + dt * b / 2, delayed_half)
        d = rhs(x[k] + dt * c, x[j + 1])
        x[k + 1] = x[k] + dt * (a + 2 * b + 2 * c + d) / 6
    result = x[lag + burn : lag + burn + n_samples * stride : stride]
    if len(result) != n_samples or not np.isfinite(result).all():
        raise ValueError("Invalid Mackey-Glass reference trajectory")
    return result


def lorenz_rhs(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    return np.array((10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z))


def lorenz_reference(seed: int = SEED, n_samples: int = 8501, *,
                     discard_steps: int = 2000) -> np.ndarray:
    if not isinstance(n_samples, (int, np.integer)) or n_samples < CLOSURE:
        raise ValueError("Lorenz reference is shorter than the observed interval")
    rng = np.random.default_rng(seed)
    initial = np.ones(3, dtype=float) + rng.normal(0.0, 0.015, 3)
    if not isinstance(discard_steps, int) or discard_steps < 0:
        raise ValueError("discard_steps must be a nonnegative integer")
    discard, stride = discard_steps, 2
    steps = discard + (n_samples + 1) * stride
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
    raw_x = trajectory[discard::stride, 0][:n_samples]
    return raw_x


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if target.shape != prediction.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("NMSE requires matching finite target and prediction arrays")
    variance = float(np.var(target))
    if variance <= 0:
        raise ValueError("NMSE requires positive target variance")
    return float(np.mean((target - prediction) ** 2) / variance)


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


def cumulative_nrmse(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    # VPT uses cumulative prediction error and test-target variance.
    variance = float(np.var(target))
    return np.sqrt(np.cumsum((prediction - target) ** 2) / np.arange(1, len(target) + 1) / variance)


def readout_offsets(config: TaskConfig) -> tuple[int, ...]:
    return tuple(range(0, 6 * config.state_spacing, config.state_spacing))


def causal_state_embedding(base: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    if base.ndim != 2 or len(base) == 0 or not np.isfinite(base).all():
        raise ValueError("A nonempty finite state matrix is required")
    indices = np.arange(len(base))
    return np.hstack([base[np.maximum(indices - offset, 0)] for offset in offsets])


def generate_reference(config: TaskConfig) -> np.ndarray:
    n_samples = CLOSURE + config.test_steps + 1
    if config.task == "mg":
        return mackey_glass_reference(n_samples, dt=config.integration_dt,
                                     sample_interval=config.sample_interval, burn_time=config.burn_time)
    if config.task == "lorenz":
        return lorenz_reference(config.seed, n_samples, discard_steps=config.discard_steps)
    raise ValueError("Unknown reference task")


def make_encoder(reference: np.ndarray, config: TaskConfig) -> Callable[[float], float]:
    # Convert raw values directly to UV power using the learning interval.
    lower = float(np.min(reference[WASHOUT:TRAIN_END]))
    upper = float(np.max(reference[WASHOUT:TRAIN_END]))
    if upper <= lower:
        raise ValueError("Learning inputs must have a positive encoding range")
    return lambda value: float(np.clip(10.0 + 90.0 * (value - lower) / (upper - lower), 10.0, 100.0))


def run_closed_loop(reference: np.ndarray, config: TaskConfig,
                    bank_factory: Callable[[], ReservoirBank] | None = None) -> dict:
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (CLOSURE + config.test_steps + 1,) or not np.isfinite(reference).all():
        raise ValueError("Reference must have the configured length and finite values")
    encode = make_encoder(reference, config)
    bank = (CNTReservoir(biases=config.biases, nodes=config.nodes)
            if bank_factory is None else bank_factory())
    base = np.array([bank.step(encode(value)) for value in reference[:CLOSURE - 1]], dtype=float)
    dimensions = len(config.biases) * config.nodes
    if base.shape != (CLOSURE - 1, dimensions) or not np.isfinite(base).all():
        raise ValueError(f"The bank must return {dimensions} finite state values per input")
    offsets = readout_offsets(config)
    features = causal_state_embedding(base, offsets)
    # Split by target index: state row k predicts target k + 1.
    train = slice(WASHOUT - 1, TRAIN_END - 1)
    validation = slice(TRAIN_END - 1, CLOSURE - 1)
    development = slice(WASHOUT - 1, CLOSURE - 1)
    best = None
    for alpha in ALPHAS:
        model = fit_readout(features[train], reference[WASHOUT:TRAIN_END], alpha, selection=True)
        predicted = predict_readout(model, features[validation])
        actual = reference[TRAIN_END:CLOSURE]
        candidate = (nmse(actual, predicted), -float(alpha))
        if best is None or candidate < best[0]:
            best = (candidate, float(alpha))
    alpha = best[1]
    readout = fit_readout(features[development], reference[WASHOUT:CLOSURE], alpha)

    # Continue the existing reservoir and state buffer from the last observation.
    history = deque((base[CLOSURE - 1 - delay].copy() for delay in range(1, offsets[-1] + 1)),
                    maxlen=offsets[-1] + 1)
    prediction = np.empty(config.test_steps, dtype=float)
    input_value = float(reference[CLOSURE - 1])
    for step in range(config.test_steps):
        state = np.asarray(bank.step(encode(input_value)), dtype=float)
        if state.shape != (dimensions,) or not np.isfinite(state).all():
            raise ValueError("Invalid reservoir state during autonomous prediction")
        history.appendleft(state.copy())
        feature = np.concatenate([history[offset] for offset in offsets])[None, :]
        value = float(predict_readout(readout, feature)[0])
        if not np.isfinite(value):
            raise ValueError("Non-finite readout output during autonomous prediction")
        prediction[step] = value
        input_value = value

    target = reference[CLOSURE:CLOSURE + config.test_steps]
    error = cumulative_nrmse(target, prediction)
    crossed = np.flatnonzero(error > THRESHOLD)
    return {
        "test_nmse": nmse(target, prediction),
        "vpt_steps": int(crossed[0] + 1) if len(crossed) else None,
    }


def evaluate(task: str = "mg", *, mg_dt: float = 0.1,
             mg_sample_interval: float = 0.1, lorenz_seed: int = SEED,
             mg_burn_time: float = 2000.0, lorenz_discard_steps: int = 2000,
             bank_factory: Callable[[], ReservoirBank] | None = None) -> dict:
    config = task_config(task, mg_dt=mg_dt, mg_sample_interval=mg_sample_interval,
                         lorenz_seed=lorenz_seed, mg_burn_time=mg_burn_time,
                         lorenz_discard_steps=lorenz_discard_steps)
    return run_closed_loop(generate_reference(config), config, bank_factory)





def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("mg", "lorenz", "both"), default="both",
                        help="Task to run (default: both)")
    parser.add_argument("--mg-dt", type=float, default=0.1,
                        help="MG integration step in system units (default: 0.1)")
    parser.add_argument("--mg-sample-interval", type=float, default=0.1,
                        help="MG reference sampling interval (default: 0.1)")
    parser.add_argument("--lorenz-seed", type=int, default=SEED)
    parser.add_argument("--mg-burn-time", type=float, default=2000.0,
                        help="MG transient duration in system units; 0 disables discarding")
    parser.add_argument("--lorenz-discard-steps", type=int, default=2000,
                        help="Lorenz transient integration steps; 0 disables discarding")
    args = parser.parse_args()
    tasks = ("mg", "lorenz") if args.task == "both" else (args.task,)
    summaries = {}
    for task in tasks:
        config = task_config(task, mg_dt=args.mg_dt, mg_sample_interval=args.mg_sample_interval,
                             lorenz_seed=args.lorenz_seed, mg_burn_time=args.mg_burn_time,
                             lorenz_discard_steps=args.lorenz_discard_steps)
        result = run_closed_loop(generate_reference(config), config)
        summaries[task] = result
    print_summary(summaries)


if __name__ == "__main__":
    main()
