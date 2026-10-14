"""Non-learned next-position predictors.

These exist to answer a question the project has never asked: does an RNN beat
arithmetic here at all? The trajectories are smooth paths snapped to a 0.2 m
grid, so a least-squares line through the window does the same denoising an
LSTM does, with no parameters. If the learned models cannot clear these, the
architecture comparison downstream is not measuring anything.

Every predictor takes a window of shape (W, 2) of absolute positions and
returns the predicted next position, shape (2,).
"""

import numpy as np


def hold(window):
    """Reactive baseline: assume the user does not move."""
    return window[-1].copy()


def constant_velocity(window):
    """Two-point extrapolation, p_hat = 2*p[-1] - p[-2].

    Uses only the final step, so it inherits the full grid-quantisation noise
    of both points -- roughly 2.4x the per-point error.
    """
    return 2.0 * window[-1] - window[-2]


def ls_linear(window):
    """Least-squares line through the whole window, extrapolated one step.

    This is the baseline that matters: averaging over W points suppresses the
    grid-snapping noise, which is most of what the LSTM is doing.
    """
    return _poly_extrapolate(window, degree=1)


def _poly_extrapolate(window, degree):
    steps = len(window)
    t = np.arange(steps, dtype=float)
    coeffs = np.polyfit(t, window, degree)
    return np.array([np.polyval(coeffs[:, axis], steps) for axis in range(window.shape[1])])


def quadratic(window):
    """Constant-acceleration fit over the window, extrapolated one step."""
    return _poly_extrapolate(window, degree=2)


def rollout(positions, window, predictor):
    """Apply a predictor across a whole trajectory.

    Matches make_local_windows in 2_lstm_training.ipynb: window pos[i:i+W]
    predicts pos[i+W], so predictions align to indices W..T-1 and MAE is
    directly comparable with the published LSTM numbers.

    Returns (predictions, truths), each of shape (T - W, 2).
    """
    positions = np.asarray(positions, dtype=float)[:, :2]
    preds = np.array([predictor(positions[i:i + window])
                      for i in range(len(positions) - window)])
    return preds, positions[window:]


BASELINES = {
    "hold": hold,
    "constant_velocity": constant_velocity,
    "ls_linear": ls_linear,
    "quadratic": quadratic,
}


def kalman_cv(dt=1.0, sigma=0.5, q=1e-3):
    """Constant-velocity Kalman filter with discrete white-noise acceleration.

    The strongest classical comparator and the first thing a tracking reviewer
    will ask for. It is optimal for genuinely constant-velocity motion observed
    in Gaussian noise, and degrades exactly where the turn rate switches -- so
    whatever a learned model wins over this filter is won by anticipating
    maneuvers, not by smoothing better.

    `q` (process-noise intensity) trades responsiveness against smoothing and
    must be tuned with the same budget as any model hyperparameter, or the
    comparison is a strawman.

    Returns a predictor with the usual (W, 2) -> (2,) signature.
    """
    state_dim = 4  # [x, vx, y, vy]
    F = np.eye(state_dim)
    F[0, 1] = F[2, 3] = dt
    H = np.zeros((2, state_dim))
    H[0, 0] = H[1, 2] = 1.0

    block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    Q = q * np.block([[block, np.zeros((2, 2))], [np.zeros((2, 2)), block]])
    R = sigma**2 * np.eye(2)

    def predict(window):
        x = np.array([window[0, 0], 0.0, window[0, 1], 0.0])
        P = np.diag([sigma**2, 10.0, sigma**2, 10.0])

        for measurement in window[1:]:
            x = F @ x
            P = F @ P @ F.T + Q

            residual = measurement - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ residual
            P = (np.eye(state_dim) - K @ H) @ P

        return (F @ x)[[0, 2]]

    return predict
