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


# --- Streaming trackers -----------------------------------------------------
#
# The windowed kalman_cv above restarts from zero velocity in every window,
# which is not how a base station would run a tracker and hands the learned
# models an easy win. These run once along the whole trajectory. Row t of every
# output is the prediction of position t+1 from obs[0..t].

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # state [x, vx, y, vy]


def _cv_transition(dt):
    F = np.eye(4)
    F[0, 1] = F[2, 3] = dt
    return F


def _ct_transition(dt, omega):
    """Coordinated turn at a known rate omega (rad/s), counter-clockwise for omega > 0."""
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    return np.array([
        [1.0, s / omega, 0.0, -(1.0 - c) / omega],
        [0.0, c, 0.0, -s],
        [0.0, (1.0 - c) / omega, 1.0, s / omega],
        [0.0, s, 0.0, c],
    ])


def _process_noise(dt, q):
    block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    zero = np.zeros((2, 2))
    return q * np.block([[block, zero], [zero, block]])


def _initial_state(first, sigma):
    return (np.array([first[0], 0.0, first[1], 0.0]),
            np.diag([sigma**2, 100.0, sigma**2, 100.0]))


def _kf_update(x, P, z, R):
    """Measurement update; also returns the innovation log-likelihood."""
    residual = z - _H @ x
    S = _H @ P @ _H.T + R
    S_inv = np.linalg.inv(S)
    K = P @ _H.T @ S_inv
    x = x + K @ residual
    P = (np.eye(4) - K @ _H) @ P
    _, logdet = np.linalg.slogdet(2.0 * np.pi * S)
    return x, P, -0.5 * (residual @ S_inv @ residual + logdet)


def kalman_cv_stream(obs, dt=0.1, sigma=0.5, q=1.0):
    """Constant-velocity Kalman filter run once along the trajectory."""
    obs = np.asarray(obs, dtype=float)[:, :2]
    F, Q, R = _cv_transition(dt), _process_noise(dt, q), sigma**2 * np.eye(2)
    x, P = _initial_state(obs[0], sigma)
    out = np.empty((len(obs), 2))
    for t, z in enumerate(obs):
        if t > 0:
            x, P = F @ x, F @ P @ F.T + Q
        x, P, _ = _kf_update(x, P, z, R)
        out[t] = (F @ x)[[0, 2]]
    return out


def _imm_mix(xs, Ps, mu, Pi):
    """IMM interaction step: predicted model probabilities and mixed priors."""
    c = Pi.T @ mu                                   # c[j] = P(model j next)
    weights = Pi * mu[:, None] / np.maximum(c[None, :], 1e-300)   # [i, j] = P(i | j)
    x_mix = [sum(weights[i, j] * xs[i] for i in range(len(xs))) for j in range(len(xs))]
    P_mix = []
    for j in range(len(xs)):
        P_j = np.zeros((4, 4))
        for i in range(len(xs)):
            d = xs[i] - x_mix[j]
            P_j += weights[i, j] * (Ps[i] + np.outer(d, d))
        P_mix.append(P_j)
    return c, x_mix, P_mix


def imm_stream(obs, dt=0.1, sigma=0.5, q=1.0, omega=0.25, p_stay=0.95):
    """Interacting-multiple-model tracker: CV, CT(+omega), CT(-omega).

    The standard maneuvering-target tracker and the comparator a tracking
    reviewer will ask for. All three models share the [x, vx, y, vy] state, so
    mixing is exact. Model likelihoods are combined in log space so a wild
    measurement cannot underflow every probability to zero.
    """
    obs = np.asarray(obs, dtype=float)[:, :2]
    Fs = [_cv_transition(dt), _ct_transition(dt, omega), _ct_transition(dt, -omega)]
    M = len(Fs)
    Q, R = _process_noise(dt, q), sigma**2 * np.eye(2)
    Pi = np.full((M, M), (1.0 - p_stay) / (M - 1))
    np.fill_diagonal(Pi, p_stay)

    x0, P0 = _initial_state(obs[0], sigma)
    xs, Ps = [x0.copy() for _ in Fs], [P0.copy() for _ in Fs]
    mu = np.full(M, 1.0 / M)
    out = np.empty((len(obs), 2))
    x_mix, P_mix = xs, Ps

    for t, z in enumerate(obs):
        if t > 0:
            xs = [F @ x for F, x in zip(Fs, x_mix)]
            Ps = [F @ P @ F.T + Q for F, P in zip(Fs, P_mix)]
            mu = c
        log_lik = np.empty(M)
        for j in range(M):
            xs[j], Ps[j], log_lik[j] = _kf_update(xs[j], Ps[j], z, R)
        log_post = np.log(np.maximum(mu, 1e-300)) + log_lik
        log_post -= log_post.max()
        mu = np.exp(log_post)
        mu /= mu.sum()

        c, x_mix, P_mix = _imm_mix(xs, Ps, mu, Pi)
        out[t] = sum(c[j] * (Fs[j] @ x_mix[j])[[0, 2]] for j in range(M))
    return out


def stream_window_predictor(obs, fn, window):
    """Run a (W, 2) -> (2,) window predictor causally along a trajectory.

    Early slots use whatever history exists; a single point falls back to hold.
    """
    obs = np.asarray(obs, dtype=float)[:, :2]
    out = np.empty((len(obs), 2))
    for t in range(len(obs)):
        recent = obs[max(0, t - window + 1):t + 1]
        out[t] = recent[-1] if len(recent) < 2 else fn(recent)
    return out
