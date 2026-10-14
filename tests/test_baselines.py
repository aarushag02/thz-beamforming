import numpy as np

from src.baselines import constant_velocity, hold, ls_linear


def _straight(steps=12, step=1.6):
    """Noiseless constant-velocity path, the easiest possible case."""
    t = np.arange(steps)[:, None]
    return np.array([260.0, 300.0]) + t * np.array([0.0, step])


def test_hold_predicts_the_last_observed_position():
    window = _straight()

    assert np.allclose(hold(window), window[-1])


def test_constant_velocity_is_exact_on_a_noiseless_straight_line():
    path = _straight(steps=12)
    window, truth = path[:10], path[10]

    assert np.allclose(constant_velocity(window), truth)


def test_ls_linear_is_exact_on_a_noiseless_straight_line():
    path = _straight(steps=12)
    window, truth = path[:10], path[10]

    assert np.allclose(ls_linear(window), truth)


def _snap(path, grid=0.2):
    """Quantise to the DeepMIMO grid, exactly as find_nearest_user does."""
    return np.round(path / grid) * grid


def _realistic(steps=200):
    """Mirrors 1_data_generation.ipynb: 200 waypoints spanning y = 300..600,
    i.e. a 1.5075 m step that is NOT a multiple of the 0.2 m grid, plus a grid
    phase offset so snapping actually perturbs the path.
    """
    t = np.arange(steps)[:, None]
    return np.array([260.0, 300.0]) + t * np.array([0.09, 300.0 / (steps - 1)])


def test_ls_linear_beats_constant_velocity_under_grid_snapping():
    """Averaging over the window suppresses quantisation noise; two-point
    extrapolation amplifies it. This is the effect the LSTM is mostly exploiting.

    Theory: per-axis snapping error has sigma = 0.2/sqrt(12) = 0.058 m.
    Two-point extrapolation gives sqrt(6)*sigma; a 10-point LS fit extrapolated
    one step gives sqrt(1.47)*sigma. The expected ratio is ~0.50.
    """
    from src.baselines import rollout

    observed = _snap(_realistic(steps=200) + 0.037)

    cv_pred, truth = rollout(observed, window=10, predictor=constant_velocity)
    ls_pred, _ = rollout(observed, window=10, predictor=ls_linear)

    cv_mae = np.mean(np.linalg.norm(cv_pred - truth, axis=1))
    ls_mae = np.mean(np.linalg.norm(ls_pred - truth, axis=1))

    assert ls_mae < 0.7 * cv_mae, f"ls={ls_mae:.4f} cv={cv_mae:.4f}"


def test_quadratic_is_exact_on_a_noiseless_curved_path():
    from src.baselines import quadratic

    t = np.arange(12)[:, None]
    path = np.array([260.0, 300.0]) + t * np.array([0.3, 1.6]) + t**2 * np.array([0.02, 0.0])

    assert np.allclose(quadratic(path[:10]), path[10])


def test_rollout_aligns_with_the_projects_sliding_window_convention():
    """make_local_windows predicts pos[i+W] from pos[i:i+W]; rollout must match
    so MAE numbers stay comparable with the published LSTM results."""
    from src.baselines import rollout

    path = _straight(steps=30)

    preds, truth = rollout(path, window=10, predictor=hold)

    assert len(preds) == len(path) - 10
    assert np.allclose(truth, path[10:])
    assert np.allclose(preds[0], path[9])


def test_kalman_cv_is_near_exact_on_noiseless_constant_velocity_motion():
    """With no measurement noise and no maneuver, the CV filter should nail it."""
    from src.baselines import kalman_cv

    path = _straight(steps=40, step=1.6)
    predictor = kalman_cv(dt=1.0, sigma=0.5, q=1e-6)

    assert np.allclose(predictor(path[:30]), path[30], atol=1e-3)


def test_kalman_cv_beats_two_point_extrapolation_under_measurement_noise():
    """Recursive filtering should dominate differencing the last two samples."""
    from src.baselines import kalman_cv, rollout

    rng = np.random.default_rng(42)
    truth = _straight(steps=400, step=1.6)
    observed = truth + rng.normal(0.0, 0.5, truth.shape)
    predictor = kalman_cv(dt=1.0, sigma=0.5, q=1e-4)

    kf_pred, _ = rollout(observed, window=20, predictor=predictor)
    cv_pred, _ = rollout(observed, window=20, predictor=constant_velocity)

    kf_mae = np.mean(np.linalg.norm(kf_pred - truth[20:], axis=1))
    cv_mae = np.mean(np.linalg.norm(cv_pred - truth[20:], axis=1))

    assert kf_mae < cv_mae / 2, f"kf={kf_mae:.4f} cv={cv_mae:.4f}"
