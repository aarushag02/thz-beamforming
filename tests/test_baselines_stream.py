import numpy as np

from src.baselines import (constant_velocity, imm_stream, kalman_cv_stream, ls_linear,
                           stream_window_predictor)
from src.trajectories import coordinated_turn


def _noisy(path, sigma=0.5, seed=0):
    return path + np.random.default_rng(seed).normal(0.0, sigma, size=path.shape)


def _one_step_mae(pred, truth, skip=20):
    # pred[t] predicts truth[t+1]
    return float(np.mean(np.linalg.norm(pred[skip:-1] - truth[skip + 1:], axis=1)))


def test_stream_outputs_have_trajectory_shape():
    obs = _noisy(coordinated_turn(50, 10.0, np.zeros(50), 0.1, (0.0, 0.0), 0.0))
    for fn in (lambda o: kalman_cv_stream(o), lambda o: imm_stream(o)):
        out = fn(obs)
        assert out.shape == (50, 2)
        assert np.all(np.isfinite(out))


def test_stream_trackers_are_causal():
    """Row t may depend on obs[0..t] only."""
    obs = _noisy(coordinated_turn(60, 10.0, np.full(60, 0.2), 0.1, (0.0, 0.0), 0.0))
    perturbed = obs.copy()
    perturbed[30:] += 25.0
    for fn in (kalman_cv_stream, imm_stream):
        a, b = fn(obs), fn(perturbed)
        assert np.allclose(a[:30], b[:30])
        assert not np.allclose(a[30:], b[30:])


def test_kalman_stream_tracks_a_straight_line():
    truth = coordinated_turn(200, 10.0, np.zeros(200), 0.1, (0.0, 0.0), 0.3)
    pred = kalman_cv_stream(_noisy(truth), dt=0.1, sigma=0.5, q=1.0)
    # Far better than the 0.5 m raw observation noise once converged.
    assert _one_step_mae(pred, truth) < 0.35


def test_imm_close_to_kalman_on_straight_motion():
    truth = coordinated_turn(200, 10.0, np.zeros(200), 0.1, (0.0, 0.0), 0.3)
    obs = _noisy(truth)
    kf = _one_step_mae(kalman_cv_stream(obs, q=1.0), truth)
    imm = _one_step_mae(imm_stream(obs, q=1.0, omega=0.3, p_stay=0.97), truth)
    # IMM pays a model-switching penalty on straight lines (measured 1.14-1.17x).
    assert imm < 1.3 * kf


def test_imm_beats_kalman_on_a_sustained_turn():
    truth = coordinated_turn(300, 10.0, np.full(300, 0.3), 0.1, (0.0, 0.0), 0.0)
    obs = _noisy(truth)
    kf = _one_step_mae(kalman_cv_stream(obs, q=1.0), truth)
    imm = _one_step_mae(imm_stream(obs, q=1.0, omega=0.3, p_stay=0.97), truth)
    assert imm < kf


def test_imm_survives_an_outlier_jump():
    """Review focus 1: all model likelihoods underflow on a 500 m jump."""
    obs = _noisy(coordinated_turn(80, 10.0, np.zeros(80), 0.1, (0.0, 0.0), 0.0))
    obs[40] += 500.0
    out = imm_stream(obs)
    assert np.all(np.isfinite(out))


def test_stream_window_predictor_matches_windowed_fn():
    obs = _noisy(coordinated_turn(40, 5.0, np.full(40, 0.1), 0.1, (0.0, 0.0), 0.0))
    W = 5
    for fn in (constant_velocity, ls_linear):
        out = stream_window_predictor(obs, fn, W)
        for t in range(W - 1, 40):
            assert np.allclose(out[t], fn(obs[t - W + 1:t + 1]))
        assert np.allclose(out[0], obs[0])   # one point: falls back to hold
