import numpy as np

from src.data import fit_scale, make_windows, split_by_regime, split_by_trajectory
from src.trajectories import generate_corpus


def _corpus(n=12, steps=60, seed=0, sigma=0.5):
    return generate_corpus(n=n, steps=steps, rng=np.random.default_rng(seed), sigma=sigma)


def test_windows_predict_true_position_from_observed_history():
    """Inputs are what the base station measures; the target is where the user
    actually is. The anchor is the last *observed* point, because that is all
    the base station has at inference time."""
    observed = np.arange(24, dtype=float).reshape(12, 2)
    true = observed + 100.0

    X, y, anchors = make_windows(observed, true, window=4)

    assert X.shape == (8, 4, 2) and y.shape == (8, 2) and anchors.shape == (8, 2)
    assert np.allclose(anchors[0], observed[3])
    assert np.allclose(X[0], observed[0:4] - observed[3])
    assert np.allclose(y[0], true[4] - observed[3])


def test_scale_is_fitted_on_training_data_not_hardcoded():
    """The published [2.0, 18.0] scale came from three noiseless hand-drawn
    paths and does not transfer to a noisy corpus."""
    X = np.random.default_rng(0).normal(0, 3.0, size=(500, 10, 2))

    scale = fit_scale(X)

    assert scale.shape == (2,)
    assert np.all(scale > 0)
    assert np.all(np.abs(X / scale) < 2.0)


def test_trajectory_split_never_puts_one_trajectory_in_two_sets():
    corpus = _corpus(n=20)

    train, val, test = split_by_trajectory(corpus, (0.6, 0.2, 0.2),
                                           rng=np.random.default_rng(1))

    ids = [{id(t) for t in part} for part in (train, val, test)]
    assert len(train) + len(val) + len(test) == 20
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_regime_split_holds_out_an_entire_mobility_band():
    """The generalisation axis: train on one kind of motion, test on another
    the model has never seen."""
    corpus = _corpus(n=40)

    seen, unseen = split_by_regime(corpus, held_out="pedestrian")

    assert len(unseen) > 0
    assert all(t["regime"] != "pedestrian" for t in seen)
    assert all(t["regime"] == "pedestrian" for t in unseen)


def test_inputs_and_targets_get_separate_scales():
    """A W-step window spans tens of metres; the target is a single ~1.6 m step.
    Sharing one scale shrinks the target to a few hundredths and recreates the
    vanishing-signal problem the project already solved once for W=10.
    """
    from src.data import build_dataset

    corpus = _corpus(n=10, steps=120, sigma=0.5)

    _, y, _, scales = build_dataset(corpus, window=20)

    assert scales["x"].shape == (2,) and scales["y"].shape == (2,)
    assert np.all(scales["y"] < scales["x"]), (
        f"target scale {scales['y']} should be far smaller than input scale {scales['x']}"
    )
    # Normalised targets must carry real signal, not be squashed toward zero.
    assert np.percentile(np.abs(y), 99) > 0.5
