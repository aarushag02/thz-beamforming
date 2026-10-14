import numpy as np

from src.trajectories import (BOUNDS_MAX, BOUNDS_MIN, GRID_STEP, coordinated_turn,
                              snap_to_grid)


def test_zero_turn_rate_gives_a_straight_line_at_constant_speed():
    path = coordinated_turn(steps=20, speed=16.0, turn_rates=np.zeros(20),
                            dt=0.1, start=(260.0, 300.0), heading=np.pi / 2)

    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.allclose(steps, 1.6)
    assert np.allclose(path[:, 0], 260.0)


def test_constant_turn_rate_traces_a_circle_of_radius_speed_over_omega():
    speed, omega, heading = 16.0, 0.3, 0.0
    start = np.array([260.0, 400.0])

    path = coordinated_turn(steps=60, speed=speed, turn_rates=np.full(60, omega),
                            dt=0.1, start=start, heading=heading)

    radius = speed / omega
    centre = start + radius * np.array([-np.sin(heading), np.cos(heading)])
    assert np.allclose(np.linalg.norm(path - centre, axis=1), radius, atol=1e-6)


def test_snapping_places_every_point_on_the_deepmimo_grid():
    rng = np.random.default_rng(0)
    path = rng.uniform([243.0, 300.0], [277.0, 800.0], size=(50, 2))

    snapped = snap_to_grid(path)

    offsets = (snapped - snap_to_grid(snapped)) / GRID_STEP
    assert np.allclose(offsets, 0.0)
    assert np.all(np.abs(snapped - path) <= GRID_STEP / 2 + 1e-9)


def test_localisation_noise_has_the_requested_sigma_and_leaves_truth_intact():
    from src.trajectories import add_localisation_noise

    rng = np.random.default_rng(1)
    truth = np.tile([260.0, 400.0], (20000, 1))
    original = truth.copy()

    observed = add_localisation_noise(truth, sigma=0.5, rng=rng)

    assert np.allclose(truth, original), "input was mutated"
    assert abs(np.std(observed - truth) - 0.5) < 0.02


def test_maneuver_turn_rates_are_piecewise_constant_and_bounded():
    from src.trajectories import maneuver_turn_rates

    rng = np.random.default_rng(2)

    omega = maneuver_turn_rates(steps=200, rng=rng, switch_prob=0.02, omega_max=0.4)

    assert omega.shape == (200,)
    assert np.all(np.abs(omega) <= 0.4)
    switches = np.count_nonzero(np.diff(omega))
    assert 0 < switches < 30, f"expected a handful of maneuvers, got {switches}"


def test_generated_trajectories_stay_inside_the_scenario_grid():
    from src.trajectories import generate_corpus

    corpus = generate_corpus(n=25, steps=200, rng=np.random.default_rng(3), sigma=0.5)

    assert len(corpus) == 25
    for traj in corpus:
        assert traj["true"].shape == (200, 2)
        assert traj["observed"].shape == (200, 2)
        assert np.all(traj["true"] >= BOUNDS_MIN) and np.all(traj["true"] <= BOUNDS_MAX)
        assert np.allclose(traj["true"], snap_to_grid(traj["true"]))


def test_corpus_is_reproducible_from_its_seed():
    from src.trajectories import generate_corpus

    a = generate_corpus(n=5, steps=50, rng=np.random.default_rng(7), sigma=0.5)
    b = generate_corpus(n=5, steps=50, rng=np.random.default_rng(7), sigma=0.5)

    for ta, tb in zip(a, b):
        assert np.array_equal(ta["true"], tb["true"])
        assert np.array_equal(ta["observed"], tb["observed"])
