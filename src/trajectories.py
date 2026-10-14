"""Maneuvering-target trajectory corpus for the DeepMIMO O1_60 scenario.

The project's three hand-drawn paths are tracked to within 0.03-0.11 m by
zero-parameter extrapolation, so they cannot separate one sequence model from
another. This module builds the harder benchmark: coordinated-turn motion with
a piecewise-constant turn rate that switches at random times -- the classic
case a constant-velocity filter cannot follow -- observed through realistic
localisation noise.
"""

import numpy as np

# O1_60 receiver grid, from 1_data_generation.ipynb: 497,931 points at 0.2 m
# spacing spanning roughly 36 m x 550 m.
GRID_STEP = 0.2
GRID_ORIGIN = np.array([242.423, 297.0])
BOUNDS_MIN = np.array([243.0, 299.0])
BOUNDS_MAX = np.array([277.0, 845.0])
# Start margins: the corridor is only ~34 m wide in x but ~546 m long in y.
START_MARGIN = np.array([3.0, 20.0])


def coordinated_turn(steps, speed, turn_rates, dt, start, heading):
    """Integrate constant-speed motion with a time-varying turn rate.

    Exact arc integration rather than Euler, so a constant turn rate traces a
    true circle of radius speed/omega and the test can assert that exactly:

        x += (v/w) * (sin(psi + w*dt) - sin(psi))
        y += (v/w) * (cos(psi) - cos(psi + w*dt))

    with the w -> 0 limit falling back to straight-line motion.

    Returns positions of shape (steps, 2); the first row is `start`.
    """
    turn_rates = np.asarray(turn_rates, dtype=float)
    position = np.array(start, dtype=float)
    psi = float(heading)

    out = np.empty((steps, 2))
    out[0] = position
    for t in range(1, steps):
        omega = turn_rates[t - 1]
        if abs(omega) < 1e-9:
            position = position + speed * dt * np.array([np.cos(psi), np.sin(psi)])
        else:
            radius = speed / omega
            psi_next = psi + omega * dt
            position = position + radius * np.array([
                np.sin(psi_next) - np.sin(psi),
                np.cos(psi) - np.cos(psi_next),
            ])
            psi = psi_next
        out[t] = position
    return out


def snap_to_grid(positions):
    """Quantise to the 0.2 m receiver grid, as find_nearest_user does."""
    positions = np.asarray(positions, dtype=float)
    return GRID_ORIGIN + np.round((positions - GRID_ORIGIN) / GRID_STEP) * GRID_STEP


# Mobility regimes, used for the held-out generalisation split: a model trained
# on one band is tested on another it never saw.
REGIMES = {
    "pedestrian": {"speed": (0.8, 2.0), "omega_max": 0.8},
    "cyclist": {"speed": (3.0, 8.0), "omega_max": 0.5},
    "vehicular": {"speed": (8.0, 20.0), "omega_max": 0.3},
}


def add_localisation_noise(positions, sigma, rng):
    """Model input = true position seen through a sensing-aided position estimate.

    A base station never knows exact UE coordinates. sigma ~ 0.5 m represents
    radar/lidar/camera-aided or 5G NR positioning -- roughly ten times the
    0.058 m grid quantisation, which is what makes filtering non-trivial.
    Returns a new array; the true positions are left untouched because they
    remain the regression target and the channel-lookup key.
    """
    positions = np.asarray(positions, dtype=float)
    return positions + rng.normal(0.0, sigma, size=positions.shape)


def maneuver_turn_rates(steps, rng, switch_prob=0.02, omega_max=0.4):
    """Piecewise-constant turn rate switching at random times.

    This is the standard maneuvering-target model from tracking literature, and
    it is deliberately the case a constant-velocity filter cannot follow: the
    filter is right between maneuvers and wrong at every switch. A sequence
    model can do better by learning how long turns last and how hard they are.
    """
    omega = np.empty(steps)
    current = rng.uniform(-omega_max, omega_max) if rng.random() < 0.5 else 0.0
    for t in range(steps):
        if rng.random() < switch_prob:
            # Half the maneuvers return to straight-line travel.
            current = 0.0 if rng.random() < 0.5 else rng.uniform(-omega_max, omega_max)
        omega[t] = current
    return omega


def generate_corpus(n, steps, rng, sigma=0.5, dt=0.1, regimes=None, max_attempts=40):
    """Build `n` trajectories that stay inside the O1_60 receiver grid.

    Each entry carries the grid-snapped `true` path (regression target and
    channel-lookup key) and the noisy `observed` path (model input), plus the
    regime label the generalisation split is built from.
    """
    regimes = regimes or list(REGIMES)
    corpus = []
    rejected = 0
    while len(corpus) < n:
        name = regimes[rng.integers(len(regimes))]
        spec = REGIMES[name]
        speed = rng.uniform(*spec["speed"])

        for _ in range(max_attempts):
            # Headings start along the street axis with some spread. A 400 m
            # vehicular path cannot wander freely across a 34 m-wide corridor,
            # so unbiased headings would be rejected almost every time.
            along_street = np.pi / 2 if rng.random() < 0.5 else -np.pi / 2
            path = coordinated_turn(
                steps=steps,
                speed=speed,
                turn_rates=maneuver_turn_rates(steps, rng, omega_max=spec["omega_max"]),
                dt=dt,
                start=rng.uniform(BOUNDS_MIN + START_MARGIN, BOUNDS_MAX - START_MARGIN),
                heading=along_street + rng.normal(0.0, 0.3),
            )
            if np.all(path >= BOUNDS_MIN) and np.all(path <= BOUNDS_MAX):
                break
        else:
            rejected += 1
            if rejected > 50 * n:
                raise RuntimeError(
                    f"could not fit {n} trajectories inside the grid; "
                    f"reduce steps ({steps}), dt ({dt}) or the speed bands"
                )
            continue  # this draw cannot be contained; resample the regime

        true = snap_to_grid(path)
        corpus.append({
            "true": true,
            "observed": add_localisation_noise(true, sigma, rng),
            "regime": name,
            "speed": speed,
            "dt": dt,
            "sigma": sigma,
        })
    return corpus
