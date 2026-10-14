import numpy as np
import pytest

from src.twin import DigitalTwin


def _grid(step=0.2, nx=40, ny=60):
    xs = 242.423 + step * np.arange(nx)
    ys = 297.0 + step * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _fake_channels(indices):
    """Stand-in for DeepMIMO: a channel that depends on the grid index, so a
    different index is a detectably different channel."""
    indices = np.atleast_1d(indices)
    return np.stack([np.full((1, 8, 1), float(i), dtype=complex) for i in indices])


def test_nearest_index_searches_the_whole_grid():
    grid = _grid()
    twin = DigitalTwin(grid, _fake_channels)

    target = grid[1234] + np.array([0.03, -0.04])

    assert twin.nearest_index(target) == 1234


def test_channels_are_computed_once_per_unique_grid_point():
    grid = _grid()
    calls = []

    def counting(indices):
        calls.append(tuple(np.atleast_1d(indices)))
        return _fake_channels(indices)

    twin = DigitalTwin(grid, counting)
    for _ in range(5):
        twin.channel_at(grid[77])

    assert len(calls) == 1, f"expected one computation, got {len(calls)}"


def test_lookup_does_not_return_the_true_future_channel_when_prediction_is_off():
    """The bug this class exists to fix.

    3_beamforming.ipynb searched positions_2d[:t+2], which contains the true
    next position. Prediction error (~0.1 m) is far below the 1.5 m step
    spacing, so argmin returned p[t+1] essentially always and the 'predicted'
    channel was the true future channel. Searching the real grid instead, a
    prediction that misses by more than half a grid cell must resolve to a
    different channel.
    """
    grid = _grid()
    twin = DigitalTwin(grid, _fake_channels)

    true_next = grid[2000]
    predicted = true_next + np.array([0.0, 0.35])   # ~1.75 grid cells away

    assert twin.nearest_index(predicted) != twin.nearest_index(true_next)
    assert not np.allclose(twin.channel_at(predicted), twin.channel_at(true_next))


def test_prediction_within_half_a_cell_resolves_to_the_same_channel():
    """Quantisation is real and should be reported, not hidden: the twin can
    only resolve to 0.2 m."""
    grid = _grid()
    twin = DigitalTwin(grid, _fake_channels)

    true_next = grid[2000]

    assert np.allclose(twin.channel_at(true_next + 0.05), twin.channel_at(true_next))
