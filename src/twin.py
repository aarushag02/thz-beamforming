"""Digital-twin channel lookup over the full receiver grid.

Fixes the leak in 3_beamforming.ipynb. That notebook resolved a predicted
position against `positions_2d[:current_t+2]` -- the trajectory so far, plus
the true next point. Since prediction error (~0.1 m) is far below the ~1.5 m
spacing between trajectory points, argmin returned index t+1 essentially every
time, so the "predicted" channel was the true future channel. SNR then became a
step function of prediction error with a threshold at half the step size, which
is why MAE 0.104 and 0.154 produced identical SNR gains.

Resolving against the real 0.2 m grid instead makes the channel vary with the
prediction, which is the only way an SNR metric can discriminate between
predictors.
"""

import numpy as np
from scipy.spatial import cKDTree


class DigitalTwin:
    """Nearest-grid-point channel lookup with memoisation.

    `channel_fn(indices) -> channels` is called lazily for grid points that are
    actually queried. Across a whole study only a few thousand of the 497,931
    O1_60 points are ever touched, so precomputing the grid is unnecessary.

    With DeepMIMO:

        twin = DigitalTwin(
            bs0['rx_pos'],
            lambda idx: np.array(bs0.trim(idxs=list(idx)).compute_channels()),
        )
    """

    def __init__(self, grid_positions, channel_fn):
        self.grid = np.asarray(grid_positions, dtype=float)[:, :2]
        self.channel_fn = channel_fn
        self._tree = cKDTree(self.grid)
        self._cache = {}

    def nearest_index(self, position):
        """Index of the closest grid point, searched over the entire grid."""
        return int(self._tree.query(np.asarray(position, dtype=float)[:2])[1])

    def nearest_indices(self, positions):
        return self._tree.query(np.asarray(positions, dtype=float)[:, :2])[1].astype(int)

    def channel_at(self, position):
        return self.channel_for_index(self.nearest_index(position))

    def channel_for_index(self, index):
        index = int(index)
        if index not in self._cache:
            self._cache[index] = np.asarray(self.channel_fn([index]))[0]
        return self._cache[index]

    def prefetch(self, indices):
        """Compute a batch of channels in one call -- far cheaper than one
        DeepMIMO trim per point."""
        missing = sorted({int(i) for i in indices} - self._cache.keys())
        if not missing:
            return
        channels = np.asarray(self.channel_fn(missing))
        for index, channel in zip(missing, channels):
            self._cache[index] = channel

    @property
    def n_cached(self):
        return len(self._cache)
