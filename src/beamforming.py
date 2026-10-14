"""MRT beamforming, SNR, and beamwidth-normalised misalignment.

Lifted from 3_beamforming.ipynb so the beamforming evaluation can be tested and
reused, with two corrections:

- Beam direction is measured from the actual base-station position. The
  notebook used arctan2(x, y), which assumes the BS sits at the origin, but
  users span x in [242, 278] and y in [297, 847].
- Misalignment is reported in half-power beamwidths as well as degrees.
  Degrees alone are not comparable across array sizes, and the whole point of
  sweeping Nt is that the same metres of error cost 16x more beamwidths at
  Nt=128 than at Nt=8.
"""

import numpy as np


def mrt_weights(H):
    """Maximum-ratio transmission weights for channel H of shape (n_rx, n_tx, n_sub).

    w = conj(h) / ||h||, the SNR-optimal single-user precoder for a known channel.
    """
    h = np.asarray(H)[:, :, 0]
    h_conj = np.conj(h)
    norm = np.linalg.norm(h_conj)
    if norm < 1e-12:
        return np.zeros(h_conj.shape[1], dtype=complex)
    return h_conj[0] / norm


def snr(H, w, noise_power):
    """Received SNR when beamformer `w` meets channel `H`."""
    h = np.asarray(H)[0, :, 0]
    return float(np.abs(np.dot(h, np.asarray(w))) ** 2 / noise_power)


def snr_db(value):
    return float(10.0 * np.log10(value))


def spectral_efficiency(value):
    """Shannon capacity in bits/s/Hz."""
    return float(np.log2(1.0 + value))


def half_power_beamwidth_deg(n_tx):
    """~102/N degrees for a uniform linear array at half-wavelength spacing.

    12.75 deg at Nt=8 against 0.80 deg at Nt=128 -- the reason the original
    Nt=8 study could not resolve any difference between predictors. A 1.6 m
    step at 200 m range subtends 0.46 deg: 3.6% of a beamwidth at Nt=8, but
    57% of one at Nt=128.
    """
    return 102.0 / n_tx


def bearing_deg(position, bs_position):
    """Angle from the base station to a position, in degrees."""
    delta = np.asarray(position, dtype=float)[:2] - np.asarray(bs_position, dtype=float)[:2]
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def misalignment_deg(predicted, truth, bs_position):
    """Absolute angular error of a beam aimed at `predicted` when the user is
    at `truth`, as seen from the base station."""
    error = bearing_deg(predicted, bs_position) - bearing_deg(truth, bs_position)
    return float(abs((error + 180.0) % 360.0 - 180.0))


def misalignment_beamwidths(predicted, truth, bs_position, n_tx):
    """Misalignment as a fraction of a half-power beamwidth.

    This is the number that is comparable across array sizes; raw degrees are
    not. Above ~0.5 the beam has lost most of its gain on the user.
    """
    return misalignment_deg(predicted, truth, bs_position) / half_power_beamwidth_deg(n_tx)


def calibrate_noise(channels, target_snr_db=20.0):
    """Noise power giving `target_snr_db` for a well-aligned beam.

    Matches the notebook's calibration so spectral efficiencies stay in the
    physically sensible 8-12 bits/s/Hz range.
    """
    return float(np.mean(np.abs(np.asarray(channels)) ** 2) / 10 ** (target_snr_db / 10))
