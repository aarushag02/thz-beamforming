# 5G and Programmable Networks Project
# Predictive THz Beamforming Using a Digital Twin and LSTM Trajectory Prediction

A complete end-to-end predictive beamforming system for 6G THz wireless communications, built entirely with open-source tools and runnable on consumer hardware (Apple Silicon MacBook Air M4).

The core idea: instead of pointing the antenna beam at where the user **is**, predict where they will **be next** and point there already. An LSTM neural network handles the prediction, and a DeepMIMO digital twin provides physically accurate channel data for any position in the scenario — including positions the user hasn't visited yet.

---

## The Problem

THz and millimetre-wave systems deliver high data rates by forming highly directional, pencil-thin antenna beams. This directionality creates a fundamental vulnerability: any mismatch between the beam direction and the user's actual position causes rapid signal degradation. Conventional **reactive** beamforming always steers toward the user's current position — meaning it is one timestep behind during turns and direction changes.

## The Solution

A three-component predictive framework:

1. **Digital twin (DeepMIMO v4)** — a ray-tracing-based channel database at 60 GHz. Given any position in the scenario grid, it returns a physically accurate channel matrix without real antenna measurements.
2. **LSTM trajectory predictor** — takes the last 10 known user positions and predicts the next position with sub-0.2 m accuracy on training trajectories.
3. **Predictive MRT beamforming** — at each timestep, the LSTM prediction is used to retrieve the future channel from the digital twin. Maximum Ratio Transmission weights are computed and the beam is pre-steered *now* to where the user will be *next*.

## Digital Twin: DeepMIMO

This project utilizes the DeepMIMO framework to generate high-fidelity 60GHz channel data. DeepMIMO is a widely recognized industry-standard dataset generator based on 3D ray-tracing, bridging the gap between theoretical modeling and real-world 5G/6G deployments.

**Digital Twin:** The simulation environment serves as a digital twin of a [Insert Scenario, e.g., "Dense Urban" or "Indoor Office"] environment, providing site-specific channel parameters that account for reflections, diffractions, and blockages.

**Industry Relevance:** DeepMIMO is extensively used in wireless R&D in companies like NVIDIA, Qualcomm and Nokia, to benchmark beamforming algorithms, channel estimation techniques, and Machine Learning models for mmWave and Sub-THz communications.

**Source:** The dataset was generated using the [DeepMIMO Framework](https://www.deepmimo.net/).

---

## Results Summary

### Training Trajectories

| Trajectory | LSTM MAE | Max Error | Reactive SNR | Predictive SNR | SNR Gain |
|---|---|---|---|---|---|
| Straight line | 0.107 m | 0.140 m | 25.59 dB | 25.66 dB | **+0.07 dB** |
| Gentle curve | 0.154 m | 0.375 m | 25.12 dB | 25.17 dB | **+0.05 dB** |
| Sharp turn | 0.104 m | 1.948 m | 26.88 dB | 26.96 dB | **+0.08 dB** |

### Generalisation Study (trajectories never seen during training)

| Trajectory | LSTM MAE | Reactive SNR | Predictive SNR | SNR Gain |
|---|---|---|---|---|
| Double turn / Z-shape | 0.167 m | 22.50 dB | 22.61 dB | **+0.11 dB** |
| Zigzag (fast oscillation) | 0.703 m | 26.65 dB | 26.59 dB | −0.06 dB |

The predictive system consistently outperforms the reactive baseline on all training trajectories and the double-turn generalisation test. The zigzag result reveals an important boundary: the model degrades when oscillation frequency significantly exceeds the training distribution — a known LSTM limitation and a concrete direction for future work.

---

## Key Technical Contribution — Local Window Normalisation

The most important finding in this project is a normalisation failure mode that affects any LSTM trained on position sequences in large environments.

**The problem:** The Y axis spans 300 m but each step is only 1.6 m. Standard min-max normalisation makes each step equal to `1.6/300 ≈ 0.005` in normalised space. The LSTM cannot learn from a signal this small and converges to predicting the training mean (~420 m), giving errors of 50–100 m.

**The fix:** For each training window, subtract the last known position (the anchor) from every point in the window and from the target. This converts the problem from "predict absolute position" to "predict how far the user moves next" — always around 1.6 m, regardless of where in the trajectory the user is. Scale factors `[sx, sy] = [2.0, 18.0]` m (99th percentile of relative values) bring inputs to `[-1, 1]`.

At inference time: `predicted_absolute = anchor + model_output × scale`

---

## Project Structure

```
thz-beamforming/
├── notebooks/
│   ├── 1_data_generation.ipynb      # DeepMIMO setup, trajectory design, channel computation
│   ├── 2_lstm_training.ipynb        # Local normalisation, sliding windows, LSTM training
│   ├── 3_beamforming.ipynb          # MRT beamforming, reactive vs predictive comparison
│   └── 4_generalization_test.ipynb  # Zigzag and double-turn unseen trajectory evaluation
├── models/
│   └── norm_params.npy              # Saved scale factors [2.0, 18.0]
│   # lstm_best.pt excluded (too large — regenerate by running notebook 2)
├── results/
│   ├── trajectories_visualization.png
│   ├── training_curves.png
│   ├── lstm_predictions.png
│   ├── prediction_error_over_time.png
│   ├── turn_zoom.png
│   ├── snr_comparison.png
│   ├── snr_turn_zoom.png
│   ├── spectral_efficiency.png
│   ├── beam_direction.png
│   ├── error_vs_snr_gain.png
│   ├── test_trajectories.png
│   ├── unseen_predictions.png
│   ├── generalisation_error.png
│   └── unseen_beamforming.png
├── .gitignore
└── README.md
```

> **Note:** The `data/` folder (trajectory `.npy` files) and `models/lstm_best.pt` are excluded from the repository because of file size. Both are regenerated by running the notebooks in order. The DeepMIMO O1\_60 scenario (~2.3 GB) downloads automatically on first run.

---

## Setup and Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) **or** any machine with Python 3.11+
- ~3 GB free disk space for the DeepMIMO scenario

### Install

```bash
# Create and activate a conda environment
conda create -n thz python=3.11 -y
conda activate thz

# Install all dependencies
pip install --pre deepmimo torch numpy matplotlib scipy jupyter ipykernel

# Register the kernel with Jupyter
python -m ipykernel install --user --name thz --display-name "Python (thz)"
```

### Run

Open the notebooks in order in VS Code or JupyterLab:

```
1_data_generation.ipynb     ← downloads O1_60 scenario on first run (~2.3 GB)
2_lstm_training.ipynb       ← trains the LSTM, saves lstm_best.pt and norm_params.npy
3_beamforming.ipynb         ← runs reactive vs predictive comparison, saves all plots
4_generalization_test.ipynb ← tests on unseen trajectories
```

Each notebook is self-contained with markdown explanations throughout.

---

## Model Architecture

A small, deliberately lightweight LSTM — the point is to show that even a minimal predictor is sufficient to improve beamforming.

```
Input: sequence of 10 relative (x, y) positions
  ↓
LSTM Layer 1 — 64 hidden units, dropout 0.1
  ↓
LSTM Layer 2 — 64 hidden units, dropout 0.1
  ↓
Take hidden state at final timestep only
  ↓
Dropout 0.1
  ↓
Linear layer — 64 → 2
  ↓
Output: predicted relative (Δx, Δy)
```

| Parameter | Value |
|---|---|
| Total parameters | 50,818 |
| Loss | Mean Squared Error |
| Optimiser | Adam, lr = 0.001 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 20) |
| Gradient clipping | max norm = 1.0 |
| Batch size | 16 |
| Epochs | 500 |
| Best val loss | 0.000486 (~0.23 m) |

---

## Scenario and Data

| Parameter | Value |
|---|---|
| Scenario | DeepMIMO O1\_60 (60 GHz outdoor corridor) |
| Grid size | 497,931 user locations |
| Grid spacing | 0.2 m |
| Coverage area | 36 m × 550 m |
| Channel shape | (1 rx antenna, 8 tx antennas, 1 subcarrier) |
| Trajectories | 3 training + 2 unseen test |
| Timesteps per trajectory | 200 |
| Step size | ~1.6 m |
| Unique positions used | 575 (training) + 372 (test) |

### Training Trajectories

- **Straight line** — X = 260 m fixed, Y from 300 m to 600 m
- **Gentle curve** — S-curve with ±8 m lateral oscillation, Y from 300 m to 600 m  
- **Sharp turn** — 100 steps straight down Y, then 90° turn and 100 steps across X

### Unseen Test Trajectories

- **Zigzag** — X = 255 m (different corridor), Y from 650 m to 800 m (outside training range), oscillation 5× faster than training
- **Double turn (Z-shape)** — two sequential 90° turns, motion pattern never seen in training

---

## Beamforming

Maximum Ratio Transmission (MRT) is used throughout. For channel vector **h** from 8 transmit antennas:

```
w = h* / ||h*||
```

where `h*` is the element-wise complex conjugate. This is the SNR-optimal single-user precoder for a known channel.

**Fair evaluation:** The predictive system is always evaluated against the *true* future channel `H[t+1]`, not the predicted one. Both systems are judged on the actual channel the user experiences.

**Noise calibration:** Noise power is set to `avg_channel_power / 100`, giving a baseline SNR of ~20 dB for a well-aligned beam. This produces physically realistic spectral efficiency values of 8–12 bits/s/Hz.

---

## Hardware and Software

| Tool | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0 (MPS backend) |
| DeepMIMO | 4.0.0 |
| NumPy | 2.2.6 |
| Matplotlib | 3.10.0 |
| Hardware | Apple MacBook Air M4 |

Training time: approximately 4 minutes for 500 epochs on Apple M4 MPS.  
No GPU cluster or cloud compute required.

---

## Future Directions

- Scale to larger antenna arrays (Nt = 64, 128) where narrower beams make prediction accuracy matter more
- Extend to OFDM with multiple subcarriers for frequency-selective channel modelling
- Train on a wider variety of motion speeds to close the generalisation gap on fast oscillations
- Multi-step prediction (2–5 steps ahead) to give the base station more lead time
- Validate on real hardware using a software-defined radio testbed
- Extend to multi-user MIMO scenarios

---

## References
1. A. Alkhateeb, "DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications," *Proc. ITA*, Feb. 2019.
2. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
3. T. S. Rappaport et al., "Wireless communications and applications above 100 GHz: Opportunities and challenges for 6G and beyond," *IEEE Access*, vol. 7, pp. 78729–78757, 2019.
4. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," *NeurIPS*, 2019.
