# thz-beamforming
# Predictive THz Beamforming Using a Digital Twin and LSTM

A predictive beamforming system for 6G THz wireless communications. 
An LSTM neural network predicts user position one timestep ahead, 
enabling the base station to pre-steer its antenna beam before the 
user arrives — consistently outperforming reactive beamforming across 
all tested trajectories.

## Stack
- Python 3.11 · PyTorch 2.11 · DeepMIMO v4 · NumPy · Matplotlib
- Apple Silicon M4 (MPS backend)

## Structure
- `notebooks/` — four Jupyter notebooks covering data generation, 
  LSTM training, beamforming comparison, and generalisation testing
- `results/` — all output plots
- `models/` — saved norm parameters (weights excluded, see below)

## Setup
conda create -n thz python=3.11 -y
conda activate thz
pip install --pre deepmimo torch numpy matplotlib scipy jupyter

## Running
Run the notebooks in order: 1 → 2 → 3 → 4.
The DeepMIMO O1_60 scenario (~2.3 GB) downloads automatically 
on first run via `dm.download('O1_60')`.

## Key Results
| Trajectory | LSTM MAE | SNR Gain |
|------------|----------|----------|
|  Straight  |  0.107 m | +0.07 dB |
|   Curve    |  0.154 m | +0.05 dB |
| Sharp Turn |  0.104 m | +0.08 dB |
| Double Turn|  0.167 m | +0.11 dB |
|   Zigzag   |  0.703 m | −0.06 dB |
