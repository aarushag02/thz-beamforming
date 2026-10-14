# Streaming minLSTM for Predictive Beam Tracking — Design

Date: 2026-10-14 · Branch: `minlstm-streaming` · Target: conference paper (VTC/Globecom/ICC class)

## 1. Goal and success criteria

Turn the v2 windowed study into a **streaming** study, where minLSTM's two
structural advantages actually apply:

- **Parallel training over long sequences** (whole trajectories, dense per-step loss).
- **O(1) per-slot inference** (carry hidden state; one recurrent update per slot).

Today's deliverable: tested code on the branch, one full run on a GCP GPU,
`results/streaming_study.json` + figures, and a LaTeX paper draft whose tables
are generated from that JSON.

A result is publishable whichever way it lands, as long as it is honest:
learned models are only claimed better than classical tracking if they beat a
**tuned streaming IMM** on held-out trajectories over multiple seeds.

Out of scope today: DeepSense 6G real data, uncertainty-aware beamwidth,
multi-horizon heads, minGRU (stretch only).

## 2. Problem formulation (streaming)

One trajectory has observed positions `obs[0..T-1]` (noisy, σ=0.5 m) and true
grid positions `true[0..T-1]`. At slot `t` the predictor has seen `obs[0..t]`
and must predict `true[t+1]`.

Sequence index `k = 0..L-1` with `L = T-2` corresponds to slot `t = k+1`:

| quantity | direct mode | hybrid mode |
|---|---|---|
| input features `X[k]` | `obs[t] - obs[t-1]` (2) | `obs[t] - obs[t-1]`, `imm[t] - obs[t]` (4) |
| anchor `A[k]` | `obs[t]` | `imm[t]` |
| target `Y[k]` | `true[t+1] - A[k]` | `true[t+1] - A[k]` |
| truth `P[k]` | `true[t+1]` | `true[t+1]` |

`imm[t]` = streaming IMM prediction of position `t+1` from `obs[0..t]` (causal).
Prediction in metres: `anchor + model_out * scale_y`.

Scales: per-axis 99th-percentile of |X| and |Y| over the **training** split,
separately (reuses `fit_scale`). Val/test reuse training scales.

**Warm-up:** loss and every metric use only `k >= WARMUP = 10`. All methods,
classical and learned, are scored on exactly this index set.

## 3. Components

### 3.1 `src/baselines.py` (add; keep existing functions)

- `kalman_cv_stream(obs, dt, sigma, q) -> (T, 2)`: CV Kalman run once along
  the trajectory; row `t` is the predicted position for `t+1` given
  `obs[0..t]`. Init state from `obs[0]`, zero velocity, P = diag(σ², 100, σ², 100)
  (state order `[x, vx, y, vy]`).
- `imm_stream(obs, dt, sigma, q, omega, p_stay) -> (T, 2)`: IMM with three linear
  models on the same `[x, vx, y, vy]` state: CV, CT(+omega), CT(-omega)
  (known-turn-rate coordinated-turn transition matrices). Markov matrix
  `p_stay` on the diagonal, `(1-p_stay)/2` off-diagonal. Standard IMM cycle:
  mix → per-model KF predict/update → model-probability update (Gaussian
  likelihood) → combine. Output row `t` = mixture-weighted one-step prediction.
- `stream_window_predictor(obs, fn, window) -> (T, 2)`: wraps existing window
  predictors (`hold`, `constant_velocity`, `ls_linear`) causally using
  `obs[max(0, t-window+1) : t+1]` (`constant_velocity`/`ls_linear` need ≥ 2
  points; for `t = 0` fall back to `hold`).

### 3.2 `src/data.py` (add)

- `build_sequences(corpus, mode="direct", imm_params=None, scales=None, warmup=10)`
  → dict with `X (N,L,F)`, `Y (N,L,2)` scaled, `anchors (N,L,2)`,
  `truth (N,L,2)`, `scales {"x","y"}`, `warmup`. All trajectories in a corpus
  share `T`, so arrays stack. `mode="hybrid"` requires `imm_params`
  (kwargs for `imm_stream` minus `obs`).
- `masked_mae_metres(pred_scaled, data)`: converts to absolute metres via
  anchors and scales; mean Euclidean error over `k >= warmup`.

### 3.3 `src/minrnn.py` (add)

- `MinLSTMLayer.step(x_t, h) -> h_new` for `x_t (B, in)`, `h (B, H)`:
  same maths as one iteration of `forward_sequential` (normalised gates,
  `g()` on the candidate when `log_space`).
- `MinLSTMSeq(input_size, hidden_size, num_layers, output_size, dropout,
  log_space=True, residual_norm=False)`:
  - `forward(x, mode="parallel"|"sequential") -> (B, L, out)` (every step).
  - `init_state(batch, device) -> list[Tensor]` (zeros, one per layer).
  - `step(x_t, state) -> (y_t (B, out), new_state)`.
  - `residual_norm=True`: input `Linear(in, H)` projection, then each layer is
    pre-norm residual `x = x + layer(LayerNorm(x))`, final `LayerNorm`, head.
  - Streaming always starts from the zero state (h_0 = None).

### 3.4 `src/models.py` (add)

- `LSTMSeq` (nn.LSTM / cuDNN): `forward(x, mode=None) -> (B, L, out)`;
  `init_state`, `step` carrying `(h, c)`.
- `PlainLSTMSeq`: built from existing `PlainLSTMLayer` maths, with `step`.
  `load_state_dict_from_cudnn(LSTMSeq)` for the equivalence test.
- Registry entries: `lstm_seq`, `plain_lstm_seq`, `minlstm_seq`,
  `minlstm_seq_res` (`residual_norm=True`). Existing entries untouched.

### 3.5 `src/train.py` (add)

- `train_seq_model(model, train, val, warmup, epochs, batch_size, lr,
  weight_decay, device, patience, grad_clip, epoch_callback=None)`:
  `train`/`val` are `(X, Y)` arrays; loss = MSE over `k >= warmup`.
  Same Adam + ReduceLROnPlateau + early stopping + best-state restore as
  `train_model`. `epoch_callback(epoch, val_loss)` is called each epoch (used
  for Optuna `trial.report` + `should_prune`; it may raise).
- `predict_seq(model, X, device, batch_size=256) -> (N, L, out)`.

### 3.6 `src/bench.py` (new)

- `time_train_step(name, seq_len, batch=64, hidden=64, layers=2, device,
  compile=False, warmup=5, iters=20) -> {"ms", "peak_mib"}`: forward + MSE on
  all steps + backward + Adam step on random data; CUDA-synchronised.
- `time_step_latency(name, device, hidden=64, layers=2, iters=500) -> µs`:
  one `step()` call at batch 1 (streaming inference cost per slot).
- `time_window_latency(name, window, device, ...) -> µs`: full forward on a
  `(1, window, 2)` input — the cost the windowed design pays every slot.
- `macs_per_step(name, input_size, hidden, layers, output_size) -> int`:
  analytic multiply-accumulates. minLSTM layer `3·in·H`; LSTM layer
  `4·(in+H)·H`; residual variant adds input projection `in·H`; head `H·out`.

### 3.7 `experiments/run_study.py` (new)

CLI: `--quick`, `--smoke` (tiny, CPU, < 2 min, for CI/local), `--device`,
`--out results`, `--skip-bench`, `--skip-search`, `--seed 0`.

| setting | smoke | quick | full |
|---|---|---|---|
| trajectories | 24 | 90 | 600 |
| Optuna trials / study | 2 | 4 | 30 |
| seeds | 1 | 2 | 5 |
| max epochs / patience | 3 / 2 | 40 / 8 | 200 / 20 |
| bench seq lens | 16, 64 | 16..512 | 16..4096 |

Phases (JSON written after each phase, so a crash keeps earlier results):

1. Corpus: `generate_corpus(T=200, σ=0.5, dt=0.1)`, split 70/15/15 by trajectory.
2. Classical: hold, CV, LS-linear (window ∈ {3,5,10,20}), KF-stream (q grid),
   IMM-stream (q × omega × p_stay grid) — all tuned on val, scored on test.
   IMM params chosen here feed hybrid mode.
3. Learned: archs `lstm_seq`, `minlstm_seq`, `minlstm_seq_res` × modes
   `direct`, `hybrid`. Optuna TPE (seed fixed) + MedianPruner wired via
   `epoch_callback`, equal trials per study. Search space: hidden {32,64,128},
   layers {1,2,3}, lr log-uniform [3e-4, 3e-3], dropout [0, 0.2],
   batch {16,32,64}, weight_decay {0, 1e-5, 1e-4}. Best config re-run over
   seeds → test MAE mean ± std, plus per-regime test MAE.
4. Generalisation: for each regime, train best config per arch×mode (seed 0) on
   the other two regimes, test on the held-out one; KF/IMM on the same set.
5. Bench: training-step time vs seq len for `lstm_seq` (cuDNN), `plain_lstm_seq`,
   `minlstm_seq`, `minlstm_seq` + `torch.compile`; per-slot step latency on the
   device and on CPU; windowed-forward latency at W ∈ {10, 40}; MACs table.
6. Save: `results/streaming_study.json`, `results/streaming_predictions.npz`
   (test truth + every method's predictions, same index set), figures
   `results/streaming_*.png`, best checkpoints `models/<arch>_<mode>.pt`.

### 3.8 `experiments/beam_eval.py` (new)

Reads `streaming_predictions.npz`. BS position from DeepMIMO `o1_60` if
importable, else `--bs-pos X Y`, else exit with a clear message. Reports
misalignment (half-power beamwidths) and outage (> 0.5 HPBW) for
Nt ∈ {8,16,32,64,128} per method. If DeepMIMO is available and
`--with-snr`, runs the twin SNR sweep (reusing `src/twin.py` +
`src/beamforming.py`). Writes `results/beam_eval.json`.

### 3.9 `experiments/make_tables.py` + `paper/`

`make_tables.py` turns the JSON files into `paper/tables/*.tex` so no number in
the paper is typed by hand. `paper/main.tex`: IEEEtran conference draft.

### 3.10 `docs/RUN_ON_GCP.md`

VM runbook: create VM, clone branch, install, `pytest`, `--quick` check,
full run under `tmux`, commit results back.

## 4. Testing (TDD, CPU-only, fast)

- `step()` loop == parallel forward == sequential forward (minLSTM both
  log/vanilla, residual variant, LSTMSeq, PlainLSTMSeq) to 1e-5.
- **Causality**: perturbing inputs at `k ≥ k0` leaves outputs `< k0` unchanged
  (every seq model).
- PlainLSTMSeq with weights copied from LSTMSeq matches it.
- `build_sequences`: shapes; `X`/`Y` alignment checked against hand-computed
  values on a tiny trajectory; hybrid features use only `obs[0..t]` (perturb
  `obs[t+1:]` → `X[:k]` unchanged); round-trip anchors + Y·scale == truth.
- KF/IMM stream causality; IMM ≈ KF on straight noisy lines; IMM beats KF on a
  constant-turn trajectory; `stream_window_predictor` matches the windowed
  predictor at `t ≥ window-1`.
- `train_seq_model` reduces loss on a toy problem; warm-up mask excludes early
  steps; callback is invoked and an exception propagates.
- `macs_per_step` hand-checked values; bench functions run on CPU at tiny size.
- `run_study.py --smoke` completes end to end on CPU and writes valid JSON.
