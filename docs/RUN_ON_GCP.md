# Running the streaming study on a GCP GPU

Everything below runs on the VM unless marked **local**. Expect roughly 20-40 min for
the quick check and about 3-4 hours for the full run on a T4.

## 0. Before you start (GCP console)

- **GPU quota.** IAM & Admin → Quotas: `GPUS_ALL_REGIONS` ≥ 1 and the regional
  `NVIDIA T4 GPUs` quota ≥ 1 in the zone you pick. New projects start at 0;
  request an increase if needed.
- **Billing.** Free-trial accounts generally cannot attach GPUs until upgraded to
  a paid account (the credits still apply). Check this in your console.

## 1. Create the VM

A Deep Learning VM image has CUDA, drivers and PyTorch preinstalled. List the
current PyTorch image families first, because their names change between
releases:

```bash
gcloud compute images list --project deeplearning-platform-release \
  --filter="family~pytorch AND family~cu" --format="value(family)" | sort -u
```

Then create the VM, substituting a family from that list:

```bash
gcloud compute instances create thz-minlstm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-project=deeplearning-platform-release \
  --image-family=<pytorch-family-from-the-list> \
  --boot-disk-size=100GB \
  --metadata=install-nvidia-driver=True

gcloud compute ssh thz-minlstm --zone=us-central1-a
```

The T4 matters for comparability: the minLSTM paper measured its speedups on a
T4. Any other GPU is fine, as long as the paper reports which one was used.

## 2. Get the code

```bash
git clone -b minlstm-streaming https://github.com/aarushag02/thz-beamforming.git
cd thz-beamforming
```

If the repository is private, authenticate first (`gh auth login`, or a GitHub
personal access token used as the HTTPS password).

## 3. Install and verify

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q
```

The second line must print `True` and the GPU name. If it prints `False`, the
driver did not install: run `sudo /opt/deeplearning/install-driver.sh`, then
reconnect.

## 4. Quick check (~20-40 min)

```bash
python experiments/run_study.py --quick --out results/quick --models models/quick
```

Confirm `results/quick/streaming_study.json` exists and the log shows every
phase (classical, learned, held-out, bench). Bring the log back if anything
looks off before spending hours on the full run.

## 5. Full run (hours), surviving SSH disconnects

```bash
tmux new -s study
python experiments/run_study.py 2>&1 | tee results/full_run.log
# detach: Ctrl-b then d      reattach later: tmux attach -t study
```

Results are written after each phase and after each learned configuration. After a
crash or preemption, rerun the same command with `--resume` to skip finished
configurations.

## 6. Beam misalignment and tables

```bash
python -c "import deepmimo as dm; dm.download('O1_60')"   # ~2.3 GB, once
python experiments/beam_eval.py                           # BS position from o1_60 tx_pos
python experiments/make_tables.py
```

If the DeepMIMO download fails, run `beam_eval.py --bs-pos X Y` with the
base-station coordinates instead.

## 7. Bring the results back

```bash
git add results/streaming_study.json results/streaming_predictions.npz \
        results/streaming_*.png results/beam_eval.json results/full_run.log paper/tables
git commit -m "Add full streaming study results (T4)"
git push origin minlstm-streaming
```

**local:** `git pull` on the same branch.

## 8. Stop paying for the GPU

```bash
gcloud compute instances stop thz-minlstm --zone=us-central1-a     # keeps the disk
# gcloud compute instances delete thz-minlstm --zone=us-central1-a # removes everything
```
