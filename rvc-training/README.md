# RVC training on Runpod

A Runpod serverless worker that trains an [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
voice model with [Applio](https://github.com/IAHispano/Applio). One job is one
training run: a dataset folder on the network volume in, a trained `.pth` and
`.index` out, written back to the volume.

Applio trains fine on Apple Silicon, but slowly: a 300-epoch RefineGAN 32k run
on ~28 minutes of singing took 6.6 h on an M4 Max (~80 s/epoch). On a CUDA GPU
it should take roughly an hour. Serverless bills by the second and scales to
zero, so an idle endpoint costs nothing.

## Layout

| File | What it does |
| --- | --- |
| `Dockerfile` | Python 3.12, PyTorch on CUDA 12.8 (needed for Blackwell: RTX 5090, RTX PRO 6000, B200), Applio at a pinned commit, and its pretrained bases baked in (~1.8 GB). |
| `handler.py` | The worker: `train` and `extract` actions (below). |
| `start.sh` | Starts the worker; with `HANDLER_REPO` set it runs the newest `handler.py` from GitHub, so handler changes need no rebuild. |
| `client.py` | Runs on your machine: uploads datasets, starts jobs, follows progress, downloads models. |
| `.env.example` | The settings `client.py` reads. Copy to `.env` (git-ignored). |

Everything large lives on the network volume, under `/runpod-volume/rvc`:

```
rvc/datasets/<dataset>/   WAVs to train on
rvc/logs/<model>/         Applio's run folder: G/D checkpoints, final .pth, .index
```

Checkpoints on the volume mean a run that times out can be continued with
`resume`, and any saved epoch can later be turned into a playable model.

## Actions

**`train`** (default)

| Input | Default | |
| --- | --- | --- |
| `dataset` | required | folder under `rvc/datasets/` |
| `vocoder` | `RefineGAN` | `HiFi-GAN`, `RefineGAN` or `MRF HiFi-GAN` |
| `sample_rate` | `32000` | `32000`, `40000` or `48000` |
| `epochs` | `300` | total epochs |
| `model_name` | `<prefix>-<dataset>-<voc><rate>-fresh` | e.g. `voice-ds6-rgan32k-fresh` |
| `prefix` | `voice` | |
| `batch_size` | `8` | |
| `save_every` | `25` | checkpoint interval (epochs) |
| `save_only_latest` | `false` | keep only the newest G/D pair |
| `precision` | `fp32` | `fp32`, `bf16` or `fp16` |
| `f0_method` | `rmvpe` | `rmvpe`, `fcpe` or `crepe` |
| `cache_in_gpu` | `true` | keep the dataset in GPU memory |
| `resume` | `false` | continue a run that has checkpoints (raise `epochs`) |
| `overwrite` | `false` | delete an existing run folder and start over |

Progress (epoch, time per epoch, lowest loss) is sent as Runpod progress
updates, which `client.py train` prints as they arrive.

**`extract`**: `model_name` plus `epochs_list`, e.g. `[50, 100]`. For each
epoch, it turns the nearest G checkpoint into a playable model in
`rvc/logs/<model>-e<N>/`, sharing the run's index.

### Defaults that differ from Applio's

- **Precision is fp32.** Applio's config template says fp16, which only takes
  effect on CUDA; on a Mac it trains in fp32. fp32 keeps cloud runs comparable
  with Mac runs. `bf16` is faster but changes training numerics.
- **A missing pretrained base is an error.** For a vocoder/rate with no base
  (MRF HiFi-GAN has none; RefineGAN has only 24k/32k), Applio silently trains
  from random weights. The worker refuses instead.
- **Existing runs aren't overwritten.** Pass `resume` or `overwrite`.

Checkpoint sizes add up: a RefineGAN 32k run saving every 25 epochs wrote
about 12 GB of G/D checkpoints over 300 epochs. Size the volume for that, or use
`save_only_latest` or a larger `save_every`.

## Deploying

1. **Network volume.** Pick a datacenter that has the GPUs you want and
   supports the S3-compatible API (EU-RO-1 does). The volume pins the endpoint
   to that datacenter.

   ```bash
   runpodctl gpu list            # stock per datacenter
   runpodctl network-volume create --name rvc-training --size 50 --data-center-id EU-RO-1
   ```

2. **Endpoint.** In the Runpod console: **Serverless → New Endpoint → Import
   Git Repository**, this repo, **Dockerfile Path** `rvc-training/Dockerfile`
   (the build context is the repo root).

   | Setting | Value |
   | --- | --- |
   | GPUs | RTX 5090 and RTX PRO 6000 tiers (Blackwell; 24 GB+ is plenty) |
   | Active workers | 0 |
   | Max workers | 1 (two runs at once would share the volume) |
   | Idle timeout | 5 s |
   | Execution timeout | 10800 s (3 h); a 300-epoch run is ~1–1.5 h |
   | Container disk | 30 GB |
   | FlashBoot | on |
   | Network volume | the one from step 1 |

   Optional environment: `HANDLER_REPO=<owner>/<repo>` (and `HANDLER_REF`) to
   run the newest `handler.py` from GitHub on each cold start.

3. **S3 API key.** Runpod console → Settings → S3 API Keys. Fill in
   `rvc-training/.env` from `.env.example`.

## Running

```bash
c="uv run --no-project --with boto3 --with requests python3 rvc-training/client.py"
$c upload ~/voice-dataset/train-v6 --dataset ds6
$c train --dataset ds6 --vocoder RefineGAN --rate 32000 --prefix my-voice
$c train --dataset ds6 --vocoder HiFi-GAN  --rate 32000 --prefix my-voice
$c ls logs/
$c download my-voice-ds6-rgan32k-fresh --with-index --out models/
$c extract my-voice-ds6-rgan32k-fresh --epochs 50 100
```

Uploads and downloads go straight to the volume through its S3 API, so no
worker runs, or bills, while files move.

Before a long run on a new GPU type, a short one (`--epochs 5`) shows the real
seconds per epoch.

## Testing the handler without Runpod

`handler.handler()` can be called directly against a local Applio checkout,
with a stub `runpod` module and `RVC_GPU=-` on a Mac (Applio's `--gpu 0` maps
to `cuda:0` and silently extracts nothing there). Point `APPLIO_DIR` at a
**copy** of Applio: the handler replaces `<applio>/logs` with a link to
`RVC_VOLUME/logs`.
