#!/usr/bin/env python3
"""Drive the rvc-training endpoint from your machine.

    uv run --no-project --with boto3 --with requests python3 rvc-training/client.py <command> ...

  upload   <local folder> --dataset ds6        WAVs -> volume datasets/ds6/
  train    --dataset ds6 [--vocoder RefineGAN] [--rate 32000] [--epochs 300] [--prefix my-voice]
           [--resume] [--overwrite] [--precision fp32|bf16] [--save-every 25]
  extract  <model> --epochs 50 100             playable models from earlier checkpoints
  status   <job id>                            check a job (train prints progress until done)
  ls       [prefix]                            list files on the volume (default logs/)
  download <model> [--out DIR] [--with-index] [--all]   final .pth (+ .index) from logs/<model>/

Settings come from the environment or rvc-training/.env (git-ignored):
  RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID                      the serverless endpoint
  RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY              an S3 API key (Runpod console > Settings)
  RUNPOD_VOLUME_ID, RUNPOD_DATACENTER (e.g. EU-RO-1)      the network volume
Files move through the volume's S3-compatible API, so no worker runs (or bills)
while you upload or download.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import boto3
import requests

HERE = Path(__file__).resolve().parent
AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def need(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        sys.exit(f"set {key} in the environment or {HERE / '.env'}")
    return v


def s3():
    dc = need("RUNPOD_DATACENTER")
    return boto3.client("s3", region_name=dc, endpoint_url=f"https://s3api-{dc.lower()}.runpod.io",
                        aws_access_key_id=need("RUNPOD_S3_ACCESS_KEY"),
                        aws_secret_access_key=need("RUNPOD_S3_SECRET_KEY"))


def api(path: str, payload: dict | None = None) -> dict:
    url = f"https://api.runpod.ai/v2/{need('RUNPOD_ENDPOINT_ID')}/{path}"
    head = {"Authorization": f"Bearer {need('RUNPOD_API_KEY')}"}
    r = requests.post(url, json=payload, headers=head, timeout=30) if payload is not None else \
        requests.get(url, headers=head, timeout=30)
    r.raise_for_status()
    return r.json()


def follow(job_id: str) -> dict:
    """Polls a job, printing progress, until it finishes."""
    last = None
    while True:
        st = api(f"status/{job_id}")
        msg = st.get("output") if st["status"] == "IN_PROGRESS" else None
        line = f"{st['status']}" + (f": {msg}" if msg else "")
        if line != last:
            print(time.strftime("%H:%M:%S"), line, flush=True)
            last = line
        if st["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return st
        time.sleep(20)


def cmd_upload(a) -> None:
    c, bucket = s3(), need("RUNPOD_VOLUME_ID")
    files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in AUDIO)
    if not files:
        sys.exit(f"no audio files in {a.folder}")
    prefix = f"rvc/datasets/{a.dataset}/"
    for p in files:
        print(f"upload {p.name} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
        c.upload_file(str(p), bucket, prefix + p.name)
    print(f"{len(files)} files -> {prefix}")


def cmd_train(a) -> None:
    job = {"action": "train", "dataset": a.dataset, "vocoder": a.vocoder, "sample_rate": a.rate,
           "epochs": a.epochs, "prefix": a.prefix, "precision": a.precision, "save_every": a.save_every,
           "resume": a.resume, "overwrite": a.overwrite}
    if a.model_name:
        job["model_name"] = a.model_name
    r = api("run", {"input": job})
    print("job", r["id"], flush=True)
    st = follow(r["id"])
    print(st.get("output") or st)


def cmd_extract(a) -> None:
    r = api("run", {"input": {"action": "extract", "model_name": a.model, "epochs_list": a.epochs}})
    print(follow(r["id"]).get("output"))


def cmd_status(a) -> None:
    print(api(f"status/{a.job_id}"))


def cmd_ls(a) -> None:
    c, bucket = s3(), need("RUNPOD_VOLUME_ID")
    pages = c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="rvc/" + a.prefix)
    for page in pages:
        for o in page.get("Contents", []):
            print(f"{o['Size'] / 1e6:10.1f} MB  {o['Key']}")


def cmd_download(a) -> None:
    c, bucket = s3(), need("RUNPOD_VOLUME_ID")
    out = Path(a.out or a.model)
    out.mkdir(parents=True, exist_ok=True)
    keys = [o["Key"] for p in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"rvc/logs/{a.model}/")
            for o in p.get("Contents", [])]
    import re
    models = [k for k in keys if re.search(r"_(\d+)e_(\d+)s\.pth$", k)]
    if not a.all:  # only the final (highest-epoch) model
        models = sorted(models, key=lambda k: int(re.search(r"_(\d+)e_", k).group(1)))[-1:]
    want = models + [k for k in keys if (a.with_index and k.endswith(".index")) or k.endswith("model_info.json")]
    if not want:
        sys.exit(f"nothing to download under rvc/logs/{a.model}/")
    for k in want:
        dest = out / Path(k).name
        print(f"download {k}", flush=True)
        c.download_file(bucket, k, str(dest))
    print(f"-> {out}")


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("upload"); p.add_argument("folder"); p.add_argument("--dataset", required=True)
    p.set_defaults(fn=cmd_upload)
    p = sub.add_parser("train"); p.add_argument("--dataset", required=True)
    p.add_argument("--vocoder", default="RefineGAN", choices=["HiFi-GAN", "RefineGAN", "MRF HiFi-GAN"])
    p.add_argument("--rate", type=int, default=32000, choices=[32000, 40000, 48000])
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--prefix", default="voice")
    p.add_argument("--model-name"); p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("extract"); p.add_argument("model"); p.add_argument("--epochs", type=int, nargs="+", required=True)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("status"); p.add_argument("job_id"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("ls"); p.add_argument("prefix", nargs="?", default="logs/"); p.set_defaults(fn=cmd_ls)
    p = sub.add_parser("download"); p.add_argument("model"); p.add_argument("--out")
    p.add_argument("--with-index", action="store_true")
    p.add_argument("--all", action="store_true", help="every saved small model, not just the final one")
    p.set_defaults(fn=cmd_download)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
