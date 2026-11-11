"""Runpod serverless handler: train an RVC voice model with Applio.

Everything big lives on the endpoint's network volume, under /runpod-volume/rvc:
  datasets/<dataset>/   the WAVs to train on (upload with client.py upload)
  logs/<model>/         Applio's run folder: checkpoints, final .pth, .index

Input (the job's "input"):
  action        "train" (default) or "extract"
  --- train ---
  dataset       folder name under datasets/ (required)
  vocoder       "HiFi-GAN" | "RefineGAN" | "MRF HiFi-GAN"   (default RefineGAN)
  sample_rate   32000 | 40000 | 48000                       (default 32000)
  epochs        total epochs                                (default 300)
  model_name    default <prefix>-<dataset>-<voc><rate>-fresh, e.g. voice-ds6-rgan32k-fresh
  prefix        name prefix for the default model_name      (default "voice")
  batch_size    (default 8, the size the Mac runs used)
  save_every    checkpoint interval in epochs               (default 25)
  save_only_latest  keep only the newest G/D checkpoint     (default false)
  precision     "fp32" | "bf16" | "fp16"                    (default fp32)
  f0_method     "rmvpe" | "fcpe" | "crepe"                  (default rmvpe)
  cache_in_gpu  keep the dataset in GPU memory              (default true)
  resume        continue a run that has checkpoints (same model_name; raise epochs)
  overwrite     wipe an existing run folder and start fresh (default false)
  --- extract ---
  model_name    an existing run
  epochs_list   e.g. [50, 100]: make <model>-e<N>/<model>-e<N>_<N>e_<step>s.pth
                from the G checkpoint nearest each epoch (playable, shares the index)

Output: {"model", "files": {path: bytes}, "epochs", "lowest_loss", "seconds",
"handler", "log": [...]}. Files stay on the volume; client.py download fetches them.

Applio's defaults that differ from what we want are set here on purpose:
- precision: Applio's template says fp16, which applies on CUDA only. Our Mac runs
  were fp32, so fp32 is the default to keep cloud runs comparable.
- the pretrained base: Applio silently trains from random weights when the
  vocoder/rate base is missing (MRF HiFi-GAN has none), so that is an error here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import runpod

APPLIO = Path(os.environ.get("APPLIO_DIR", "/opt/applio"))
VOLUME = Path(os.environ.get("RVC_VOLUME", "/runpod-volume/rvc"))
DATASETS = VOLUME / "datasets"
LOGS = VOLUME / "logs"
PY = sys.executable
GPU = os.environ.get("RVC_GPU", "0")  # "-" for CPU extraction when testing on a Mac
VOCODER_TAG = {"HiFi-GAN": "hfg", "RefineGAN": "rgan", "MRF HiFi-GAN": "mrf"}
EPOCH_LINE = re.compile(r"epoch=(\d+).*?training_speed=([0-9:]+)(?:.*?lowest_value=([0-9.]+))?")


def link_logs(log: list[str]) -> None:
    """Applio writes to <applio>/logs; point it at the volume so runs persist."""
    LOGS.mkdir(parents=True, exist_ok=True)
    target = APPLIO / "logs"
    if target.is_symlink() and target.resolve() == LOGS.resolve():
        return
    if target.exists() and not target.is_symlink():
        for item in target.iterdir():  # keep what Applio ships there (logs/mute, used by --include-mutes)
            dest = LOGS / item.name
            if not dest.exists():
                shutil.move(str(item), dest)
        shutil.rmtree(target)  # whatever is left was already on the volume
    elif target.is_symlink():
        target.unlink()
    target.symlink_to(LOGS)
    log.append(f"logs -> {LOGS}")


def set_precision(precision: str) -> None:
    cfg_path = APPLIO / "assets" / "config.json"
    if not cfg_path.exists():
        shutil.copy(APPLIO / "assets" / "config_template.json", cfg_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["precision"] = precision
    cfg_path.write_text(json.dumps(cfg, indent=2))


def run(cmd: list[str], log: list[str]) -> str:
    log.append("$ " + " ".join(cmd))
    done = subprocess.run(cmd, cwd=APPLIO, capture_output=True, text=True)
    out = done.stdout + done.stderr
    log.append(out[-3000:])
    if done.returncode:
        raise RuntimeError(f"{' '.join(cmd[1:3])[:60]} exited {done.returncode}")
    return out


def train_streaming(cmd: list[str], job, log: list[str], epochs: int) -> dict:
    """Runs training, sending a progress update to Runpod once per epoch."""
    log.append("$ " + " ".join(cmd))
    stats = {"last_epoch": 0, "lowest_loss": None, "epoch_time": None}
    tail: list[str] = []
    proc = subprocess.Popen(cmd, cwd=APPLIO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        for part in line.replace("\r", "\n").splitlines():
            m = EPOCH_LINE.search(part)  # Applio prints it on the same line as the tqdm bar
            if not m and (not part.strip() or "it/s]" in part or "s/it]" in part):
                continue  # tqdm bars
            tail = (tail + [part[part.find(" | epoch=") - 40:] if m else part])[-60:]
            if m:
                stats["last_epoch"] = int(m.group(1))
                stats["epoch_time"] = m.group(2)
                if m.group(3):
                    stats["lowest_loss"] = float(m.group(3))
                runpod.serverless.progress_update(
                    job, f"epoch {m.group(1)}/{epochs}, {m.group(2)}/epoch, lowest loss {m.group(3)}")
    code = proc.wait()
    log.append("\n".join(tail))
    if code:
        raise RuntimeError(f"train exited {code}")
    return stats


def model_files(name: str) -> dict[str, int]:
    run_dir = LOGS / name
    found = sorted(run_dir.glob(f"{name}_*e_*s.pth")) + sorted(run_dir.glob("*.index"))
    return {str(p.relative_to(VOLUME)): p.stat().st_size for p in found}


def do_train(job, inp: dict, log: list[str]) -> dict:
    dataset = inp["dataset"]
    vocoder = inp.get("vocoder", "RefineGAN")
    sr = int(inp.get("sample_rate", 32000))
    epochs = int(inp.get("epochs", 300))
    if vocoder not in VOCODER_TAG:
        raise ValueError(f"vocoder must be one of {list(VOCODER_TAG)}")
    name = inp.get("model_name") or f"{inp.get('prefix', 'voice')}-{dataset}-{VOCODER_TAG[vocoder]}{str(sr)[:2]}k-fresh"
    data_dir = DATASETS / dataset
    if not data_dir.is_dir() or not any(data_dir.iterdir()):
        raise ValueError(f"no dataset at {data_dir}; upload it first")

    # Refuse to train from random weights (Applio's selector returns "" silently).
    base = APPLIO / "rvc" / "models" / "pretraineds" / vocoder.lower()
    for kind in "GD":
        p = base / f"f0{kind}{str(sr)[:2]}k.pth"
        if not p.exists():
            raise ValueError(f"no pretrained base {p.name} for {vocoder} at {sr} Hz")

    run_dir = LOGS / name
    has_ckpt = any(run_dir.glob("G_*.pth"))
    resume = bool(inp.get("resume"))
    if has_ckpt and not resume:
        if not inp.get("overwrite"):
            raise ValueError(f"{name} already has checkpoints: pass resume (continue) or overwrite (start over)")
        shutil.rmtree(run_dir)
        has_ckpt = False
    if resume and not has_ckpt:
        raise ValueError(f"{name} has no checkpoints to resume from")

    set_precision(inp.get("precision", "fp32"))
    cores = str(os.cpu_count() or 8)
    t0 = time.time()
    if not resume:
        run([PY, "core.py", "preprocess", "--model-name", name, "--dataset-path", str(data_dir),
             "--sample-rate", str(sr), "--cpu-cores", cores, "--cut-preprocess", "Automatic",
             "--chunk-len", "3.0", "--overlap-len", "0.3", "--normalization-mode", "post"], log)
        run([PY, "core.py", "extract", "--model-name", name, "--f0-method", inp.get("f0_method", "rmvpe"),
             "--cpu-cores", cores, "--gpu", GPU, "--sample-rate", str(sr),
             "--embedder-model", "contentvec", "--include-mutes", "2"], log)
    chunks = len(list((run_dir / "sliced_audios").glob("*.wav")))
    log.append(f"chunks: {chunks}")
    runpod.serverless.progress_update(job, f"prepared {chunks} chunks; training {name}")

    cmd = [PY, "core.py", "train", "--model-name", name, "--vocoder", vocoder,
           "--save-every-epoch", str(inp.get("save_every", 25)), "--total-epoch", str(epochs),
           "--sample-rate", str(sr), "--batch-size", str(inp.get("batch_size", 8)), "--gpu", "0",
           "--pretrained", "--index-algorithm", "Auto"]
    if inp.get("save_only_latest"):
        cmd.append("--save-only-latest")
    if inp.get("cache_in_gpu", True):
        cmd.append("--cache-data-in-gpu")
    stats = train_streaming(cmd, job, log, epochs)

    if not any(run_dir.glob("*.index")):
        run([PY, "core.py", "index", "--model-name", name, "--index-algorithm", "Auto"], log)
    files = model_files(name)
    done = [int(m.group(1)) for k in files if (m := re.search(r"_(\d+)e_\d+s\.pth$", k))]
    if not done or max(done) < epochs:
        raise RuntimeError(f"training finished but no {name}_{epochs}e_*s.pth was written (got epochs {done})")
    # Applio prints no epoch line for the last epoch, so the final model's name is the count.
    return {"model": name, "chunks": chunks, "epochs": max(done), "epoch_time": stats["epoch_time"],
            "lowest_loss": stats["lowest_loss"], "seconds": round(time.time() - t0), "files": files}


EXTRACT_SNIPPET = r"""
import json, sys, torch
sys.path.insert(0, ".")
from rvc.train.process.extract_model import extract_model
from rvc.train.utils import HParams
src, name, out, epoch, step, final = sys.argv[1:7]
cfg = json.load(open(f"{src}/config.json"))
vocoder = torch.load(final, map_location="cpu", weights_only=False).get("vocoder", "HiFi-GAN")
ck = torch.load(f"{src}/G_{step}.pth", map_location="cpu", weights_only=False)
extract_model(ckpt=ck["model"], sr=cfg["data"]["sample_rate"], name=name, model_path=out,
              epoch=int(epoch), step=int(step), hps=HParams(**cfg), vocoder=vocoder)
"""


def do_extract(inp: dict, log: list[str]) -> dict:
    name = inp["model_name"]
    src = LOGS / name
    steps = sorted(int(p.stem.split("_")[1]) for p in src.glob("G_*.pth"))
    if not steps:
        raise ValueError(f"{name} has no G checkpoints")
    final = next(iter(src.glob(f"{name}_*e_*s.pth")), None)
    if final is None:
        raise ValueError(f"{name} has no final model to read steps-per-epoch from")
    m = re.search(r"_(\d+)e_(\d+)s\.pth$", final.name)
    per_epoch = int(m.group(2)) / int(m.group(1))
    index = next(iter(src.glob("*.index")), None)
    made = {}
    for epoch in inp["epochs_list"]:
        step = min(steps, key=lambda s: abs(s - epoch * per_epoch))
        ep = round(step / per_epoch)
        sub = f"{name}-e{ep}"
        dst = LOGS / sub
        dst.mkdir(exist_ok=True)
        if (src / "model_info.json").exists():
            shutil.copy(src / "model_info.json", dst / "model_info.json")
        out = dst / f"{sub}_{ep}e_{step}s.pth"
        run([PY, "-c", EXTRACT_SNIPPET, str(src), sub, str(out), str(ep), str(step), str(final)], log)
        if index:
            link = dst / f"{sub}.index"
            if not link.exists():
                link.symlink_to(Path("..") / name / index.name)
        made[sub] = str(out.relative_to(VOLUME))
    return {"model": name, "extracted": made}


def handler(job):
    inp = job.get("input") or {}
    log: list[str] = [f"handler: {os.environ.get('HANDLER_SOURCE', 'local')}"]
    try:
        link_logs(log)
        action = inp.get("action", "train")
        if action == "train":
            result = do_train(job, inp, log)
        elif action == "extract":
            result = do_extract(inp, log)
        else:
            raise ValueError(f"unknown action {action!r}")
        return {**result, "handler": os.environ.get("HANDLER_SOURCE", "local"), "log": log[-12:]}
    except Exception as e:  # report, don't crash the worker
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:],
                "handler": os.environ.get("HANDLER_SOURCE", "local"), "log": log[-12:]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
