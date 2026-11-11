# Runpod serverless worker that trains an RVC voice model with Applio.
# A dataset folder on the network volume in, a trained .pth + .index out
# (also on the volume). See README.md.
FROM python:3.12-slim

# Applio at a known commit (main on 2026-09-27, release 3.6.5).
ARG APPLIO_COMMIT=3fd4fe2a50260d69dbdd4ca79c18609ba1420f39

ENV PYTHONUNBUFFERED=1 \
    APPLIO_DIR=/opt/applio

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ffmpeg libsndfile1 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# uv installs in parallel and far faster than pip. --torch-backend keeps
# PyTorch on its CUDA 12.8 build, which Blackwell GPUs (RTX 5090, RTX PRO 6000, B200) need.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

RUN git clone https://github.com/IAHispano/Applio "${APPLIO_DIR}" \
    && git -C "${APPLIO_DIR}" checkout "${APPLIO_COMMIT}" \
    && uv pip install --system --no-cache --torch-backend cu128 \
         -r "${APPLIO_DIR}/requirements.txt" runpod

# The pretrained bases (HiFi-GAN 32/40/48k, RefineGAN 24/32k), RMVPE/FCPE and
# ContentVec, about 1.8 GB, baked in so a cold worker never downloads them.
# Applio skips writing the final model without assets/config.json.
RUN cd "${APPLIO_DIR}" \
    && cp assets/config_template.json assets/config.json \
    && python core.py prerequisites --pretraineds-hifigan --models --no-exe \
    && ls rvc/models/pretraineds/hifi-gan/f0G32k.pth rvc/models/pretraineds/refinegan/f0G32k.pth \
         rvc/models/predictors/rmvpe.pt rvc/models/embedders/contentvec/pytorch_model.bin

# Paths are from the repository root, the build context Runpod's GitHub builder uses.
COPY rvc-training/handler.py rvc-training/start.sh /worker/
RUN chmod +x /worker/start.sh

# start.sh can run the newest handler.py from GitHub (HANDLER_REPO), falling
# back to the one copied in above.
CMD ["/worker/start.sh"]
