# syntax=docker/dockerfile:1
# Reuse the validated B3 Python/FA3 installation without inheriting its source layers.
FROM ghcr.io/joshuakauling/diffusion-arc@sha256:c87550eedf97f30d09b5a082e37a3dc38cada515185de4a650a6679e8bbfb4b3 AS dependencies
# Avoid the source checkout's egg-info shadowing the installed distribution.
WORKDIR /tmp
ENV PYTHONPATH=""
RUN python -m pip uninstall -y diffusion-arc \
    && python -m pip install --no-cache-dir kagglehub==1.0.2 \
    && python -m pip check

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6 AS runtime
ARG TARGETARCH
RUN test "$TARGETARCH" = "amd64" \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=dependencies /opt/python/ /opt/python/
COPY --from=dependencies /opt/venv/ /opt/venv/
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    KAGGLEHUB_CACHE=/workspace/.cache/kagglehub
RUN python - <<'CHECK'
import importlib.util
import torch
import flash_attn_3._C
from flash_attn_config import CONFIG
assert torch.__version__.split('+')[0] == '2.13.0'
assert torch.version.cuda == '13.0'
assert importlib.util.find_spec('diffusion_arc') is None
for feature in ('BACKWARD', 'VARLEN', 'HDIM128'):
    assert not CONFIG['build_flags'][f'FLASHATTENTION_DISABLE_{feature}']
CHECK

FROM runtime AS models
# Credentials and downloader source are build mounts, absent from the published image.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN,required=true \
    --mount=type=secret,id=kaggle_api_token,env=KAGGLE_API_TOKEN \
    --mount=type=secret,id=kaggle_username,env=KAGGLE_USERNAME \
    --mount=type=secret,id=kaggle_key,env=KAGGLE_KEY \
    --mount=type=bind,source=scripts/cloud/build_image_models.py,target=/tmp/build_image_models.py \
    python /tmp/build_image_models.py /opt/models

FROM scratch AS bundle-manifest
COPY --from=models /opt/models/manifest.json /manifest.json

FROM runtime AS final
# GHCR limits individual layers to 10 GB; each model stays below that limit.
COPY --from=models /opt/models/nvarc-qwen3-4b-grids15-sft139/ /opt/models/nvarc-qwen3-4b-grids15-sft139/
COPY --from=models /opt/models/flux2-klein-base-4b/ /opt/models/flux2-klein-base-4b/
COPY --from=models /opt/models/manifest.json /opt/models/manifest.json
RUN python -m pip check \
    && test ! -e /opt/diffusion-arc \
    && test -s /opt/models/manifest.json \
    && mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT []
CMD ["sleep", "infinity"]
