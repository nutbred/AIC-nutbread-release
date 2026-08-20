# AIC-nutbread (online release)

Local Vietnamese video retrieval server. This repository contains **only the online
runtime code** — the Flask UI + API, retrieval/fusion logic, and the Docker packaging.
Offline preprocessing tools (frame extraction, embedding building, index construction,
ASR/leader tooling) are intentionally **not** included here.

## What you need to run it

1. **Docker Desktop** (WSL2 backend) — [docker.com](https://www.docker.com/products/docker-desktop/)
2. **An NVIDIA GPU** with a recent driver (RTX 4060 / 8 GB works; 8 GB is the tested profile)
3. **The all-in-one data + model zip** (`aic-data-full.zip`, ~39 GB)

The zip is **not** in this repo (too large for GitHub). It contains the full retrieval
dataset and indexes (Keyframes L21-L30 including all subseries, ASR, Qwen index, OCR,
maps, representative captions, SigLIP index, L23 scene metadata) **and** the three model
weights (Qwen3-VL-Embedding-2B, BGE-M3, SigLIP-so400m) under `huggingface/hub/` so
everything loads **fully offline** (`AIC_OFFLINE_MODELS=1`).

> **Download link: { PUT YOUR SHARE LINK HERE }**

## Setup — step by step

### 1. Get the code
```powershell
git clone git@github.com:nutbred/AIC-nutbread-release.git
cd AIC-nutbread-release
```

### 2. Extract the zip
Extract `aic-data-full.zip` so it makes a folder named **`AIC-data`**, e.g.
`C:\Users\<you>\AIC-data\`.

### 3. Configure
```powershell
Copy-Item .env.example .env
notepad .env
```
Set these to your machine (forward slashes), keeping offline mode ON:

```ini
HOST_WORKSPACE_ROOT=C:/Users/<you>/AIC-data
HOST_5FPS_DATA=C:/Users/<you>/AIC-data/kaggle/working/data
HOST_SIGLIP_INDEX_DIR=C:/Users/<you>/AIC-data/aic-runtime/indexes
HOST_REPRESENTATIVE_EMBEDDINGS=C:/Users/<you>/AIC-data/representative_bge_m3_embeddings.npy
HOST_HF_CACHE_DIR=C:/Users/<you>/AIC-data/huggingface
AIC_OFFLINE_MODELS=1
PORT=5000
HOST_BIND_ADDRESS=127.0.0.1
```

### 4. Start
```powershell
.\run-docker.ps1 -Build
```
Open **http://127.0.0.1:5000**. Stop with `.\stop-docker.ps1`.

### 5. Validate
```powershell
.\test-docker.ps1
```

## Share with one friend over Tailscale (private)

The app has **no login** — never expose it directly to the public internet.
For a single trusted person, the safe way is a private Tailscale mesh VPN:

```powershell
# 1. Both of you install Tailscale and sign into the SAME account
# 2. Run the helper on this machine
.\share-tailscale.ps1
#    -> prints the URL, e.g.  http://100.101.102.103:5000
```

Your friend opens that URL. Only devices on your tailnet can reach it — it is not
on the public internet. To go back to localhost-only:

```powershell
.\share-tailscale.ps1 -Off
```

> `share-tailscale.ps1` sets `HOST_BIND_ADDRESS=0.0.0.0` in `.env`, recreates the
> container, and prints your Tailscale URL. It also sets the value back to
> `127.0.0.1` with `-Off`. If it can't auto-detect your Tailscale IP, run
> `tailscale ip -4` and type it in.

## Model resolution

The release uses `AIC_OFFLINE_MODELS=1` (models shipped in the zip's `huggingface/`).
If you have the zip **without** models, set `AIC_OFFLINE_MODELS=0` and the app will
auto-download Qwen3-VL-Embedding-2B, BGE-M3, and SigLIP from HuggingFace on first boot
(all public, not gated). SigLIP defaults to CPU; Qwen and BGE run on CUDA.

## Architecture (online runtime)

- `app.py` — Flask server (search, temporal-search, previews, KIS/VQA export)
- `retrieval.py` — source wrappers (Qwen3-VL random, BGE-M3/ASR, representative,
  SigLIP 5-FPS, BTC clip) + time fusion + evaluator-aware ranking
- `preview.py` — frame preview resolution
- `submission.py` — KIS / temporal CSV exports
- `l23_scenes.py` — L23 scene & clothing-color metadata
- `docker/`, `Dockerfile`, `docker-compose.yml` — reproducible GPU container

## Limits

- Search quality depends on the supplied indexes/maps/ASR/OCR data and model cache.
- Raw videos are optional (needed only for FFmpeg preview fallback).
- Private / local use only.
