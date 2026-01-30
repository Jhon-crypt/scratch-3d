# 3D Asset Pipeline — Production Infrastructure

Production-grade pipeline: **Prompt or images → multi-view images → 3D mesh**.  
Deterministic orchestration; each step is an isolated service.

## High-Level Concept

The system is a **pipeline**, not a single model:

1. **Images** are generated (FLUX) or accepted (uploads/URLs)
2. **Views** are normalized into canonical multi-view set
3. **3D reconstruction** (TripoSR/InstantMesh) converts views → geometry
4. **Post-processing** (optional) cleans mesh, normals, UVs

## Core Services

| Service | Role | ML? |
|--------|------|-----|
| **Orchestrator** | Control plane: accept requests, dispatch jobs, track state | No |
| **FLUX Service** | Real FLUX.1-schnell (or dev) image generation; one model load per GPU | Yes (GPU) |
| **Image Service** | Multi-view prompts → calls FLUX or stub; neutral backgrounds | Orchestrator |
| **Reconstruction** | TripoSR/InstantMesh: images → mesh | Yes (GPU 4–7) |
| **Post-Process** | Decimate, normals, UV; headless Blender or trimesh | No (CPU) |

## API Surface

### Orchestrator (public)

- `POST /3d/generate` — Prompt → 3D  
  Body: `{ "prompt": "...", "output_format": "glb"|"obj"|"fbx", "quality_tier": "standard"|"high" }`  
  Returns: `{ "job_id", "status_url" }`

- `POST /3d/from-image` — Image(s) → 3D  
  Body: `{ "image_urls": [...] }` or `{ "image_paths": [...] }`, `output_format`, `quality_tier`  
  Returns: `{ "job_id", "status_url" }`

- `GET /3d/status/{job_id}` — Job state and `asset_path` when done  
- `GET /3d/download/{job_id}` — Download final asset file

### Internal (service-to-service)

- **Image**: `POST /imagine/generate` (prompts), `POST /imagine/from-urls` (urls)
- **Reconstruction**: `POST /reconstruct/mesh` (image_paths, output_format)
- **Post-process**: `POST /postprocess/clean` (mesh_path, output_format)

## Execution Paths

**Path A — Prompt to 3D**

1. Orchestrator receives prompt  
2. Builds multi-view prompts (4 or 8 views)  
3. Image service (FLUX) generates 4–8 images  
4. Reconstruction service produces mesh  
5. Optional post-processing  
6. Asset saved under `/outputs`, returned via status/download  

**Path B — Image to 3D**

1. Orchestrator receives image URLs or paths under `/inputs`  
2. Images validated/normalized (download if URLs)  
3. Reconstruction → mesh → optional post-process → save  

## Storage Layout

All services mount the same volumes:

- `/models` — model weights
- `/inputs` — uploaded or generated images
- `/outputs` — final 3D assets
- `/cache` — temporary files

## Queue and State (Redis)

- **Queue**: `3d:queue` (list of job_id)
- **State**: `job:{job_id}` hash with `state`, `message`, `asset_path`

**Job states**: `queued` → `generating_images` → `reconstructing_3d` → `post_processing` → `completed` | `failed`

## Domain: elohim-bitch-gpu.insanelabs.org (no port)

The API is served at **http://elohim-bitch-gpu.insanelabs.org** (port 80, no port in the URL).

- **Host nginx** on this machine proxies that hostname to the orchestrator (`127.0.0.1:38100`).
- **Config**: `/etc/nginx/sites-available/elohim-bitch-gpu.insanelabs.org` (symlinked in `sites-enabled/`).
- **DNS**: Point `elohim-bitch-gpu.insanelabs.org` to this host’s public IP; port 80 must be open.
- **Endpoints**:
  - `POST http://elohim-bitch-gpu.insanelabs.org/3d/generate`
  - `GET http://elohim-bitch-gpu.insanelabs.org/3d/status/{job_id}`
  - `GET http://elohim-bitch-gpu.insanelabs.org/3d/download/{job_id}`
  - `GET http://elohim-bitch-gpu.insanelabs.org/health`

For HTTPS, add a cert (e.g. `certbot --nginx -d elohim-bitch-gpu.insanelabs.org`) and a 443 server block in that config.

## UI at /3d-test (runs automatically)

The UI is built and served by Docker. No manual `npm run build` needed.

- **Start stack**: `cd scratch-3d && docker-compose up -d` — the **ui** service builds (Node 22) and serves on port 38101; host nginx proxies **/3d-test/** to it.
- **Open**: **http://elohim-bitch-gpu.insanelabs.org/3d-test/**
- **Start on boot**: Copy `scripts/scratch-3d.service` to `/etc/systemd/system/`, then `sudo systemctl enable scratch-3d` and `sudo systemctl start scratch-3d`. The full stack (API + UI + FLUX + reconstruction + Redis) will start automatically on boot.
- **Dev** (optional): `cd ui/test_3d && npm install && npm run dev` — app at `http://localhost:5174/3d-test/` with API proxied to the domain.

## Where are the outputs?

Final 3D assets (`.glb`, `.obj`, `.fbx`) are written to **`scratch-3d/outputs/`**.  
Generated or uploaded images live in **`scratch-3d/inputs/`**.  
Docker Compose binds these so you see files directly in your project folder.

## Run Locally (Docker)

```bash
cd scratch-3d
docker compose up --build
```

Orchestrator: **http://localhost:8000** (or the host port you map in `docker-compose.yml`)

- Health: `GET http://localhost:8000/health`
- Generate from prompt:  
  `curl -X POST http://localhost:8000/3d/generate -H "Content-Type: application/json" -d '{"prompt": "a red cube", "output_format": "glb"}'`
- Status: `GET http://localhost:8000/3d/status/{job_id}`
- Download: `GET http://localhost:8000/3d/download/{job_id}`

## GPU Allocation (production)

- **GPUs 0–3**: FLUX image generation (one container per GPU; set `CUDA_VISIBLE_DEVICES` per replica)
- **GPUs 4–7**: 3D reconstruction (one container per GPU)

Uncomment the `deploy.resources.reservations.devices` blocks in `docker-compose.yml` and scale:

```yaml
# Example: 2 image-service replicas on GPU 0 and 1
image-service:
  deploy:
    replicas: 2
```

One job per GPU at a time; models loaded once per container (hot).

## Design Principles

- **Stateless models** — Services only do one job  
- **Orchestrator owns logic** — Execution path and job flow  
- **Reproducible** — Same input → same pipeline steps  
- **Replaceable** — Swap FLUX/TripoSR/Blender without changing API contract  

## FLUX Service (real FLUX inside scratch-3d)

The repo includes a **FLUX service** (`flux-service/`) that runs **FLUX.1-schnell** (or FLUX.1-dev) via Hugging Face diffusers. It loads the model once at startup and stays hot.

- **Location**: `scratch-3d/flux-service/`
- **API**: `POST /generate` with `prompt`, `width`, `height`, optional `seed`; returns `image_path` and `image_b64`.
- **Default model**: `black-forest-labs/FLUX.1-schnell` (fast, 1–4 steps). Set `FLUX_MODEL=black-forest-labs/FLUX.1-dev` for higher quality (~50 steps).
- **Requires**: GPU (NVIDIA); ~24GB+ VRAM for schnell, or use CPU offload (slower).

Docker Compose wires the image-service to `http://flux-service:8000` by default. If you have no GPU, either:

1. Run without the FLUX service: comment out the `flux-service` block and the image-service `depends_on: flux-service`, then set image-service env `USE_IMAGE_STUB=true` and `FLUX_BACKEND_URL=""`, or  
2. Use stub-only: `USE_IMAGE_STUB=true` and leave `FLUX_BACKEND_URL` empty so the pipeline uses placeholder images.

## Current stubs

Out of the box (no GPU):

- **Image service**: stub (placeholder PNGs) unless `FLUX_BACKEND_URL` is set and `USE_IMAGE_STUB=false`.
- **Reconstruction**: stub (trimesh cube or minimal OBJ) until TripoSR/InstantMesh is wired.
- **Post-process**: trimesh or copy-as-is.

To go production: point image-service at your FLUX API, add TripoSR/InstantMesh in reconstruction-service, and optionally headless Blender in postprocess-service.
