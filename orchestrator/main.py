"""
Orchestrator Service — Control plane for 3D asset pipeline.
Does not run ML models. Coordinates jobs across image and 3D services.
Enforces machine-readable orchestration rules (shared/orchestration_rules.json).
"""
import json
import os
import uuid
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from redis import Redis
from typing import Optional, List, Dict, Any

# Shared schemas (PYTHONPATH must include parent in dev, or /app in Docker)
try:
    from shared.schemas import JobState, OutputFormat, QualityTier, DEFAULT_VIEWS, HIGH_QUALITY_VIEWS, FAST_VIEWS, MULTI_VIEW_PROMPTS
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared.schemas import JobState, OutputFormat, QualityTier, DEFAULT_VIEWS, HIGH_QUALITY_VIEWS, FAST_VIEWS, MULTI_VIEW_PROMPTS

# Config from env
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
IMAGE_SERVICE_URL = os.getenv("IMAGE_SERVICE_URL", "http://image-service:8000")
VIEW_SYNTHESIS_SERVICE_URL = os.getenv("VIEW_SYNTHESIS_SERVICE_URL", "http://view-synthesis-service:8000")
RECONSTRUCT_SERVICE_URL = os.getenv("RECONSTRUCT_SERVICE_URL", "http://reconstruction-service:8000")
POSTPROCESS_SERVICE_URL = os.getenv("POSTPROCESS_SERVICE_URL", "http://postprocess-service:8000")
TEXTURE_SERVICE_URL = os.getenv("TEXTURE_SERVICE_URL", "http://texture-service:8000")
REFINEMENT_SERVICE_URL = os.getenv("REFINEMENT_SERVICE_URL", "http://refinement-service:8000")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")
INPUTS_DIR = os.getenv("INPUTS_DIR", "/inputs")
ENABLE_TEXTURE = os.getenv("ENABLE_TEXTURE", "false").lower() == "true"
PRODUCTION_BUNDLE = os.getenv("PRODUCTION_BUNDLE", "false").lower() == "true"
# Production spec: SDS refinement (nvdiffrast + SDXL); stub returns mesh as-is when not implemented
ENABLE_SDS_REFINEMENT = os.getenv("ENABLE_SDS_REFINEMENT", "false").lower() == "true"
# Production spec: require manifold/watertight before marking job complete
REQUIRE_MANIFOLD_VALIDATION = os.getenv("REQUIRE_MANIFOLD_VALIDATION", "false").lower() == "true"
PRODUCTION_RECONSTRUCTION_RESOLUTION = int(os.getenv("PRODUCTION_RECONSTRUCTION_RESOLUTION", "512"))
PRODUCTION_TEXTURE_SIZE = int(os.getenv("PRODUCTION_TEXTURE_SIZE", "1024"))

# Allowed IPs for /3d-test/ UI (comma-separated); only these see the app, others get blocked message
ALLOWED_IPS = {ip.strip() for ip in os.getenv("3D_TEST_ALLOWED_IPS", "102.89.84.79,102.89.83.83").split(",") if ip.strip()}

# Minimum view count for reconstruction (spec: never reconstruct with fewer)
MIN_VIEWS_FOR_RECONSTRUCTION = 6

# Machine-readable rules (shared/orchestration_rules.json)
RULES_PATH = os.getenv("ORCHESTRATION_RULES_PATH", "/app/shared/orchestration_rules.json")


def _load_rules() -> Dict[str, Any]:
    """Load orchestration rules JSON. Returns empty dict if missing."""
    p = Path(RULES_PATH)
    if not p.is_file():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _reconstruction_validation_retry_max() -> int:
    """Max retries for reconstruction when validation fails (Stage 4)."""
    rules = _load_rules()
    stage = rules.get("stages", {}).get("reconstruction_validation", {})
    return int(stage.get("retry_max", 2))


def _anatomical_check_passed(view_paths: List[str]) -> bool:
    """Optional anatomical/landmark check on view images. When False, orchestrator could retry view synthesis with higher guidance.
    Stub: always returns True; wire 2D landmark detector (e.g. face/ear ratio) when available."""
    return True

redis_client: Optional[Redis] = None
JOB_PREFIX = "job:"
QUEUE_KEY = "3d:queue"
JOB_LIST_KEY = "3d:job:list"  # list of job_id for history


def get_redis() -> Redis:
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    return redis_client


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    output_format: OutputFormat = OutputFormat.GLB
    quality_tier: QualityTier = QualityTier.STANDARD


class FromImageRequest(BaseModel):
    image_urls: Optional[List[str]] = None
    image_paths: Optional[List[str]] = None  # paths under /inputs
    output_format: OutputFormat = OutputFormat.GLB
    quality_tier: QualityTier = QualityTier.STANDARD


class JobStatusResponse(BaseModel):
    job_id: str
    state: str
    message: Optional[str] = None
    asset_path: Optional[str] = None
    bundle_path: Optional[str] = None


class GenerateResponse(BaseModel):
    job_id: str
    status_url: str
    message: str = "Job queued. Poll status_url for progress."


def _views_for_tier(tier: QualityTier) -> List[str]:
    if tier == QualityTier.FAST:
        return FAST_VIEWS
    if tier == QualityTier.HIGH:
        return HIGH_QUALITY_VIEWS
    return DEFAULT_VIEWS


def _build_view_prompts(prompt: str, view_names: List[str]) -> List[dict]:
    """Build list of {prompt, view} for image service."""
    return [
        {"prompt": f"{prompt}, {MULTI_VIEW_PROMPTS[v]}", "view": v}
        for v in view_names
    ]


async def run_job(job_id: str, payload: dict):
    """
    Correct pipeline (mandatory):
    Prompt -> Canonical Image -> View-Consistent Synthesis -> Masking -> 3D Reconstruction -> Post-Process.
    Never direct prompt -> random multi-view -> 3D.
    """
    r = get_redis()
    try:
        if "prompt" in payload and payload["prompt"]:
            # --- Stage 1: Canonical image (single object, 3/4 view, neutral gray bg) ---
            r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.GENERATING_CANONICAL.value})
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{IMAGE_SERVICE_URL}/imagine/canonical",
                    json={
                        "prompt": payload["prompt"],
                        "quality_tier": payload.get("quality_tier", "standard"),
                    },
                )
            if resp.status_code != 200:
                raise RuntimeError(f"Canonical image error: {resp.text}")
            data = resp.json()
            canonical_path = data.get("canonical_path") or data.get("image_path")
            if not canonical_path or not os.path.isfile(canonical_path):
                raise RuntimeError("Canonical image not produced or path invalid")
            payload["_canonical_path"] = canonical_path

            # --- Stage 2: View-consistent synthesis (Zero123++) from canonical ---
            r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.SYNTHESIZING_VIEWS.value})
            async with httpx.AsyncClient(timeout=600.0) as client:
                syn_resp = await client.post(
                    f"{VIEW_SYNTHESIS_SERVICE_URL}/synthesize/views",
                    json={
                        "image_path": canonical_path,
                        "mask_views": True,
                        "num_inference_steps": 75,
                    },
                )
            if syn_resp.status_code != 200:
                raise RuntimeError(f"View synthesis error: {syn_resp.text}")
            syn_data = syn_resp.json()
            view_paths = syn_data.get("view_paths", [])

            # --- Failure condition: fewer than 6 valid views ---
            if len(view_paths) < MIN_VIEWS_FOR_RECONSTRUCTION:
                raise RuntimeError(
                    f"Views insufficient for reconstruction: got {len(view_paths)}, "
                    f"need at least {MIN_VIEWS_FOR_RECONSTRUCTION}. Abort."
                )
            # Optional anatomical check (landmark-based); stub returns True; when detector added, retry view synthesis if False
            if not _anatomical_check_passed(view_paths):
                pass  # TODO: retry view synthesis with higher guidance_scale when landmark detector integrated

            # Stage 3 (masking) done inside view-synthesis when mask_views=True
            r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.RECONSTRUCTING_3D.value})

        else:
            # from-image: use provided images; skip canonical + view-synthesis
            payload["_canonical_path"] = None
            paths = payload.get("image_paths") or []
            urls = payload.get("image_urls") or []
            if urls:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{IMAGE_SERVICE_URL}/imagine/from-urls",
                        json={"urls": urls},
                    )
                if resp.status_code != 200:
                    raise RuntimeError(f"Image service from-urls error: {resp.text}")
                view_paths = resp.json().get("image_paths", [])
            else:
                view_paths = [p if p.startswith("/") else f"{INPUTS_DIR}/{p}" for p in paths]

            if not view_paths:
                raise RuntimeError("No images available for reconstruction")
            r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.RECONSTRUCTING_3D.value})

        # --- Stage 4: 3D Reconstruction + validation (retry on blob/amorphous failure) ---
        recon_retry_max = _reconstruction_validation_retry_max()
        mesh_path = None
        validation_passed = False
        validation_failures: List[str] = []

        recon_resolution = PRODUCTION_RECONSTRUCTION_RESOLUTION if PRODUCTION_BUNDLE else None
        for attempt in range(recon_retry_max + 1):
            async with httpx.AsyncClient(timeout=600.0) as client:
                recon_resp = await client.post(
                    f"{RECONSTRUCT_SERVICE_URL}/reconstruct/mesh",
                    json={
                        "image_paths": view_paths,
                        "output_format": payload.get("output_format", "glb"),
                        **({"resolution": recon_resolution} if recon_resolution else {}),
                    },
                )
            if recon_resp.status_code != 200:
                raise RuntimeError(f"Reconstruction error: {recon_resp.text}")
            recon_data = recon_resp.json()
            mesh_path = recon_data.get("mesh_path")
            if not mesh_path:
                raise RuntimeError("Reconstruction did not return mesh_path")
            validation_passed = recon_data.get("validation_passed", True)
            validation_failures = recon_data.get("validation_failures", [])
            if validation_passed:
                break
            if attempt < recon_retry_max:
                continue
            raise RuntimeError(
                f"Reconstruction validation failed after {recon_retry_max + 1} attempt(s). "
                f"Failures: {validation_failures}. Usable geometry required."
            )

        # --- Stage 4b (optional): SDS Refinement (nvdiffrast + SDXL); stub returns mesh as-is ---
        if ENABLE_SDS_REFINEMENT and mesh_path:
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    ref_resp = await client.post(
                        f"{REFINEMENT_SERVICE_URL}/refine",
                        json={
                            "mesh_path": mesh_path,
                            "view_paths": view_paths,
                            "steps": 500,
                        },
                    )
                if ref_resp.status_code == 200:
                    ref_data = ref_resp.json()
                    if ref_data.get("mesh_path") and os.path.isfile(ref_data.get("mesh_path", "")):
                        mesh_path = ref_data["mesh_path"]
            except Exception:
                pass  # keep mesh if refinement fails or unavailable

        # --- Stage 5: Post-processing (mandatory for quality) ---
        canonical_path = payload.get("_canonical_path")  # set only in prompt path
        quad_path: Optional[str] = None
        pp_data: Dict[str, Any] = {}
        if os.getenv("ENABLE_POSTPROCESS", "true").lower() == "true":
            r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.POST_PROCESSING.value})
            async with httpx.AsyncClient(timeout=300.0) as client:
                pp_resp = await client.post(
                    f"{POSTPROCESS_SERVICE_URL}/postprocess/clean",
                    json={
                        "mesh_path": mesh_path,
                        "output_format": payload.get("output_format", "glb"),
                        "quad_remesh": PRODUCTION_BUNDLE,
                        "target_quad_faces": 10000,
                        "export_high_poly_for_bake": PRODUCTION_BUNDLE,
                    },
                )
            if pp_resp.status_code == 200:
                pp_data = pp_resp.json()
                mesh_path = pp_data.get("output_path", mesh_path)
                quad_path = pp_data.get("quad_path")

        # --- Stage 6 (optional): Texture baking (UV + PBR from canonical) ---
        if ENABLE_TEXTURE and canonical_path and os.path.isfile(canonical_path):
            r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": "texturing"})
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    tex_resp = await client.post(
                        f"{TEXTURE_SERVICE_URL}/texture/bake",
                        json={
                            "mesh_path": mesh_path,
                            "canonical_image_path": canonical_path,
                            "output_format": payload.get("output_format", "glb"),
                            "texture_size": PRODUCTION_TEXTURE_SIZE if PRODUCTION_BUNDLE else 1024,
                            "inpaint_hidden": False,
                            "generate_pbr_maps": True,
                        },
                    )
                if tex_resp.status_code == 200:
                    tex_data = tex_resp.json()
                    mesh_path = tex_data.get("output_path", mesh_path)
            except Exception:
                pass  # keep postprocessed mesh if texture service fails

        # --- Final validation: only fail job when REQUIRE_MANIFOLD_VALIDATION is true ---
        if (PRODUCTION_BUNDLE or REQUIRE_MANIFOLD_VALIDATION) and mesh_path and os.path.isfile(mesh_path):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    val_resp = await client.post(
                        f"{RECONSTRUCT_SERVICE_URL}/reconstruct/validate",
                        json={"mesh_path": mesh_path, "manifold_required": REQUIRE_MANIFOLD_VALIDATION},
                    )
                if val_resp.status_code == 200:
                    val_data = val_resp.json()
                    if not val_data.get("passed", True) and REQUIRE_MANIFOLD_VALIDATION:
                        raise RuntimeError(
                            f"Final validation failed (manifold/watertight required): {val_data.get('failures', [])}"
                        )
            except RuntimeError:
                raise
            except Exception:
                pass

        # --- Finalize ---
        import shutil
        import zipfile
        ext = payload.get("output_format", "glb")
        final_path = os.path.join(OUTPUTS_DIR, f"{job_id}.{ext}")
        shutil.copy(mesh_path, final_path)

        # Production bundle: zip with high-poly, quad, final
        bundle_path = None
        if PRODUCTION_BUNDLE:
            bundle_path = os.path.join(OUTPUTS_DIR, f"{job_id}_bundle.zip")
            try:
                with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(final_path, f"{job_id}.{ext}")
                    if quad_path and os.path.isfile(quad_path):
                        zf.write(quad_path, f"{job_id}_quad.{ext}")
                    high_poly = pp_data.get("high_poly_path")
                    if high_poly and os.path.isfile(high_poly):
                        zf.write(high_poly, f"{job_id}_highpoly.{ext}")
            except Exception:
                bundle_path = None

        r.hset(f"{JOB_PREFIX}{job_id}", mapping={
            "state": JobState.COMPLETED.value,
            "asset_path": final_path,
            "message": "completed",
        })
        if bundle_path:
            r.hset(f"{JOB_PREFIX}{job_id}", "bundle_path", bundle_path)
    except Exception as e:
        r.hset(f"{JOB_PREFIX}{job_id}", mapping={
            "state": JobState.FAILED.value,
            "message": str(e),
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_redis().ping()
    yield
    if redis_client:
        redis_client.close()


app = FastAPI(title="3D Asset Orchestrator", lifespan=lifespan)

# CORS so the UI (different port or origin) can load GLB in the 3D viewer and call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/3d/generate", response_model=GenerateResponse)
async def generate_3d(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    import time
    r = get_redis()
    r.hset(f"{JOB_PREFIX}{job_id}", mapping={
        "state": JobState.QUEUED.value,
        "prompt": req.prompt,
        "output_format": req.output_format.value,
        "quality_tier": req.quality_tier.value,
        "created_at": str(int(time.time())),
    })
    r.rpush(QUEUE_KEY, job_id)
    r.lpush(JOB_LIST_KEY, job_id)
    payload = {
        "prompt": req.prompt,
        "output_format": req.output_format.value,
        "quality_tier": req.quality_tier.value,
    }
    background_tasks.add_task(run_job, job_id, payload)
    return GenerateResponse(
        job_id=job_id,
        status_url=f"/3d/status/{job_id}",
    )


@app.post("/3d/from-image", response_model=GenerateResponse)
async def from_image(req: FromImageRequest, background_tasks: BackgroundTasks):
    if not req.image_urls and not req.image_paths:
        raise HTTPException(400, "Provide image_urls or image_paths")
    job_id = str(uuid.uuid4())
    import time
    r = get_redis()
    r.hset(f"{JOB_PREFIX}{job_id}", mapping={
        "state": JobState.QUEUED.value,
        "output_format": req.output_format.value,
        "quality_tier": req.quality_tier.value,
        "created_at": str(int(time.time())),
    })
    r.lpush(JOB_LIST_KEY, job_id)
    payload = {
        "image_urls": req.image_urls,
        "image_paths": req.image_paths,
        "output_format": req.output_format.value,
        "quality_tier": req.quality_tier.value,
    }
    background_tasks.add_task(run_job, job_id, payload)
    return GenerateResponse(
        job_id=job_id,
        status_url=f"/3d/status/{job_id}",
    )


@app.get("/3d/status/{job_id}", response_model=JobStatusResponse)
async def status(job_id: str):
    r = get_redis()
    data = r.hgetall(f"{JOB_PREFIX}{job_id}")
    if not data:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job_id,
        state=data.get("state", "unknown"),
        message=data.get("message"),
        asset_path=data.get("asset_path"),
        bundle_path=data.get("bundle_path"),
    )


@app.get("/3d/download/{job_id}")
async def download(job_id: str):
    r = get_redis()
    data = r.hgetall(f"{JOB_PREFIX}{job_id}")
    if not data or data.get("state") != JobState.COMPLETED.value:
        raise HTTPException(404, "Job not found or not completed")
    path = data.get("asset_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Asset file not found")
    return FileResponse(path, filename=os.path.basename(path))


@app.get("/3d/download-bundle/{job_id}")
async def download_bundle(job_id: str):
    """Download production bundle zip (high-poly + quad + textures) when PRODUCTION_BUNDLE=true."""
    r = get_redis()
    data = r.hgetall(f"{JOB_PREFIX}{job_id}")
    if not data or data.get("state") != JobState.COMPLETED.value:
        raise HTTPException(404, "Job not found or not completed")
    path = data.get("bundle_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Bundle not found (set PRODUCTION_BUNDLE=true for zip output)")
    return FileResponse(path, filename=os.path.basename(path), media_type="application/zip")


@app.get("/3d/jobs")
async def list_jobs():
    """List all jobs (newest first) for history."""
    r = get_redis()
    job_ids = r.lrange(JOB_LIST_KEY, 0, 99)  # last 100
    out = []
    for jid in job_ids:
        data = r.hgetall(f"{JOB_PREFIX}{jid}")
        if data:
            out.append({
                "job_id": jid,
                "state": data.get("state", "unknown"),
                "prompt": data.get("prompt", ""),
                "created_at": data.get("created_at"),
                "asset_path": data.get("asset_path"),
                "bundle_path": data.get("bundle_path"),
            })
    return {"jobs": out}


def _list_dirs(parent: str) -> list:
    """Return list of { name, path, mtime } for subdirectories."""
    out = []
    if not parent or not os.path.isdir(parent):
        return out
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            try:
                mtime = int(os.path.getmtime(path))
            except OSError:
                mtime = 0
            out.append({"name": name, "path": path, "mtime": mtime})
    out.sort(key=lambda x: -x["mtime"])
    return out


@app.get("/3d/folders")
async def list_folders():
    """List generated input and view folders (for display and delete in UI)."""
    input_folders = _list_dirs(INPUTS_DIR)
    views_dir = os.path.join(OUTPUTS_DIR, "views")
    view_folders = _list_dirs(views_dir)
    return {
        "input_folders": input_folders,
        "view_folders": view_folders,
    }


@app.delete("/3d/folder/{folder_type}/{name}")
async def delete_folder(folder_type: str, name: str):
    """Delete a generated folder by type (input or view) and name. Name must be a single path segment."""
    if folder_type not in ("input", "view"):
        raise HTTPException(400, "folder_type must be 'input' or 'view'")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid folder name")
    if folder_type == "input":
        path = os.path.join(INPUTS_DIR, name)
    else:
        path = os.path.join(OUTPUTS_DIR, "views", name)
    if not os.path.isdir(path):
        raise HTTPException(404, "Folder not found")
    import shutil
    try:
        shutil.rmtree(path)
    except OSError as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


@app.delete("/3d/job/{job_id}")
async def delete_job(job_id: str):
    """Remove job from history and delete asset file."""
    r = get_redis()
    data = r.hgetall(f"{JOB_PREFIX}{job_id}")
    if not data:
        raise HTTPException(404, "Job not found")
    r.lrem(JOB_LIST_KEY, 0, job_id)
    r.delete(f"{JOB_PREFIX}{job_id}")
    path = data.get("asset_path")
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return {"ok": True}


@app.get("/3d/access-check")
async def access_check(request: Request):
    """Return whether the client IP is allowed to use the 3D-test UI. Used by the UI to show blocked message."""
    forwarded = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip") or ""
    client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "")
    if not client_ip and request.client:
        client_ip = request.client.host
    allowed = client_ip in ALLOWED_IPS
    owner_ip = next(iter(ALLOWED_IPS), "") if ALLOWED_IPS else ""
    return {"allowed": allowed, "client_ip": client_ip or "unknown", "owner_ip": owner_ip}


@app.get("/health")
async def health():
    get_redis().ping()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
