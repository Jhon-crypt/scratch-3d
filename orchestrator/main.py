"""
Orchestrator Service — Control plane for 3D asset pipeline.
Does not run ML models. Coordinates jobs across image and 3D services.
"""
import os
import uuid
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from redis import Redis
from typing import Optional, List

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
RECONSTRUCT_SERVICE_URL = os.getenv("RECONSTRUCT_SERVICE_URL", "http://reconstruction-service:8000")
POSTPROCESS_SERVICE_URL = os.getenv("POSTPROCESS_SERVICE_URL", "http://postprocess-service:8000")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")
INPUTS_DIR = os.getenv("INPUTS_DIR", "/inputs")

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
    """Execute pipeline: images -> reconstruct -> optional postprocess -> save."""
    r = get_redis()
    try:
        # 1. Images: either generate or use provided
        r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.GENERATING_IMAGES.value})
        image_paths = []

        if "prompt" in payload and payload["prompt"]:
            view_names = _views_for_tier(QualityTier(payload.get("quality_tier", "standard")))
            prompts = _build_view_prompts(payload["prompt"], view_names)
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{IMAGE_SERVICE_URL}/imagine/generate",
                    json={"prompts": [p["prompt"] for p in prompts], "quality_tier": payload.get("quality_tier", "standard")},
                )
            if resp.status_code != 200:
                raise RuntimeError(f"Image service error: {resp.text}")
            data = resp.json()
            image_paths = data.get("image_paths", [])
        else:
            # from-image: use provided paths (under /inputs)
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
                image_paths = resp.json().get("image_paths", [])
            else:
                image_paths = [p if p.startswith("/") else f"{INPUTS_DIR}/{p}" for p in paths]

        if not image_paths:
            raise RuntimeError("No images available for reconstruction")

        # 2. 3D Reconstruction
        r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.RECONSTRUCTING_3D.value})
        async with httpx.AsyncClient(timeout=600.0) as client:
            recon_resp = await client.post(
                f"{RECONSTRUCT_SERVICE_URL}/reconstruct/mesh",
                json={"image_paths": image_paths, "output_format": payload.get("output_format", "glb")},
            )
        if recon_resp.status_code != 200:
            raise RuntimeError(f"Reconstruction error: {recon_resp.text}")
        recon_data = recon_resp.json()
        mesh_path = recon_data.get("mesh_path")
        if not mesh_path:
            raise RuntimeError("Reconstruction did not return mesh_path")

        # 3. Optional post-processing
        if os.getenv("ENABLE_POSTPROCESS", "true").lower() == "true":
            r.hset(f"{JOB_PREFIX}{job_id}", mapping={"state": JobState.POST_PROCESSING.value})
            async with httpx.AsyncClient(timeout=300.0) as client:
                pp_resp = await client.post(
                    f"{POSTPROCESS_SERVICE_URL}/postprocess/clean",
                    json={"mesh_path": mesh_path, "output_format": payload.get("output_format", "glb")},
                )
            if pp_resp.status_code == 200:
                pp_data = pp_resp.json()
                mesh_path = pp_data.get("output_path", mesh_path)

        # 4. Copy/finalize to job output path
        import shutil
        ext = payload.get("output_format", "glb")
        final_path = os.path.join(OUTPUTS_DIR, f"{job_id}.{ext}")
        shutil.copy(mesh_path, final_path)
        r.hset(f"{JOB_PREFIX}{job_id}", mapping={
            "state": JobState.COMPLETED.value,
            "asset_path": final_path,
            "message": "completed",
        })
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
            })
    return {"jobs": out}


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


@app.get("/health")
async def health():
    get_redis().ping()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
