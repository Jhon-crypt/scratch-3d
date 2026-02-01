"""
Image Generation Service — FLUX-based multi-view image generation.
Generates images from prompts; supports multi-view templated prompts.
All images on neutral backgrounds for 3D reconstruction.
"""
import os
import uuid
import base64
import tempfile
from pathlib import Path
from typing import List, Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

INPUTS_DIR = os.getenv("INPUTS_DIR", "/inputs")
# FLUX backend URL when running as separate GPU service; or in-process
FLUX_BACKEND_URL = os.getenv("FLUX_BACKEND_URL", "")  # e.g. http://flux-backend:8000
USE_INTERNAL_STUB = os.getenv("USE_IMAGE_STUB", "true").lower() == "true"

app = FastAPI(title="Image Generation Service (FLUX multi-view)")


class GenerateRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=16)
    quality_tier: str = "standard"  # standard -> 512, high -> 768 or 1024


class CanonicalRequest(BaseModel):
    """Stage 1: Generate exactly ONE canonical reference image (object identity)."""
    prompt: str = Field(..., min_length=1)
    quality_tier: str = "standard"


class FromUrlsRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1)


def _resolution_for_tier(tier: str) -> int:
    return 768 if tier == "high" else 512


def _ensure_inputs_dir():
    Path(INPUTS_DIR).mkdir(parents=True, exist_ok=True)


async def _generate_with_flux(prompts: List[str], quality_tier: str) -> List[str]:
    """Call FLUX backend or stub. Returns list of absolute image paths."""
    _ensure_inputs_dir()
    job_id = str(uuid.uuid4())
    out_dir = Path(INPUTS_DIR) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if FLUX_BACKEND_URL and not USE_INTERNAL_STUB:
        size = _resolution_for_tier(quality_tier)
        flux_failed = False
        async with httpx.AsyncClient(timeout=300.0) as client:
            for i, prompt in enumerate(prompts):
                resp = await client.post(
                    f"{FLUX_BACKEND_URL}/generate",
                    json={
                        "prompt": prompt,
                        "width": size,
                        "height": size,
                        "num_inference_steps": 28,
                        "guidance_scale": 3.5,
                    },
                )
                if resp.status_code != 200:
                    flux_failed = True
                    break
                data = resp.json()
                if "image_b64" in data:
                    raw = base64.b64decode(data["image_b64"])
                    path = out_dir / f"view_{i:02d}.png"
                    path.write_bytes(raw)
                    paths.append(str(path))
                elif "image_path" in data:
                    paths.append(data["image_path"])
        if flux_failed:
            import logging
            logging.warning("FLUX returned error; using stub canonical image so pipeline can continue.")
    if not paths:
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        try:
            from PIL import Image
            for i in range(len(prompts)):
                path = out_dir / f"view_{i:02d}.png"
                img = Image.new("RGB", (_resolution_for_tier(quality_tier), _resolution_for_tier(quality_tier)), (128, 128, 128))
                img.save(path)
                paths.append(str(path))
        except ImportError:
            for i in range(len(prompts)):
                path = out_dir / f"view_{i:02d}.png"
                path.write_bytes(minimal_png)
                paths.append(str(path))

    return paths


def _canonical_prompt(user_prompt: str) -> str:
    """Single object, centered, 3/4 view, neutral gray background."""
    return (
        f"studio photograph of {user_prompt}, 3/4 front view, neutral gray background, "
        "soft lighting, highly detailed, single object centered, no clutter"
    )


@app.post("/imagine/canonical")
async def generate_canonical(req: CanonicalRequest):
    """Stage 1: Generate exactly ONE canonical reference image. Defines object identity."""
    try:
        canonical_prompt = _canonical_prompt(req.prompt)
        paths = await _generate_with_flux([canonical_prompt], req.quality_tier)
        if not paths:
            raise RuntimeError("No image generated")
        return {"image_path": paths[0], "canonical_path": paths[0]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/imagine/generate")
async def generate(req: GenerateRequest):
    """Generate multiple images from prompts (multi-view). Neutral background implied in prompts."""
    try:
        paths = await _generate_with_flux(req.prompts, req.quality_tier)
        return {"image_paths": paths, "count": len(paths)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/imagine/from-urls")
async def from_urls(req: FromUrlsRequest):
    """Download images from URLs and save to INPUTS_DIR. Returns local paths."""
    _ensure_inputs_dir()
    job_id = str(uuid.uuid4())
    out_dir = Path(INPUTS_DIR) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, url in enumerate(req.urls):
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(400, f"Failed to fetch {url}")
            ext = "png" if "png" in resp.headers.get("content-type", "") else "jpg"
            path = out_dir / f"input_{i:02d}.{ext}"
            path.write_bytes(resp.content)
            paths.append(str(path))
    return {"image_paths": paths, "count": len(paths)}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
