"""
3D Reconstruction Service — Converts images to 3D mesh.
Uses TripoSR for real single-image reconstruction.
Load model once per GPU; accept multiple images (use first); produce vertex-colored mesh.
"""
import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

INPUTS_DIR = os.getenv("INPUTS_DIR", "/inputs")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")
CACHE_DIR = os.getenv("CACHE_DIR", "/cache")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Reconstruction Service (TripoSR)")

# Lazy-loaded TripoSR model and rembg session
_triposr_model = None
_rembg_session = None


def _device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_model():
    global _triposr_model
    if _triposr_model is not None:
        return _triposr_model
    try:
        from tsr.system import TSR
    except Exception as e:
        raise RuntimeError(
            f"TripoSR import failed (check PYTHONPATH=/app/triposr and rebuild): {e}"
        ) from e
    device = _device()
    logger.info("Loading TripoSR model on %s ...", device)
    try:
        _triposr_model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        _triposr_model.renderer.set_chunk_size(8192)
        _triposr_model.to(device)
        logger.info("TripoSR model loaded.")
        return _triposr_model
    except Exception as e:
        raise RuntimeError(f"TripoSR model load failed: {e}") from e


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    try:
        import rembg
        _rembg_session = rembg.new_session()
        return _rembg_session
    except Exception as e:
        logger.warning("rembg session failed: %s", e)
        return None


def _preprocess_image(image_path: str, foreground_ratio: float = 0.85):
    """Load image, remove background, resize foreground, composite on gray. Returns PIL Image."""
    from PIL import Image
    import numpy as np
    from tsr.utils import remove_background, resize_foreground

    pil = Image.open(image_path).convert("RGB")
    session = _get_rembg_session()
    if session is not None:
        pil = remove_background(pil, session)
        if pil.mode == "RGBA":
            pil = resize_foreground(pil, foreground_ratio)
    arr = np.array(pil).astype(np.float32) / 255.0
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _run_triposr(image_paths: List[str], output_path: str, fmt: str) -> str:
    """
    Run TripoSR on the first image. Write mesh to output_path.
    Requires TripoSR to be loaded; no silent cube fallback.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _get_model()
    if model is None:
        raise RuntimeError(
            "TripoSR failed to load. Rebuild reconstruction-service with TripoSR (see Dockerfile) "
            "and ensure the container has GPU (NVIDIA_VISIBLE_DEVICES)."
        )

    import torch
    device = _device()
    # TripoSR is single-image; use first view
    image_path = image_paths[0]
    image = _preprocess_image(image_path)

    with torch.no_grad():
        scene_codes = model([image], device=device)
    meshes = model.extract_mesh(scene_codes, True, resolution=256)
    mesh = meshes[0]

    # Export: TripoSR mesh has .export(path)
    mesh.export(str(out_path))
    return str(out_path)


def _ensure_dirs():
    for d in (INPUTS_DIR, OUTPUTS_DIR, CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


class MeshRequest(BaseModel):
    image_paths: List[str] = Field(..., min_length=1, max_length=16)
    output_format: str = "glb"


@app.post("/reconstruct/mesh")
async def reconstruct_mesh(req: MeshRequest):
    """Produce 3D mesh from list of image paths (TripoSR uses first image)."""
    _ensure_dirs()
    for p in req.image_paths:
        if not os.path.isfile(p):
            raise HTTPException(400, f"Image not found: {p}")
    job_id = str(uuid.uuid4())
    out_dir = Path(OUTPUTS_DIR) / "reconstruct"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = req.output_format.lower()
    if ext not in ("glb", "obj", "fbx"):
        ext = "glb"
    output_path = str(out_dir / f"{job_id}.{ext}")
    try:
        final_path = _run_triposr(req.image_paths, output_path, ext)
        return {"mesh_path": final_path, "job_id": job_id}
    except Exception as e:
        logger.exception("Reconstruction failed")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
