"""
View-Synthesis Service — Stage 2: View-consistent image synthesis from canonical image.
Uses Zero123++ (image-conditioned diffusion). NEVER text-only multi-view.
All views represent the SAME object. Outputs are masked (RemBG) before return.
"""
# Patch torch so diffusers/deps don't crash on CUDA-only or older PyTorch builds
import torch as _torch_module

if not hasattr(_torch_module, "xpu"):
    class _FakeXPU:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def empty_cache():
            pass
        @staticmethod
        def device_count():
            return 0
        @staticmethod
        def current_device():
            return 0
        @staticmethod
        def set_device(device):
            pass
        @staticmethod
        def manual_seed(seed):
            pass

        @staticmethod
        def __getattr__(name):
            def _noop(*args, **kwargs):
                return None
            return _noop
    _torch_module.xpu = _FakeXPU()

# PyTorch 2.1 lacks torch.distributed.device_mesh (added in 2.2+); diffusers may require it
if hasattr(_torch_module, "distributed") and not hasattr(_torch_module.distributed, "device_mesh"):
    class _FakeDeviceMesh:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            def _noop(*a, **k):
                return None
            return _noop
    _FakeDeviceMesh.DeviceMesh = _FakeDeviceMesh  # diffusers may access device_mesh.DeviceMesh
    _torch_module.distributed.device_mesh = _FakeDeviceMesh

import os
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

INPUTS_DIR = os.getenv("INPUTS_DIR", "/inputs")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")
# Engine: zero123 (default) | sv3d (SV3D_u orbital; requires chenguolin/sv3d-diffusers)
VIEW_SYNTHESIS_ENGINE = os.getenv("VIEW_SYNTHESIS_ENGINE", "zero123").lower()
# Masking: rembg (default) | sam (Segment Anything for razor-sharp silhouettes; set USE_SAM_MASKING=true when SAM integrated)
USE_SAM_MASKING = os.getenv("USE_SAM_MASKING", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="View-Synthesis Service (Zero123++ / SV3D)")

_pipeline = None
_rembg_session = None


def _device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        # Custom pipeline from cloned Zero123++ repo
        custom_path = os.getenv("ZERO123_CUSTOM_PIPELINE", "/app/zero123plus/diffusers-support")
        model_id = os.getenv("ZERO123_MODEL", "sudo-ai/zero123plus-v1.1")

        logger.info("Loading Zero123++ pipeline from %s ...", model_id)
        _pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            custom_pipeline=custom_path,
            torch_dtype=torch.float16,
        )
        _pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            _pipeline.scheduler.config, timestep_spacing="trailing"
        )
        _pipeline.to(_device())
        logger.info("Zero123++ pipeline loaded.")
        return _pipeline
    except Exception as e:
        logger.exception("Zero123++ load failed: %s", e)
        raise


def _get_rembg():
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    try:
        import rembg
        _rembg_session = rembg.new_session()
        return _rembg_session
    except Exception as e:
        logger.warning("rembg failed: %s", e)
        return None


class SynthesizeRequest(BaseModel):
    """Input: path to canonical image (Stage 1 output)."""
    image_path: str = Field(..., min_length=1)
    mask_views: bool = True  # Stage 3: remove background from all views
    num_inference_steps: int = 75


def _synthesize_sv3d(image_path: str, out_dir: Path, mask_views: bool, num_steps: int):
    """SV3D: orbital video (21 frames) → sample 6 frames as views. Falls back to None if pipeline unavailable."""
    try:
        from PIL import Image
        import torch
        from diffusers import StableVideo3DDiffusionPipeline

        device = _device()
        pipe = StableVideo3DDiffusionPipeline.from_pretrained(
            "chenguolin/sv3d-diffusers",
            torch_dtype=torch.float16,
            variant="sv3d_p",
        )
        pipe.to(device)
        cond = Image.open(image_path).convert("RGB")
        w, h = cond.size
        if w != h:
            side = max(w, h)
            new = Image.new("RGB", (side, side), (128, 128, 128))
            new.paste(cond, ((side - w) // 2, (side - h) // 2))
            cond = new
        if max(cond.size) < 256:
            cond = cond.resize((256, 256), Image.Resampling.LANCZOS)

        with torch.no_grad():
            out = pipe(
                cond,
                num_frames=21,
                num_inference_steps=num_steps,
                fps=7,
                motion_bucket_id=127,
                noise_aug_strength=0.02,
            )
        frames = getattr(out, "frames", None) or []
        if not frames or len(frames) < 6:
            logger.warning("SV3D returned %d frames; need 6+", len(frames))
            return None
        # Sample 6 views: frames 0, 3, 6, 9, 12, 15
        indices = [0, 3, 6, 9, 12, 15][: min(6, len(frames))]
        view_paths = []
        for i, idx in enumerate(indices):
            frame = frames[idx] if hasattr(frames[idx], "save") else Image.fromarray(frames[idx])
            raw_path = str(out_dir / f"view_{i:02d}.png")
            frame.save(raw_path)
            if mask_views:
                session = _get_rembg()
                if session is not None:
                    import rembg
                    out_img = rembg.remove(frame, session=session)
                    masked_path = str(out_dir / f"masked_{i:02d}.png")
                    out_img.convert("RGBA").save(masked_path)
                    view_paths.append(masked_path)
                else:
                    view_paths.append(raw_path)
            else:
                view_paths.append(raw_path)
        return view_paths
    except Exception as e:
        logger.warning("SV3D pipeline failed (%s); fallback to Zero123++", e)
        return None


@app.post("/synthesize/views")
async def synthesize_views(req: SynthesizeRequest):
    """
    Stage 2: Generate view-consistent multi-view images from canonical image.
    Engine: Zero123++ (default) or SV3D when VIEW_SYNTHESIS_ENGINE=sv3d.
    Returns list of view image paths (6+). Optionally masked.
    """
    if not os.path.isfile(req.image_path):
        raise HTTPException(400, f"Image not found: {req.image_path}")

    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    out_dir = Path(OUTPUTS_DIR) / "views" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if VIEW_SYNTHESIS_ENGINE == "sv3d":
        view_paths = _synthesize_sv3d(req.image_path, out_dir, req.mask_views, req.num_inference_steps)
        if view_paths is not None:
            return {"view_paths": view_paths, "job_id": job_id, "count": len(view_paths), "engine": "sv3d"}

    try:
        from PIL import Image
        import torch

        pipeline = _get_pipeline()
        device = _device()
        cond = Image.open(req.image_path).convert("RGB")
        # Zero123++ expects square; resize if needed
        w, h = cond.size
        if w != h:
            side = max(w, h)
            new = Image.new("RGB", (side, side), (128, 128, 128))
            new.paste(cond, ((side - w) // 2, (side - h) // 2))
            cond = new
        if max(cond.size) < 320:
            cond = cond.resize((320, 320), Image.Resampling.LANCZOS)

        with torch.no_grad():
            result = pipeline(cond, num_inference_steps=req.num_inference_steps)

        # result.images: list of PIL (one per view) or single grid image to split
        images = getattr(result, "images", None) or []
        if not images:
            images = [getattr(result, "image", None)] if hasattr(result, "image") else []
        if not isinstance(images, list):
            images = [images]
        # If pipeline returns one grid image (e.g. 2x3 or 3x2), split into tiles
        if len(images) == 1:
            img = images[0]
            w, h = img.size
            # Zero123++ often outputs 6 views in 2 rows x 3 cols (e.g. 1536x1024)
            if w >= 3 * 256 and h >= 2 * 256:
                ncols, nrows = 3, 2
                tw, th = w // ncols, h // nrows
                images = [
                    img.crop((j * tw, i * th, (j + 1) * tw, (i + 1) * th))
                    for i in range(nrows) for j in range(ncols)
                ]
            elif h >= 3 * 256 and w >= 2 * 256:
                ncols, nrows = 2, 3
                tw, th = w // ncols, h // nrows
                images = [
                    img.crop((j * tw, i * th, (j + 1) * tw, (i + 1) * th))
                    for i in range(nrows) for j in range(ncols)
                ]

        view_paths = []
        for i, img in enumerate(images):
            raw_path = str(out_dir / f"view_{i:02d}.png")
            img.save(raw_path)
            if req.mask_views:
                masked_path = str(out_dir / f"masked_{i:02d}.png")
                session = _get_rembg()
                if session is not None:
                    import rembg
                    out_img = rembg.remove(img, session=session)
                    out_img.convert("RGBA").save(masked_path)
                    view_paths.append(masked_path)
                else:
                    view_paths.append(raw_path)
            else:
                view_paths.append(raw_path)

        if len(view_paths) < 6:
            raise RuntimeError(
                f"Zero123++ returned {len(view_paths)} views; need at least 6. "
                "Abort: views insufficient for reconstruction."
            )

        return {
            "view_paths": view_paths,
            "job_id": job_id,
            "count": len(view_paths),
            "engine": VIEW_SYNTHESIS_ENGINE,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("View synthesis failed: %s", e)
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
