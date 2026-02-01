"""
FLUX Service — Real FLUX image generation (one model load per GPU).
Uses FLUX.1-schnell by default (fast, 1–4 steps); optional FLUX.1-dev for higher quality.
"""
import os
import uuid
import base64
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Model: schnell = fast/low VRAM; dev = higher quality, more steps
FLUX_MODEL = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
INPUTS_DIR = os.getenv("INPUTS_DIR", "/inputs")
USE_SCHNELL = "schnell" in FLUX_MODEL.lower()

app = FastAPI(title="FLUX Image Generation Service")

pipe = None


def get_pipe():
    global pipe
    if pipe is not None:
        return pipe
    import torch
    from diffusers import FluxPipeline

    dtype = torch.bfloat16
    token = os.getenv("HF_TOKEN") or None  # required for gated FLUX.1-schnell
    pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=dtype, token=token)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    width: int = Field(512, ge=256, le=1536)
    height: int = Field(512, ge=256, le=1536)
    num_inference_steps: Optional[int] = None  # schnell: 1–4, dev: ~50
    guidance_scale: Optional[float] = None    # schnell: 0, dev: 3.5
    seed: Optional[int] = None


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate one image from prompt. Returns image_path and image_b64."""
    import torch

    try:
        steps = req.num_inference_steps
        guidance = req.guidance_scale
        if USE_SCHNELL:
            steps = steps or 4
            guidance = guidance if guidance is not None else 0.0
            if guidance != 0.0:
                guidance = 0.0
        else:
            steps = steps or 28
            guidance = guidance if guidance is not None else 3.5

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(req.seed)

        pipe = get_pipe()
        kwargs = {
            "prompt": req.prompt,
            "height": req.height,
            "width": req.width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if USE_SCHNELL:
            kwargs["max_sequence_length"] = 256

        out = pipe(**kwargs)
        image = out.images[0]

        Path(INPUTS_DIR).mkdir(parents=True, exist_ok=True)
        job_id = str(uuid.uuid4())
        out_dir = Path(INPUTS_DIR) / f"flux_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "image.png"
        image.save(str(path))

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"image_path": str(path), "image_b64": image_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": FLUX_MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
