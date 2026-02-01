"""
SDS Refinement Service — Optional Stage: sculpt mesh to match high-fidelity detail.
Production spec: Score Distillation Sampling (SDS) loop with nvdiffrast + SDXL,
500–1000 steps to carve micro-details. This stub returns the mesh as-is until
nvdiffrast and SDXL are integrated.
"""
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")

app = FastAPI(title="SDS Refinement Service (stub)")


class RefineRequest(BaseModel):
    mesh_path: str = Field(..., min_length=1)
    view_paths: list = Field(default_factory=list)  # optional multi-view images for SDS
    steps: int = Field(default=500, ge=1, le=2000)  # SDS iterations when implemented


@app.post("/refine")
async def refine(req: RefineRequest):
    """
    Refine mesh with SDS (nvdiffrast + SDXL). Stub: copy mesh to output and return path.
    When implemented: render mesh from multiple angles, backpropagate SDXL errors, update geometry.
    """
    if not os.path.isfile(req.mesh_path):
        raise HTTPException(400, "Mesh file not found")
    out_dir = Path(OUTPUTS_DIR) / "refinement"
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    ext = Path(req.mesh_path).suffix or ".glb"
    out_path = str(out_dir / f"{job_id}{ext}")
    import shutil
    shutil.copy(req.mesh_path, out_path)
    return {"mesh_path": out_path, "refined": False, "message": "stub: mesh copied; SDS not yet implemented"}


@app.get("/health")
async def health():
    return {"status": "ok"}
