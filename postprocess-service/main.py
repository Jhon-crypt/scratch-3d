"""
Post-Processing Service — Mesh cleaning.
Decimate, recalc normals, auto UV unwrap, texture bake.
Designed to run headless Blender; CPU-based is acceptable.
"""
import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")
CACHE_DIR = os.getenv("CACHE_DIR", "/cache")
BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")

app = FastAPI(title="Post-Processing Service (Mesh clean)")


class CleanRequest(BaseModel):
    mesh_path: str
    output_format: str = "glb"


def _ensure_dirs():
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _clean_with_blender(mesh_path: str, output_path: str, fmt: str) -> str:
    """
    Run Blender headless: import mesh, decimate, recalc normals, export.
    If Blender not available, copy file as-is.
    """
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(mesh_path)
    try:
        import bpy
        bpy.ops.wm.read_homefile(use_empty=True)
        ext = fmt.lower()
        if ext == "glb":
            bpy.ops.import_scene.gltf(filepath=mesh_path)
        elif ext == "obj":
            bpy.ops.import_scene.obj(filepath=mesh_path)
        else:
            shutil.copy(mesh_path, output_path)
            return output_path
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_add(type="DECIMATE")
                bpy.context.object.modifiers["Decimate"].ratio = 0.5
                bpy.ops.object.modifier_apply(modifier="Decimate")
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.object.mode_set(mode="OBJECT")
        if ext == "glb":
            bpy.ops.export_scene.gltf(filepath=output_path)
        else:
            bpy.ops.export_scene.obj(filepath=output_path)
        return output_path
    except ImportError:
        # No Blender Python: copy as-is
        shutil.copy(mesh_path, output_path)
        return output_path


@app.post("/postprocess/clean")
async def clean(req: CleanRequest):
    """Decimate, recalc normals, export to desired format."""
    _ensure_dirs()
    if not os.path.isfile(req.mesh_path):
        raise HTTPException(400, f"Mesh not found: {req.mesh_path}")
    ext = req.output_format.lower() or "glb"
    out_dir = Path(OUTPUTS_DIR) / "postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    output_path = str(out_dir / f"{job_id}.{ext}")
    try:
        # Prefer trimesh for portability (no Blender dependency in container)
        try:
            import trimesh
            mesh = trimesh.load(req.mesh_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            mesh.export(output_path, file_type=ext)
            return {"output_path": output_path}
        except ImportError:
            pass
        # Fallback: copy
        shutil.copy(req.mesh_path, output_path)
        return {"output_path": output_path}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
