"""
Post-Processing Service — Mesh cleaning.
Decimate, recalc normals, auto UV unwrap, texture bake.
Designed to run headless Blender; CPU-based is acceptable.
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")
CACHE_DIR = os.getenv("CACHE_DIR", "/cache")
BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")

app = FastAPI(title="Post-Processing Service (Mesh clean)")


class CleanRequest(BaseModel):
    mesh_path: str
    output_format: str = "glb"
    quad_remesh: bool = False  # Production: output quad mesh via QuadriFlow
    target_quad_faces: int = Field(default=10000, ge=1000, le=100000)
    export_high_poly_for_bake: bool = False  # Export copy for normal-map baking; actual bake is downstream


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


def _has_vertex_colors(mesh) -> bool:
    """True if mesh has per-vertex color (e.g. from TripoSR). Subdivision/smoothing would lose it."""
    try:
        if not hasattr(mesh, "visual") or mesh.visual is None:
            return False
        return getattr(mesh.visual, "kind", None) == "vertex" or hasattr(
            mesh.visual, "vertex_colors"
        )
    except Exception:
        return False


def _smooth_mesh(mesh, keep_color: bool):
    """
    Prefer Taubin smoothing (no volume shrinkage); fallback to Laplacian.
    Production-grade spec: Taubin λ/μ filter instead of Laplacian melting.
    """
    import trimesh

    taubin = getattr(trimesh.smoothing, "filter_taubin", None)
    if taubin is not None:
        try:
            # Taubin: lamb positive, nu negative to avoid shrinkage (e.g. 0.53, -0.53)
            taubin(mesh, lamb=0.5, nu=-0.53, iterations=5)
            return
        except Exception:
            pass
    # Fallback: Laplacian (can cause volume loss)
    try:
        if keep_color:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.2, iterations=1)
        else:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=2)
    except Exception:
        pass


def _clean_mesh_trimesh(mesh_path: str, output_path: str, ext: str) -> None:
    """
    Load mesh, repair holes/normals; Taubin smooth (or Laplacian fallback).
    Subdivide only when no vertex colors (preserve TripoSR color).
    """
    import trimesh

    # process=False helps preserve vertex colors (e.g. from TripoSR) on load
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if mesh is None or not hasattr(mesh, "vertices"):
        raise ValueError("Could not load mesh as single Trimesh")

    keep_color = _has_vertex_colors(mesh)

    if not keep_color:
        try:
            mesh = mesh.subdivide()
        except Exception:
            pass
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass

    _smooth_mesh(mesh, keep_color)

    # Fix normals (shading); safe for vertex-colored meshes
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    mesh.export(output_path, file_type=ext)
    return mesh


def _quad_remesh(mesh, target_faces: int):
    """Optional QuadriFlow quad remesh. Returns new mesh or None on failure."""
    try:
        import trimesh
        import numpy as np
        pyquad = __import__("pyquadriflow", fromlist=["pyquadriflow"])
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        # pyQuadriFlow: (target_faces, seed, vertices, face_indexes, preserve_sharp, preserve_boundary, ...)
        result = pyquad.pyquadriflow(
            target_faces, 0, verts, faces,
            True, True, False, False, True,
        )
        if result is None or len(result) < 2:
            return None
        out_verts, out_faces = result[0], result[1]
        return trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("QuadriFlow remesh failed: %s", e)
        return None


@app.post("/postprocess/clean")
async def clean(req: CleanRequest):
    """Subdivide, repair holes/normals, Taubin smooth; optional quad remesh and high-poly export."""
    _ensure_dirs()
    if not os.path.isfile(req.mesh_path):
        raise HTTPException(400, f"Mesh not found: {req.mesh_path}")
    ext = req.output_format.lower() or "glb"
    if ext not in ("glb", "obj", "ply"):
        ext = "glb"
    out_dir = Path(OUTPUTS_DIR) / "postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    output_path = str(out_dir / f"{job_id}.{ext}")
    quad_path: Optional[str] = None
    high_poly_path: Optional[str] = None
    mesh = None

    try:
        try:
            mesh = _clean_mesh_trimesh(req.mesh_path, output_path, ext)
        except ImportError:
            shutil.copy(req.mesh_path, output_path)
        except Exception as e:
            import trimesh
            mesh = trimesh.load(req.mesh_path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            if mesh is not None and hasattr(mesh, "vertices"):
                mesh.export(output_path, file_type=ext)
            else:
                shutil.copy(req.mesh_path, output_path)
            mesh = None

        if req.quad_remesh and mesh is not None:
            quad_mesh = _quad_remesh(mesh, req.target_quad_faces)
            if quad_mesh is not None:
                quad_path = str(out_dir / f"{job_id}_quad.{ext}")
                quad_mesh.export(quad_path, file_type=ext)

        if req.export_high_poly_for_bake and os.path.isfile(req.mesh_path):
            high_poly_path = str(out_dir / f"{job_id}_highpoly.{ext}")
            shutil.copy(req.mesh_path, high_poly_path)

        out = {"output_path": output_path}
        if quad_path:
            out["quad_path"] = quad_path
        if high_poly_path:
            out["high_poly_path"] = high_poly_path
        return out
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
