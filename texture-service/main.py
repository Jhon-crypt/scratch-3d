"""
Texture Service — UV unwrap, texture projection from canonical image, optional inpainting.
Production-grade: replace vertex colors with 1024²/2048² PBR albedo texture.
"""
import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/outputs")
CACHE_DIR = os.getenv("CACHE_DIR", "/cache")
TEXTURE_SIZE = int(os.getenv("TEXTURE_SIZE", "1024"))
UV_COVERAGE_MIN = float(os.getenv("UV_COVERAGE_MIN", "0.5"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Texture Service (UV + PBR)")


class TextureRequest(BaseModel):
    mesh_path: str = Field(..., min_length=1)
    canonical_image_path: str = Field(..., min_length=1)
    output_format: str = "glb"
    texture_size: int = Field(default=1024, ge=256, le=4096)
    inpaint_hidden: bool = False  # fill occluded UV texels via neighbor diffusion


def _inpaint_texture_holes(tex: "np.ndarray", default_color=(128, 128, 128), iterations: int = 4) -> "np.ndarray":
    """Fill texels at default color with neighbor average (diffusion inpainting)."""
    import numpy as np
    tex = np.asarray(tex, dtype=np.float64)
    default = np.array(default_color, dtype=np.float64)
    mask = np.all(np.abs(tex - default) < 2, axis=-1)
    if not np.any(mask):
        return tex.astype(np.uint8)
    h, w = tex.shape[:2]
    for _ in range(iterations):
        tex_new = tex.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if mask[y, x]:
                    n = tex[y - 1 : y + 2, x - 1 : x + 2].reshape(-1, 3)
                    non_default = np.any(np.abs(n - default) >= 2, axis=1)
                    if np.any(non_default):
                        tex_new[y, x] = np.mean(n[non_default], axis=0)
                        mask[y, x] = False
        tex = tex_new
    return tex.astype(np.uint8)


def _uv_coverage(faces, uvs) -> float:
    """Approximate UV island coverage: sum of triangle areas in UV space (0..1)."""
    try:
        import numpy as np
        uvs = np.asarray(uvs).reshape(-1, 2)
        total = 0.0
        for tri in faces:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            if i >= len(uvs) or j >= len(uvs) or k >= len(uvs):
                continue
            ua, va = uvs[i][0], uvs[i][1]
            ub, vb = uvs[j][0], uvs[j][1]
            uc, vc = uvs[k][0], uvs[k][1]
            area = 0.5 * abs((ua * (vb - vc) + ub * (vc - va) + uc * (va - vb)))
            total += area
        return min(1.0, total)
    except Exception:
        return 0.0


def _unwrap_xatlas(vertices, faces):
    """Return (new_vertices, new_faces, uvs) from xatlas parametrize."""
    try:
        import xatlas
        import numpy as np
        vmapping, indices, uvs = xatlas.parametrize(np.asarray(vertices), np.asarray(faces))
        new_verts = np.asarray(vertices)[np.asarray(vmapping)]
        return new_verts, np.asarray(indices), np.asarray(uvs)
    except Exception as e:
        logger.warning("xatlas parametrize failed: %s", e)
        return None


def _project_canonical_to_texture(vertices_3d, faces_3d, uvs, canonical_path: str, texture_size: int):
    """
    Create texture image by projecting canonical image onto mesh via simple front view.
    Uses vertices_3d, faces_3d, and uvs (one UV per vertex from xatlas).
    """
    import numpy as np
    from PIL import Image

    img = Image.open(canonical_path).convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]

    verts = np.asarray(vertices_3d)
    faces = np.asarray(faces_3d)
    uv_flat = np.asarray(uvs).reshape(-1, 2)
    n_verts = len(verts)
    if n_verts == 0 or len(faces) == 0:
        tex = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 128
        return tex

    # UV per vertex (xatlas order matches vertex order)
    if len(uv_flat) >= n_verts:
        uv_per_vertex = uv_flat[:n_verts]
    else:
        uv_per_vertex = np.zeros((n_verts, 2))

    # Build texture: default gray
    tex = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 128

    # Simple projection: front view (orthographic) for canonical
    verts_2d = verts[:, :2]
    mn, mx = verts_2d.min(axis=0), verts_2d.max(axis=0)
    span = (mx - mn)
    span[span < 1e-6] = 1.0
    verts_2d_norm = (verts_2d - mn) / span
    verts_img_x = (verts_2d_norm[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
    verts_img_y = ((1 - verts_2d_norm[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)

    # Rasterize each triangle in UV space and sample from image
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if i >= n_verts or j >= n_verts or k >= n_verts:
            continue
        ua, va = uv_per_vertex[i]
        ub, vb = uv_per_vertex[j]
        uc, vc = uv_per_vertex[k]
        tx_a = int(ua * (texture_size - 1)) % texture_size
        ty_a = int((1 - va) * (texture_size - 1)) % texture_size
        tx_b = int(ub * (texture_size - 1)) % texture_size
        ty_b = int((1 - vb) * (texture_size - 1)) % texture_size
        tx_c = int(uc * (texture_size - 1)) % texture_size
        ty_c = int((1 - vc) * (texture_size - 1)) % texture_size
        xa, ya = verts_img_x[i], verts_img_y[i]
        xb, yb = verts_img_x[j], verts_img_y[j]
        xc, yc = verts_img_x[k], verts_img_y[k]
        ca = img_arr[ya, xa]
        cb = img_arr[yb, xb]
        cc = img_arr[yc, xc]
        tmin_x = max(0, min(tx_a, tx_b, tx_c))
        tmax_x = min(texture_size - 1, max(tx_a, tx_b, tx_c) + 1)
        tmin_y = max(0, min(ty_a, ty_b, ty_c))
        tmax_y = min(texture_size - 1, max(ty_a, ty_b, ty_c) + 1)
        if tmax_x <= tmin_x or tmax_y <= tmin_y:
            continue
        avg = (np.array(ca) + np.array(cb) + np.array(cc)) // 3
        tex[tmin_y:tmax_y, tmin_x:tmax_x] = avg

    return tex


def _apply_texture_to_mesh(mesh, texture_array, uvs, texture_size: int):
    """Attach texture to mesh and return mesh with TextureVisuals."""
    import trimesh
    from PIL import Image
    from trimesh.visual import TextureVisuals
    from trimesh.visual import material

    img = Image.fromarray(texture_array)
    mat = material.SimpleMaterial(image=img)
    visual = TextureVisuals(uv=uvs, image=img, material=mat)
    mesh_tex = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        visual=visual,
        process=False,
    )
    return mesh_tex


def run_texture_pipeline(
    mesh_path: str,
    canonical_path: str,
    output_path: str,
    texture_size: int = 1024,
    inpaint: bool = False,
) -> Tuple[str, float]:
    """
    UV unwrap, project canonical to texture, optional inpainting, export GLB.
    Returns (output_path, uv_coverage).
    """
    import trimesh
    import numpy as np

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if mesh is None or not hasattr(mesh, "vertices"):
        raise ValueError("Could not load mesh")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    # UV unwrap with xatlas
    result = _unwrap_xatlas(verts, faces)
    if result is None:
        # Fallback: no UV, export mesh as-is (vertex color preserved)
        mesh.export(output_path, file_type="glb")
        return output_path, 0.0

    new_verts, new_faces, uvs = result
    uvs = np.asarray(uvs, dtype=np.float64)

    # UV coverage
    coverage = _uv_coverage(new_faces, uvs)
    if coverage < UV_COVERAGE_MIN:
        logger.warning("UV coverage %.2f < min %.2f", coverage, UV_COVERAGE_MIN)

    # Rebuild mesh with new verts/faces (xatlas may have duplicated vertices)
    mesh_uv = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    # Assign UVs: xatlas returns uvs per vertex in new order
    mesh_uv.visual = trimesh.visual.TextureVisuals(uv=uvs)

    # Project canonical image to texture
    texture_array = _project_canonical_to_texture(
        mesh_uv.vertices, mesh_uv.faces, uvs, canonical_path, texture_size
    )

    # Inpaint occluded UV: fill default-color texels with neighbor diffusion (no external API)
    if inpaint:
        texture_array = _inpaint_texture_holes(texture_array, default_color=(128, 128, 128))

    # Attach texture and export
    mesh_final = _apply_texture_to_mesh(mesh_uv, texture_array, uvs, texture_size)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mesh_final.export(output_path, file_type="glb")
    return output_path, coverage


@app.post("/texture/bake")
async def bake_texture(req: TextureRequest):
    """
    UV unwrap mesh, project canonical image to texture, export textured GLB.
    Returns output_path and uv_coverage (validation: >0.5).
    """
    if not os.path.isfile(req.mesh_path):
        raise HTTPException(400, f"Mesh not found: {req.mesh_path}")
    if not os.path.isfile(req.canonical_image_path):
        raise HTTPException(400, f"Canonical image not found: {req.canonical_image_path}")

    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    ext = req.output_format.lower() or "glb"
    out_path = str(Path(OUTPUTS_DIR) / "textured" / f"{job_id}.{ext}")

    try:
        final_path, uv_coverage = run_texture_pipeline(
            req.mesh_path,
            req.canonical_image_path,
            out_path,
            texture_size=req.texture_size,
            inpaint=req.inpaint_hidden,
        )
        return {
            "output_path": final_path,
            "uv_coverage": round(uv_coverage, 4),
            "validation_passed": uv_coverage >= UV_COVERAGE_MIN,
        }
    except Exception as e:
        logger.exception("Texture bake failed: %s", e)
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
