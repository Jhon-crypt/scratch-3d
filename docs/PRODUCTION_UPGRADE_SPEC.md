# Technical Specification: Production-Grade 3D Pipeline Upgrade

This document maps the upgrade from **experimental/low-fidelity** (TripoSR, vertex colors) to **production-grade/artistic** (high-geometry, UV-textured, quad topology).

---

## Stage Comparison

| Stage | Current (Experimental) | Production Grade (Target) |
|-------|------------------------|----------------------------|
| **Input** | FLUX (1 image) | FLUX (depth-conditioned) |
| **Views** | Zero123++ (6 tiles) | Unique3D or SV3D_P (high-res consistent views) |
| **Recon** | TripoSR (fast/melted) | InstantMesh or SF3D (sharp/detailed) |
| **Texture** | Vertex colors (blurry) | UV unwrapping + 4K PBR albedo map |
| **Topology** | Raw triangles | Quad-remeshed (retopology) |

---

## 1. View Synthesis Engine Upgrade

**Current:** Zero123++ (6-view grid).  
**Target:** Unique3D or SV3D_P.

- **Implementation:** Replace `view-synthesis-service` logic.
- **Objective:** 4–8 high-resolution (1024×1024), multi-view consistent images.
- **Key feature:** Symmetry constraints for organic/humanoid subjects to prevent anatomical warping.

**Tasks:**
- [x] Engine switch in `view-synthesis-service`: `VIEW_SYNTHESIS_ENGINE=zero123|sv3d` (sv3d stub; uses Zero123++).
- [ ] Integrate SV3D pipeline (chenguolin/sv3d-diffusers) for orbital views; sample 6–8 frames.
- [ ] Output 4–8 views at 1024²; optional symmetry constraint API.

---

## 2. Geometry Reconstruction Engine Upgrade

**Current:** TripoSR (transformer-based, low-poly, blobby).  
**Target:** InstantMesh or SF3D (LGM/CRM-based).

- **Implementation:** Replace or add backend in `reconstruction-service`.
- **Framework:** LGM (Large Gaussian Model) or CRM-based reconstruction.
- **Resolution:** Mesh target **50k–100k faces** before post-processing.
- **Validation:** Update `mesh_validation.py` for higher geometric complexity (relax sphericity, raise sharp-edge threshold).

**Tasks:**
- [x] Engine switch in `reconstruction-service`: `RECONSTRUCTION_ENGINE=triposr|instantmesh` (instantmesh stub; uses TripoSR).
- [x] Face-count target 50k–100k via `PRODUCTION_FACE_COUNT_MIN`; production thresholds in `mesh_validation.py`.
- [ ] Integrate InstantMesh (TencentARC/InstantMesh) or SF3D for high-poly output.

---

## 3. Texture Mapping (The "Realism" Layer)

**Current:** Vertex coloring (color per vertex, inherently blurry).  
**Target:** UV unwrapping + PBR texture projection.

- **Step A (Unwrapping):** xatlas or Blender Python for automatic UV unwrap.
- **Step B (Projection):** Texture projection from high-res FLUX canonical image onto UV layout.
- **Step C (Inpainting):** Latent-consistency model (e.g. Stable Diffusion) to inpaint hidden UV regions from multi-view images.
- **Output:** `.glb` with 1024×1024 or 2048×2048 albedo texture map.

**Tasks:**
- [x] New `texture-service`: UV unwrap (xatlas), texture projection from canonical image to UV, export GLB with 1024² texture.
- [x] Validation: UV island coverage >50% (`UV_COVERAGE_MIN`); returns `validation_passed` and `uv_coverage`.
- [x] Inpainting step for occluded UV regions (neighbor-diffusion in texture-service; optional SD/LC later).

---

## 4. Retopology and Post-Processing

**Current:** Laplacian smoothing (causes melting/volume loss).  
**Target:** Quad remeshing + Taubin smoothing.

- **Retopology:** Integrate QuadriFlow in `postprocess-service` (triangle → clean quads).
- **Smoothing:** Replace Laplacian with **Taubin smoothing** (λ/μ filter; no volume shrinkage).
- **Normal map:** Bake high-poly detail to normal map for low-poly real-time use.

**Tasks:**
- [x] Replace Laplacian with Taubin in postprocess (trimesh `filter_taubin`).
- [x] Optional QuadriFlow integration in postprocess (`quad_remesh=true`, `target_quad_faces`); requires `pyQuadriFlow`.
- [x] Optional high-poly export for normal-map baking (`export_high_poly_for_bake=true`).
- [ ] Normal map baking: downstream step; high-poly export done.

---

## 5. Orchestration & Validation Logic

**Target:** Refined failure detection in `shared/orchestration_rules.json`.

- **Geometry:** Manifold check (no holes, non-manifold edges) for production output.
- **Texture:** UV island coverage >50% of texture square.
- **Retry:** If face/ear ratio anatomically impossible (2D landmark detection on views), retry view synthesis with higher guidance scale.

**Tasks:**
- [x] Production validation in `orchestration_rules.json`: `production_grade.reconstruction_validation`, `texture_validation.uv_coverage_min`, `postprocessing.smoothing: taubin`.
- [x] UV coverage validation in texture-service; manifold in `mesh_validation` when `manifold_required=true` / `PRODUCTION_VALIDATION`.
- [x] Optional anatomical check stub; landmark-based retry can be wired when detector available.

---

## Final Output Structure (Production)

Deliver a **zipped package** containing:

1. **High-poly GLB** — For rendering and digital art.
2. **Retopologized quad mesh** — For animators (Blender/Maya/Unreal).
3. **4K textures** — Albedo, Normal, Roughness maps.

**Tasks:**
- [x] Orchestrator: when `PRODUCTION_BUNDLE=true`, create `{job_id}_bundle.zip` with final GLB, quad GLB, high-poly GLB.
- [x] API: `GET /3d/download-bundle/{job_id}`; status includes `bundle_path` when present.
- [ ] UI: "Download bundle" link when `bundle_path` is set.

---

## Implementation Order (Recommended)

1. **Foundation:** Orchestration rules update (production validation criteria).  
2. **Postprocess:** Taubin smoothing (done), then QuadriFlow stub, normal map stub.  
3. **Validation:** Mesh validation relaxed for high-face-count / production geometry.  
4. **View synthesis:** Unique3D/SV3D_P evaluation and integration.  
5. **Reconstruction:** InstantMesh/SF3D evaluation and integration.  
6. **Texture:** UV unwrap + projection + inpainting service.  
7. **Bundle:** Zip output (high-poly + quad + 4K textures).

