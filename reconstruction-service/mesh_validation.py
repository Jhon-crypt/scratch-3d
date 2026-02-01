"""
Stage 4 reconstruction validation: blob and amorphous mesh detection.
Machine-enforceable heuristics from orchestration_rules.json failure conditions.
Production-grade: set PRODUCTION_VALIDATION=true for relaxed blob/melted thresholds
and optional face_count_min (e.g. 50k for InstantMesh/SF3D output).
"""
import math
import logging
import os
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Default thresholds (aligned with shared/orchestration_rules.json)
# Relaxed so round/smooth objects (vases, bowls, buildings) pass; only reject obvious blobs.
DEFAULT_SPHERICITY_MAX = 0.98  # above = blob-like; 0.98 allows slightly round shapes (e.g. bungalow)
DEFAULT_SHARP_EDGE_RATIO_MIN = 0.0005  # below = melted; allow smooth meshes (TripoSR is often soft)
DEFAULT_ELONGATION_MIN_RATIO = 1.2  # elongated (vase, house, car) → skip blob check; 1.2 = slight stretch
DEFAULT_VERTEX_COUNT_MIN = 100
# Production-grade: higher geometric complexity (InstantMesh/SF3D 50k–100k faces)
PRODUCTION_SPHERICITY_MAX = 0.99  # very lenient for high-poly
PRODUCTION_SHARP_EDGE_RATIO_MIN = 0.0003
PRODUCTION_FACE_COUNT_MIN = 0  # set 50000 when using InstantMesh/SF3D
# TripoSR often outputs non-watertight; production may require manifold.
DEFAULT_MANIFOLD_REQUIRED = False


def _sphericity(volume: float, area: float) -> float:
    """Sphericity = surface area of sphere with same volume / actual area. 1.0 = sphere."""
    if area <= 0 or volume <= 0:
        return 0.0
    # Sphere with V has area A_s = (36*pi*V^2)^(1/3)
    sphere_area = (36.0 * math.pi * volume * volume) ** (1.0 / 3.0)
    return min(1.0, sphere_area / area)


def _sharp_edge_ratio(mesh) -> float:
    """Fraction of edges where dihedral angle exceeds threshold. Man-made = higher."""
    try:
        # face_adjacency: (E, 2) array of face indices sharing an edge; trimesh may return (faces, edges) tuple
        raw = getattr(mesh, "face_adjacency", None)
        if raw is None:
            return 0.0
        adj = raw[0] if isinstance(raw, (list, tuple)) and len(raw) >= 1 else raw
        if getattr(adj, "shape", None) is None or len(adj) == 0:
            return 0.0
        normals = mesh.face_normals
        threshold_rad = math.radians(25)
        sharp = 0
        n_adj = len(adj)
        for i in range(n_adj):
            a, b = int(adj[i, 0]), int(adj[i, 1])
            n_a = normals[a]
            n_b = normals[b]
            dot = max(-1.0, min(1.0, float(n_a.dot(n_b))))
            angle = math.acos(dot)
            if angle >= threshold_rad:
                sharp += 1
        return sharp / n_adj if n_adj > 0 else 0.0
    except Exception as e:
        logger.warning("sharp_edge_ratio failed: %s", e)
        return 0.0


def _is_elongated(mesh, min_ratio: float) -> bool:
    """True if shape is elongated (vase, bottle, car) — not a sphere blob."""
    try:
        extents = mesh.extents
        if extents.min() <= 0:
            return False
        ratio = float(extents.max() / extents.min())
        return ratio >= min_ratio
    except Exception:
        return False


def _float_env(name: str) -> Optional[float]:
    """Parse optional float from env (e.g. SPHERICITY_MAX=0.99)."""
    val = os.getenv(name)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _production_thresholds():
    """Use production-grade thresholds when PRODUCTION_VALIDATION=true."""
    if os.getenv("PRODUCTION_VALIDATION", "").lower() in ("1", "true", "yes"):
        face_min = int(os.getenv("PRODUCTION_FACE_COUNT_MIN", str(PRODUCTION_FACE_COUNT_MIN)))
        return PRODUCTION_SPHERICITY_MAX, PRODUCTION_SHARP_EDGE_RATIO_MIN, face_min
    return DEFAULT_SPHERICITY_MAX, DEFAULT_SHARP_EDGE_RATIO_MIN, 0


def validate_mesh(
    mesh_path: str,
    sphericity_max: float = None,
    sharp_edge_ratio_min: float = None,
    elongation_min_ratio: float = None,
    vertex_count_min: int = DEFAULT_VERTEX_COUNT_MIN,
    face_count_min: int = 0,
    manifold_required: bool = DEFAULT_MANIFOLD_REQUIRED,
) -> Tuple[bool, List[str]]:
    """
    Run Stage 4 reconstruction validation. Returns (passed, list of failure reasons).
    Failure conditions: blob_like_geometry, melted_amorphous_surfaces, missing_major_components.
    Env overrides: SPHERICITY_MAX, SHARP_EDGE_RATIO_MIN, ELONGATION_MIN_RATIO (float).
    Production: set PRODUCTION_VALIDATION=true or pass face_count_min (e.g. 50000) for high-poly.
    """
    failures: List[str] = []
    if sphericity_max is None or sharp_edge_ratio_min is None:
        prod_sph, prod_sharp, prod_faces = _production_thresholds()
        sphericity_max = sphericity_max or _float_env("SPHERICITY_MAX") or prod_sph
        sharp_edge_ratio_min = sharp_edge_ratio_min or _float_env("SHARP_EDGE_RATIO_MIN") or prod_sharp
        if face_count_min == 0 and prod_faces > 0:
            face_count_min = prod_faces
    if elongation_min_ratio is None:
        elongation_min_ratio = _float_env("ELONGATION_MIN_RATIO") or DEFAULT_ELONGATION_MIN_RATIO
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh is None or not hasattr(mesh, "vertices"):
            failures.append("missing_major_components")
            return False, failures
    except Exception as e:
        logger.warning("Mesh load failed for validation: %s", e)
        failures.append("missing_major_components")
        return False, failures

    n_vertices = len(mesh.vertices)
    if n_vertices < vertex_count_min:
        failures.append("missing_major_components")
        return False, failures

    # Production-grade: minimum face count (e.g. 50k for InstantMesh/SF3D)
    if face_count_min > 0:
        n_faces = len(mesh.faces) if hasattr(mesh, "faces") else 0
        if n_faces < face_count_min:
            failures.append("missing_major_components")

    # Optional: fail on non-watertight or multiple components (TripoSR often has small holes)
    if manifold_required:
        if not mesh.is_watertight:
            failures.append("closed_manifold")
        try:
            components = mesh.split()
            if len(components) > 1:
                failures.append("missing_major_components")
        except Exception:
            pass

    # Blob-like: high sphericity (shape too round/ball-like). Skip if elongated (vase, bottle).
    elongated = _is_elongated(mesh, elongation_min_ratio)
    if not elongated:
        try:
            if mesh.is_watertight and mesh.volume > 0 and mesh.area > 0:
                sph = _sphericity(float(mesh.volume), float(mesh.area))
                if sph >= sphericity_max:
                    failures.append("blob_like_geometry")
            else:
                try:
                    extents = mesh.extents
                    if extents.min() > 0:
                        vol_bb = float(extents.prod())
                        area = float(mesh.area)
                        if area > 0:
                            sph = _sphericity(vol_bb * 0.5, area)  # rough proxy
                            if sph >= sphericity_max:
                                failures.append("blob_like_geometry")
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Sphericity check failed: %s", e)

    # Melted/amorphous: essentially zero sharp edges (truly amorphous). Vases are smooth but have some faceting.
    try:
        sharp_ratio = _sharp_edge_ratio(mesh)
        if sharp_ratio < sharp_edge_ratio_min:
            failures.append("melted_amorphous_surfaces")
    except Exception as e:
        logger.warning("Sharp edge check failed: %s", e)

    passed = len(failures) == 0
    return passed, failures
