"""
InstantMesh runner: call TencentARC/InstantMesh when repo is available.
Usage: python run_instantmesh.py <image_path> <output_glb_path> [resolution]
Exit 0 on success, non-zero on failure.
"""
import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Usage: run_instantmesh.py <image_path> <output_glb_path> [resolution]", file=sys.stderr)
        sys.exit(1)
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(2)
    try:
        # Try InstantMesh from TencentARC (clone to /app/instantmesh or set INSTANTMESH_PATH)
        instantmesh_path = os.getenv("INSTANTMESH_PATH", "/app/instantmesh")
        if os.path.isdir(instantmesh_path):
            sys.path.insert(0, instantmesh_path)
        from src.inference import run_inference  # noqa: E402
        run_inference(image_path, output_path, resolution=resolution)
        if os.path.isfile(output_path):
            sys.exit(0)
    except ImportError as e:
        print(f"InstantMesh not available: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"InstantMesh inference failed: {e}", file=sys.stderr)
        sys.exit(4)
    sys.exit(5)


if __name__ == "__main__":
    main()
