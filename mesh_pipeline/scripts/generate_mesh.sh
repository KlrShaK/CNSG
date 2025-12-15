#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# Load paths from config via Python to avoid YAML parsing in bash
eval "$(
python3 - <<'PY'
import sys
from pathlib import Path
from shlex import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from utils.config_utils import load_paths  # noqa: E402

p = load_paths()
print(f'POINTCLOUD={quote(str(p.raw_pointcloud))}')
print(f'POISSON_OUT={quote(str(p.poisson_mesh))}')
PY
)"

VOXEL_OUT="${POISSON_OUT%.ply}.voxelized.ply"

echo "Mesh generation options:"
echo "  1) Poisson mesh only"
echo "  2) Poisson mesh + voxelized mesh"
echo "  3) Full pipeline (Poisson + voxelize + depth maps)"
read -rp "Select option [1-3, default=3]: " choice
choice=${choice:-3}

run_poisson() {
  echo "Running Poisson meshing..."
  python3 "$ROOT_DIR/src/mesh_generation/run_poisson_meshing.py" \
    --input "$POINTCLOUD" \
    --output "$POISSON_OUT"
}

run_voxelize() {
  echo "Running voxelization..."
  python3 "$ROOT_DIR/src/mesh_generation/simplify_mesh.py" \
    --input "$POISSON_OUT" \
    --output "$VOXEL_OUT" \
    --save_voxel_mesh
}

run_depth() {
  echo "Generating depth maps..."
  python3 "$ROOT_DIR/src/mesh_generation/generate_depth.py"
}

case "$choice" in
  1)
    run_poisson
    ;;
  2)
    run_poisson
    run_voxelize
    ;;
  3)
    run_poisson
    run_voxelize
    run_depth
    ;;
  *)
    echo "Invalid selection. Please choose 1, 2, or 3." >&2
    exit 1
    ;;
esac

echo "Done."
