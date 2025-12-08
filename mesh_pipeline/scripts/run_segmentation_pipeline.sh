#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# Load key paths from config using the Python helper (avoids bash YAML parsing)
eval "$(
python3 - <<'PY'
import sys
from pathlib import Path
from shlex import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from utils.config_utils import load_paths  # noqa: E402

p = load_paths()
print(f'SESSION_DIR={quote(str(p.session_dir))}')
print(f'DEPTH_DIR={quote(str(p.depth_maps_dir))}')
PY
)"

echo "=== 3D Segmentation Pipeline ==="
echo "Session directory: $SESSION_DIR"
echo "Depth maps path:  $DEPTH_DIR"
echo

ensure_depth_maps() {
  if [[ ! -d "$DEPTH_DIR" ]] || ! find "$DEPTH_DIR" -mindepth 1 -print -quit >/dev/null; then
    echo "Depth maps not found at $DEPTH_DIR."
    read -rp "Generate depth maps now via generate_depth.py? [y/N]: " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      echo "--- Running depth generation (mesh_generation/generate_depth.py) ---"
      python3 "$ROOT_DIR/src/mesh_generation/generate_depth.py"
      echo "--- Depth generation complete ---"
    else
      echo "Depth maps are required. Aborting pipeline."
      exit 1
    fi
  fi
}

step() {
  local label="$1"; shift
  echo
  echo "=== $label ==="
  "$@"
}

ensure_depth_maps

step "Segmentation (run_segmentation.py)" \
  python3 "$ROOT_DIR/src/3D_segmentation/run_segmentation.py"

step "Fuse labels to mesh (fuse_labels.py)" \
  python3 "$ROOT_DIR/src/3D_segmentation/fuse_labels.py"

step "Clean mesh labels (clean_mesh.py)" \
  python3 "$ROOT_DIR/src/3D_segmentation/clean_mesh.py"

step "Transfer labels to trimesh ordering (transfer_labels_to_trimesh.py)" \
  python3 "$ROOT_DIR/src/3D_segmentation/transfer_labels_to_trimesh.py"

step "Export Habitat artifacts (export_hm3d.py)" \
  python3 "$ROOT_DIR/src/3D_segmentation/export_hm3d.py"

echo
echo "Pipeline complete."
