"""
Transfer semantic labels from an Open3D-loaded mesh to a trimesh-loaded mesh.

Why: Open3D and trimesh can reorder vertices differently. If you fuse labels
with Open3D but export GLB with trimesh, you need labels in the trimesh order.

Workflow:
1) Load the mesh + labels as seen by Open3D.
2) Load the same mesh file via trimesh.
3) For each trimesh vertex, find its nearest Open3D vertex (KD-tree) and copy
   that label.
4) Save the re-ordered labels for downstream export_hm3d.py.

All paths are resolved from config/paths.yml via config_utils.load_paths().
"""

import open3d as o3d
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.config_utils import load_paths

# ===== CONFIG =====
paths = load_paths()
# Mesh + labels as seen by Open3D
SRC_MESH_PATH   = paths.mesh_path                 # same mesh used in fuse_labels
SRC_LABELS_PATH = paths.fused_ids_clean  # or paths.fused_ids, same length

# Mesh as seen by trimesh (the one export_hm3d.py will use)
TGT_MESH_PATH   = paths.mesh_path                 # same file, but loaded via trimesh
OUT_LABELS_PATH = paths.trimesh_labels     # new labels matching trimesh vertex count
# ==================


def main() -> None:
    """
    Transfer labels from Open3D vertex ordering to trimesh vertex ordering.
    """
    # ---- load source (Open3D) ----
    print(f"Loading source mesh (Open3D): {SRC_MESH_PATH}")
    src_mesh = o3d.io.read_triangle_mesh(str(SRC_MESH_PATH))
    src_verts = np.asarray(src_mesh.vertices)
    print("  source vertices:", len(src_verts))

    print(f"Loading source labels: {SRC_LABELS_PATH}")
    src_labels = np.load(str(SRC_LABELS_PATH))
    print("  source labels:  ", len(src_labels))

    if len(src_labels) != len(src_verts):
        raise RuntimeError(
            f"Source mesh and labels length mismatch: "
            f"{len(src_verts)} vs {len(src_labels)}"
        )

    # ---- load target (trimesh) ----
    print(f"Loading target mesh (trimesh): {TGT_MESH_PATH}")
    tgt_mesh = trimesh.load(str(TGT_MESH_PATH))
    tgt_verts = np.asarray(tgt_mesh.vertices)
    print("  target vertices:", len(tgt_verts))

    # ---- build KD-tree on source vertices ----
    print("Building KD-tree on source vertices...")
    tree = cKDTree(src_verts)

    # ---- for each target vertex, take 1-NN label from source ----
    num_tgt = len(tgt_verts)
    tgt_labels = np.empty(num_tgt, dtype=src_labels.dtype)

    chunk_size = 200000  # to avoid big memory spikes
    print("Transferring labels with 1-NN...")
    for start in tqdm(range(0, num_tgt, chunk_size)):
        end = min(start + chunk_size, num_tgt)
        batch = tgt_verts[start:end]

        dists, idx = tree.query(batch, k=1, workers=-1)
        tgt_labels[start:end] = src_labels[idx]

    print(f"Saving transferred labels to {OUT_LABELS_PATH}")
    np.save(str(OUT_LABELS_PATH), tgt_labels)

    # Quick sanity check
    print("Sanity check:")
    print("  First 10 src labels:", src_labels[:10])
    print("  First 10 tgt labels:", tgt_labels[:10])


if __name__ == "__main__":
    main()
