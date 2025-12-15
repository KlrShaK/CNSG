#!/usr/bin/env python3
"""
Helper utility to apply global alignment transformation to an existing mesh.

This script is useful for fixing meshes that were generated without the
global alignment transformation applied. It loads a PLY file, applies the
alignment from paths.yml, and saves it back (replacing the original file).

Usage:
    python apply_global_alignment.py --input path/to/mesh.ply
    python apply_global_alignment.py --input path/to/mesh.ply --alignment-file custom_alignment.txt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

# Add repo root to path to import config utils
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_utils import load_paths


def quaternion_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix.

    Args:
        q_wxyz: Quaternion in (w, x, y, z) format.

    Returns:
        3x3 rotation matrix.
    """
    q = np.array(q_wxyz, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = q / n

    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def pose_to_T(q_wxyz: np.ndarray, t_xyz: np.ndarray) -> np.ndarray:
    """
    Build 4x4 transform from quaternion and translation.

    Args:
        q_wxyz: Quaternion (qw, qx, qy, qz).
        t_xyz: Translation (tx, ty, tz).

    Returns:
        4x4 transformation matrix.
    """
    R = quaternion_to_rotation_matrix(q_wxyz)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.array(t_xyz, dtype=float)
    return T


def read_alignment_pose_line(
    alignment_path: Path,
    label: str = "pose_graph_optimized",
) -> np.ndarray:
    """
    Parse alignment_global.txt of the form:
      label, reference_id, qw, qx, qy, qz, tx, ty, tz, ...

    Args:
        alignment_path: Path to alignment_global.txt file.
        label: Label to search for in the file.

    Returns:
        4x4 transformation matrix T_s2w.
    """
    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    with open(alignment_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            if parts[0] != label:
                continue

            # parts: [label, reference_id, qw, qx, qy, qz, tx, ty, tz, ...]
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])
            return pose_to_T(np.array([qw, qx, qy, qz], float), np.array([tx, ty, tz], float))

    raise ValueError(
        f"Could not find label '{label}' in alignment file: {alignment_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply global alignment transformation to a mesh file (replaces original)"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input PLY mesh file to transform",
    )

    parser.add_argument(
        "--alignment-file",
        type=Path,
        default=None,
        help="Optional explicit alignment_global.txt path. If omitted, uses config paths.yml",
    )

    parser.add_argument(
        "--alignment-label",
        type=str,
        default="pose_graph_optimized",
        help="Which label to read from alignment_global.txt (default: pose_graph_optimized)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not args.input.suffix.lower() == ".ply":
        print(f"Warning: Input file is not a .ply file: {args.input}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)

    # Get alignment file path from config or command line
    if args.alignment_file is not None:
        alignment_path = args.alignment_file
    else:
        # Load from paths.yml using config_utils
        paths = load_paths()
        alignment_path = paths.global_alignment_file

    print("=" * 60)
    print("  Apply Global Alignment to Mesh")
    print("=" * 60)
    print(f"Input mesh:       {args.input}")
    print(f"Alignment file:   {alignment_path}")
    print(f"Alignment label:  {args.alignment_label}")
    print("=" * 60)
    print()

    # Load mesh
    print("[1/4] Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(str(args.input))
    if mesh.is_empty():
        # Try loading as point cloud
        pcd = o3d.io.read_point_cloud(str(args.input))
        if pcd.is_empty():
            print(f"Error: Could not load geometry from: {args.input}")
            sys.exit(1)
        print(f"      Loaded point cloud with {len(pcd.points):,} points")
        is_pointcloud = True
        geom = pcd
    else:
        print(f"      Loaded mesh with {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
        is_pointcloud = False
        geom = mesh

    # Read alignment transformation
    print("[2/4] Reading alignment transformation...")
    T_s2w = read_alignment_pose_line(alignment_path, label=args.alignment_label)
    print(f"      Transformation matrix:")
    print(f"      {T_s2w[0]}")
    print(f"      {T_s2w[1]}")
    print(f"      {T_s2w[2]}")
    print(f"      {T_s2w[3]}")

    # Apply transformation
    print("[3/4] Applying transformation...")
    geom.transform(T_s2w)
    print("      Transformation applied successfully")

    # Save back to original file
    print("[4/4] Saving transformed mesh...")
    print(f"      Overwriting: {args.input}")

    if is_pointcloud:
        success = o3d.io.write_point_cloud(str(args.input), geom)
    else:
        success = o3d.io.write_triangle_mesh(str(args.input), geom)

    if not success:
        print(f"Error: Failed to save mesh to {args.input}")
        sys.exit(1)

    print("      Saved successfully")
    print()
    print("=" * 60)
    print("  Alignment applied and mesh saved!")
    print("=" * 60)


if __name__ == "__main__":
    main()
