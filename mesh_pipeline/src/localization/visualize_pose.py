#!/usr/bin/env python3
"""
Visualize camera poses on a 3D mesh/pointcloud with Open3D.

- Loads a mesh/pointcloud (PLY/PCD/etc.)
- Loads poses from poses.txt:
    timestamp, sensor_id, qw,qx,qy,qz, tx,ty,tz, ...
- Optionally interprets poses as COLMAP-style world->camera (w2c) and converts to camera->world (c2w)
- Optionally applies alignment_global.txt to the MESH (NOT to the poses)

alignment_global.txt expected format:
# label, reference_id, qw, qx, qy, qz, tx, ty, tz, [info]+
pose_graph_optimized, __absolute__, qw, qx, qy, qz, tx, ty, tz

IMPORTANT:
- This script DOES NOT apply alignment to the query pose.
- Use --apply-mesh-alignment to bring the mesh into the same aligned frame as the poses.

Usage:
  python visualize_pose.py --poses poses.txt --apply-mesh-alignment
  python visualize_pose.py --poses poses.txt --apply-mesh-alignment --pose-format w2c
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is required. Install with: pip install open3d")
    sys.exit(1)

# Add repo root to path (adjust if needed)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_utils import load_paths
from src.utils.apply_global_alignment import (
    quaternion_to_rotation_matrix,
    read_alignment_pose_line,
)


# ----------------------------
# IO helpers
# ----------------------------
def read_poses_file(poses_file: Path):
    """
    Read camera poses from poses.txt file.

    Expected CSV:
      timestamp, sensor_id, qw, qx, qy, qz, tx, ty, tz, *optional stuff...

    Returns list of dicts with:
      timestamp, sensor_id, q (qw,qx,qy,qz), t (tx,ty,tz)
    """
    poses = []
    with open(poses_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            timestamp = int(parts[0])
            sensor_id = parts[1]
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])

            poses.append(
                dict(
                    timestamp=timestamp,
                    sensor_id=sensor_id,
                    q=np.array([qw, qx, qy, qz], dtype=float),  # (w,x,y,z)
                    t=np.array([tx, ty, tz], dtype=float),
                )
            )
    return poses


# ----------------------------
# Pose conversions
# ----------------------------
def convert_pose_w2c_to_c2w(q_w2c: np.ndarray, t_w2c: np.ndarray):
    """
    Given world->camera:
      x_cam = R_w2c x_world + t_w2c

    Convert to camera->world:
      x_world = R_c2w x_cam + C_world

    Using:
      R_w2c = quat(q_w2c)
      C_world = -R_w2c^T t_w2c
      R_c2w = R_w2c^T
    """
    R_w2c = quaternion_to_rotation_matrix(q_w2c)
    C_world = -R_w2c.T @ t_w2c
    R_c2w = R_w2c.T
    return R_c2w, C_world


# ----------------------------
# Geometry helpers
# ----------------------------
def load_geometry(mesh_path: Path):
    """Load pointcloud if possible, else triangle mesh."""
    pcd = o3d.io.read_point_cloud(str(mesh_path))
    if not pcd.is_empty():
        return pcd

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load geometry from: {mesh_path}")
    mesh.compute_vertex_normals()
    return mesh


def create_camera_frustum(position, R_c2w, scale=0.5, color=(1.0, 0.0, 0.0)):
    """
    Create a camera frustum visualization given c2w rotation and world position.

    Frustum defined in camera coords looking +Z (OpenCV-ish).
    """
    frustum_points = np.array(
        [
            [0, 0, 0],
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 1.0],
        ],
        dtype=float,
    ) * scale

    frustum_points_world = (R_c2w @ frustum_points.T).T + position

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(frustum_points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.5)
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = position
    axes.transform(T)

    return [line_set, axes]


def jump_view_params_from_pose(position, R_c2w, camera_convention: str):
    """
    Compute Open3D view params from c2w.
    """
    if camera_convention == "opencv":
        forward = R_c2w[:, 2]
        up = -R_c2w[:, 1]
    elif camera_convention == "opengl":
        forward = -R_c2w[:, 2]
        up = R_c2w[:, 1]
    elif camera_convention == "ros":
        forward = R_c2w[:, 0]
        up = R_c2w[:, 2]
    else:
        forward = R_c2w[:, 2]
        up = -R_c2w[:, 1]

    lookat = position + forward * 5.0
    front = -forward
    return front, up, lookat


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize camera poses on a mesh/pointcloud")

    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Path to mesh/pointcloud. If omitted, uses config paths.yml",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
        help="Path to poses.txt (timestamp,sensor,qw,qx,qy,qz,tx,ty,tz,...)",
    )

    # This is the terminal argument you asked for (enable the block conversion).
    parser.add_argument(
        "--pose-format",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Interpretation of poses.txt. Use 'w2c' if poses are COLMAP cam_from_world.",
    )

    parser.add_argument(
        "--apply-mesh-alignment",
        action="store_true",
        help="Apply alignment_global.txt to the MESH (NOT to poses).",
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
        help="Which label to read from alignment_global.txt",
    )

    parser.add_argument(
        "--camera-convention",
        choices=["opencv", "opengl", "ros"],
        default="opencv",
        help="Affects jump-to-view with key 'I'.",
    )
    parser.add_argument("--frustum-scale", type=float, default=0.5)

    args = parser.parse_args()

    # Resolve mesh path
    if args.mesh is None:
        paths = load_paths()
        mesh_path = paths.mesh_path
        print(f"[mesh] Using mesh from config: {mesh_path}")
    else:
        mesh_path = args.mesh

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not args.poses.exists():
        raise FileNotFoundError(f"Poses file not found: {args.poses}")

    # Load geometry
    print(f"[mesh] Loading: {mesh_path}")
    geom = load_geometry(mesh_path)

    # Apply alignment to mesh only
    if args.apply_mesh_alignment:
        if args.alignment_file is not None:
            alignment_path = args.alignment_file
        else:
            paths = load_paths()
            alignment_path = paths.global_alignment_file

        T_s2w = read_alignment_pose_line(alignment_path, label=args.alignment_label)
        print(f"[mesh] Applying alignment to mesh: {alignment_path} (label={args.alignment_label})")
        geom.transform(T_s2w)

    # Load poses
    poses_raw = read_poses_file(args.poses)
    if not poses_raw:
        raise RuntimeError(f"No poses parsed from: {args.poses}")
    print(f"[poses] Loaded {len(poses_raw)} pose(s) from: {args.poses}")
    print(f"[poses] Interpreting pose file as: {args.pose_format}")

    # Convert to c2w for drawing
    poses_c2w = []
    for p in poses_raw:
        q = p["q"]
        t = p["t"]

        if args.pose_format == "c2w":
            R_c2w = quaternion_to_rotation_matrix(q)
            C_world = t
        else:
            # Your requested conversion block:
            # R_w2c = quaternion_to_rotation_matrix(q_w2c)
            # C_world = -R_w2c.T @ t_w2c
            # R_c2w = R_w2c.T
            R_c2w, C_world = convert_pose_w2c_to_c2w(q, t)

        poses_c2w.append(
            dict(
                timestamp=p["timestamp"],
                sensor_id=p["sensor_id"],
                position=C_world,
                R_c2w=R_c2w,
            )
        )

    # Build vis geometries
    geometries = [geom]
    palette = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    ]

    for i, p in enumerate(poses_c2w):
        color = palette[i % len(palette)]
        geometries.extend(
            create_camera_frustum(
                position=p["position"],
                R_c2w=p["R_c2w"],
                scale=args.frustum_scale,
                color=color,
            )
        )
        print(f"\nCamera {i+1}: {p['sensor_id']} @ {p['timestamp']}")
        print(f"  position (world): {p['position']}")

    print("\n" + "=" * 60)
    print("Controls:")
    print("  - Mouse: rotate view")
    print("  - Scroll: zoom")
    print("  - Ctrl+Click: pan")
    print("  - I: jump to camera view (cycles through poses)")
    print("  - Q/ESC: close")
    print("=" * 60 + "\n")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Camera Pose Visualization", width=1280, height=720)

    for g in geometries:
        vis.add_geometry(g)

    idx_state = {"i": 0}

    def on_jump_to_camera_view(v):
        if not poses_c2w:
            return False
        i = idx_state["i"] % len(poses_c2w)
        p = poses_c2w[i]
        pos = p["position"]
        R_c2w = p["R_c2w"]

        front, up, lookat = jump_view_params_from_pose(pos, R_c2w, args.camera_convention)

        ctr = v.get_view_control()
        ctr.set_lookat(lookat)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(0.5)

        print(f"\nJumped to Camera {i+1}/{len(poses_c2w)}")
        print(f"  pos: {pos}")
        print(f"  lookat: {lookat}")

        idx_state["i"] += 1
        return False

    vis.register_key_callback(ord("I"), on_jump_to_camera_view)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
