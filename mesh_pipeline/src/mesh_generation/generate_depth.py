"""
Raycast depth maps from a reconstructed mesh using NavVis trajectories.

Workflow:
1) Load camera intrinsics, poses, and image list for the session.
2) Load the Poisson mesh (solid surface).
3) For each frame, raycast depth from the mesh using the camera pose.
4) Save depth as both .npy (meters) and 16-bit PNG (millimeters).

All input/output paths are resolved from config/paths.yml via config_utils.load_paths().
"""

from typing import Dict, List
import open3d as o3d
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from pathlib import Path
import sys

# Make shared config loader available
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.config_utils import load_paths  # noqa: E402

# ================= CONFIGURATION =================
paths = load_paths()
# Solid mesh (not the point cloud)
MESH_PATH = paths.poisson_mesh
TRAJ_PATH = paths.trajectories_file
IMAGES_TXT = paths.images_file
SENSORS_TXT = paths.sensors_file
OUTPUT_DIR = paths.depth_maps_dir

# prefix for raw_data:
RAW_DATA_PREFIX = paths.raw_images_dir.parent
# =================================================

def load_intrinsics(sensors_file: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Parse sensors.txt to build the intrinsic matrix (K) for each camera.

    Args:
        sensors_file: Path to sensors.txt.

    Returns:
        Mapping sensor_id -> {"K": 3x3 intrinsics, "width": int, "height": int}
    """
    intrinsics = {}
    print(f"Loading intrinsics from {sensors_file}...")
    
    with open(sensors_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(', ')
            
            sensor_id = parts[0]
            if parts[2] != 'camera': continue
            
            # Parse params (indices based on your provided file)
            # W=1280(idx 4), H=1920(idx 5), fx(6), fy(7), cx(8), cy(9)
            w = int(parts[4])
            h = int(parts[5])
            fx = float(parts[6])
            fy = float(parts[7])
            cx = float(parts[8])
            cy = float(parts[9])
            
            # Build 3x3 Intrinsic Matrix
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            intrinsics[sensor_id] = {
                "K": K,
                "width": w,
                "height": h
            }
    return intrinsics

def load_poses_and_images(traj_file: Path, images_file: Path) -> List[Dict]:
    """
    Match images to their poses using trajectories.txt and images.txt.

    Args:
        traj_file: Path to trajectories.txt (timestamp, device, quaternion, translation).
        images_file: Path to images.txt (timestamp, sensor_id, image_path).

    Returns:
        List of dicts with filename, sensor_id, and 4x4 pose matrix.
    """
    print("Parsing trajectories and images...")
    
    # 1. Parse Trajectories
    # File: timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, ...
    pose_lookup = {} # Key: (timestamp, device_id) -> Matrix
    
    with open(traj_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(', ')
            
            ts = parts[0]
            dev_id = parts[1]
            
            # Quaternion (Input is w, x, y, z)
            qw, qx, qy, qz = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            # Scipy expects (x, y, z, w)
            r = R.from_quat([qx, qy, qz, qw])
            
            # Translation
            tx, ty, tz = float(parts[6]), float(parts[7]), float(parts[8])
            
            # Build Matrix
            T = np.eye(4)
            T[:3, :3] = r.as_matrix()
            T[:3, 3] = [tx, ty, tz]
            
            pose_lookup[(ts, dev_id)] = T

    # 2. Parse Images to find which poses we actually need
    # File: timestamp, sensor_id, image_path
    worklist = []
    
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(', ')
            
            ts = parts[0]
            sensor_id = parts[1]
            rel_path = parts[2]
            filename = Path(rel_path).name

            if (ts, sensor_id) in pose_lookup:
                pose = pose_lookup[(ts, sensor_id)]
                worklist.append({
                    "filename": filename,
                    "sensor": sensor_id,
                    "pose": pose
                })
            else:
                print(f"Warning: No pose found for {filename} (ts: {ts})")
                
    return worklist

def main() -> None:
    """
    Raycast depth maps from the Poisson mesh for every frame in the session.
    """
    # 1. Load Data
    cam_intrinsics = load_intrinsics(SENSORS_TXT)
    worklist = load_poses_and_images(TRAJ_PATH, IMAGES_TXT)
    
    # 2. Load Mesh & Setup Raycaster
    print(f"Loading Mesh from {MESH_PATH}...")
    # Using Open3D Tensor-based geometry for Raycasting
    mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Ray Casting for {len(worklist)} images...")
    
    # Group by sensor to avoid re-creating rays constantly if W/H is same
    for item in tqdm(worklist):
        sensor_id = item['sensor']
        filename = item['filename']
        c2w = item['pose'] # Camera-to-World
        
        K = cam_intrinsics[sensor_id]["K"]
        W = cam_intrinsics[sensor_id]["width"]
        H = cam_intrinsics[sensor_id]["height"]
        
        # Open3D Raycasting needs "World-to-Camera" (Extrinsic) not "Camera-to-World" (Pose)
        # We calculate the inverse.
        w2c = np.linalg.inv(c2w)
        
        # Generate Rays
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=K,
            extrinsic_matrix=w2c,
            width_px=W,
            height_px=H
        )
        
        # Cast
        result = scene.cast_rays(rays)
        
        # Get Depth (Hit distance)
        depth = result['t_hit'].numpy()
        
        # Filter Infinity
        depth[np.isinf(depth)] = 0.0
        
        # SAVE
        # Option A: Save as .npy (Recommended for data processing)
        save_path = OUTPUT_DIR / Path(filename).with_suffix('.npy').name
        np.save(save_path, depth)
        
        # Option B: Save as PNG (For debugging visual only - scales data!)
        depth_mm = (depth * 1000).astype(np.uint16)

        # We use OpenCV to save 16-bit PNG
        save_path_png = OUTPUT_DIR / Path(filename).with_suffix('.png').name
        cv2.imwrite(str(save_path_png), depth_mm)

    print(f"Done! Depth maps saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
