"""
Fuse per-frame semantic masks onto a mesh to produce vertex-aligned labels.

Pipeline:
1) Load mesh, camera intrinsics, trajectories, and image lists.
2) For each frame: unproject depth + mask to 3D using intrinsics and pose.
3) Vote semantics onto nearest mesh vertices via a KD-tree.
4) Save a colorized mesh for quick visualization and raw vertex labels (.npy).

All input/output paths are resolved from config/paths.yml through config_utils.load_paths().
"""

from pathlib import Path
from typing import Dict, List, Tuple
import open3d as o3d
import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.config_utils import load_paths

# ================= CONFIGURATION =================
paths = load_paths()
# Update these absolute/relative paths to match your system
MESH_PATH = paths.mesh_path

TRAJ_PATH = paths.trajectories_file
IMAGES_TXT = paths.images_file
SENSORS_TXT = paths.sensors_file
DEPTH_DIR = paths.depth_maps_dir
MASK_DIR = paths.semantic_masks_dir

OUTPUT_PLY = paths.fused_mesh
OUTPUT_IDS = paths.fused_ids
# =================================================

def load_intrinsics(sensors_file: Path) -> Dict[str, np.ndarray]:
    """
    Parse sensors.txt for per-sensor intrinsic matrices (K).

    Args:
        sensors_file: Path to sensors.txt.

    Returns:
        Mapping from sensor_id -> 3x3 intrinsic matrix.
    """
    intrinsics = {}
    print(f"Loading intrinsics from {sensors_file}...")
    with open(sensors_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(', ')
            sensor_id = parts[0]
            if parts[2] != 'camera': continue
            
            # Format: id, name, type, model, W, H, fx, fy, cx, cy
            fx, fy = float(parts[6]), float(parts[7])
            cx, cy = float(parts[8]), float(parts[9])
            
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            intrinsics[sensor_id] = K
    return intrinsics

def load_poses_and_images(traj_file: Path, images_file: Path) -> List[Dict]:
    """
    Parse trajectories and images.txt to pair each frame with its pose.

    Args:
        traj_file: Path to trajectories.txt.
        images_file: Path to images.txt.

    Returns:
        List of dicts with filename, sensor id, and 4x4 pose matrix.
    """
    print("Parsing trajectories and images...")
    
    # 1. Parse Trajectories
    # Format: timestamp, device_id, qw, qx, qy, qz, tx, ty, tz
    pose_lookup = {}
    with open(traj_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(', ')
            ts, dev_id = parts[0], parts[1]
            
            # Quaternion: Input is (qw, qx, qy, qz) -> Scipy wants (x, y, z, w)
            qw, qx, qy, qz = [float(x) for x in parts[2:6]]
            r = R.from_quat([qx, qy, qz, qw])
            t = [float(x) for x in parts[6:9]]
            
            T = np.eye(4)
            T[:3, :3] = r.as_matrix()
            T[:3, 3] = t
            pose_lookup[(ts, dev_id)] = T

    # 2. Parse Image List
    worklist = []
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(', ')
            ts, sensor_id = parts[0], parts[1]
            # Filename is the last part of the path
            filename = os.path.basename(parts[2]) 
            
            if (ts, sensor_id) in pose_lookup:
                worklist.append({
                    "filename": filename,
                    "sensor": sensor_id,
                    "pose": pose_lookup[(ts, sensor_id)]
                })
    return worklist

def unproject_pixels(
    depth_im: np.ndarray,
    mask_im: np.ndarray,
    K: np.ndarray,
    T_c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth + semantic mask pixels to world-space 3D points.

    depth_im: 16-bit depth in millimeters.
    mask_im: 8-bit class IDs.

    Args:
        depth_im: Depth image in millimeters (H x W).
        mask_im: Semantic class ID image (H x W).
        K: 3x3 intrinsic matrix.
        T_c2w: 4x4 camera-to-world pose.

    Returns:
        points_world: (N, 3) array of 3D points in world coordinates.
        labels: (N,) array of class IDs aligned to points_world.
    """
    # Create pixel grid
    H, W = depth_im.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Filter valid depth (0 is sky/invalid, >20m is unreliable)
    # Using 20000mm (20m) as cutoff
    valid = (depth_im > 0) & (depth_im < 20000)
    
    # Flatten valid points
    u, v = u[valid], v[valid]
    z_mm = depth_im[valid]
    z = z_mm.astype(np.float32) / 1000.0 # Convert to meters
    labels = mask_im[valid]
    
    # Pinhole Unprojection (Camera Space)
    # x = (u - cx) * z / fx
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    
    points_cam = np.vstack((x, y, z)).T
    
    # Camera to World Transform
    # P_world = R * P_cam + t
    R_mat = T_c2w[:3, :3]
    t_vec = T_c2w[:3, 3]
    points_world = (R_mat @ points_cam.T).T + t_vec
    
    return points_world, labels

def main() -> None:
    """
    Fuse per-frame semantics into vertex labels and save mesh + labels.
    """
    # 1. Load Data
    cam_intrinsics = load_intrinsics(SENSORS_TXT)
    worklist = load_poses_and_images(TRAJ_PATH, IMAGES_TXT)
    
    print(f"Loading Mesh from {MESH_PATH}...")
    mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))
    vertices = np.asarray(mesh.vertices)
    
    print(f"Building KD-Tree for {len(vertices)} vertices...")
    # This enables fast lookup of "which vertex is this pixel hitting?"
    pcd_tree = cKDTree(vertices)
    
    # 2. Vote Matrix
    # ADE20k has 150 classes. We use 151 to handle potential 0-indexing confusion.
    num_classes = 151
    # Rows = Vertices, Cols = Classes. Value = Number of votes.
    vote_matrix = np.zeros((len(vertices), num_classes), dtype=np.uint16)
    
    print("Starting Fusion/Voting...")
    
    # Optimization: Skip frames if you have too many (e.g. stride=5)
    stride = 1 
    for i in tqdm(range(0, len(worklist), stride)):
        item = worklist[i]
        fname = item['filename']
        sensor = item['sensor']
        
        # Construct paths
        # Note: We assume depth and masks have same filename as image but different extension
        base_name = os.path.splitext(fname)[0]
        depth_path = DEPTH_DIR / f"{base_name}.png"
        mask_path = MASK_DIR / f"{base_name}.png"
        
        if not depth_path.exists() or not mask_path.exists():
            continue
            
        # Load Images
        # IMREAD_UNCHANGED is CRITICAL for 16-bit depth
        depth_im = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        mask_im = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        
        # Unproject
        points_3d, labels = unproject_pixels(depth_im, mask_im, cam_intrinsics[sensor], item['pose'])
        
        if len(points_3d) == 0: continue
        
        # Map 3D points to Nearest Mesh Vertices
        # distance_upper_bound=0.05 means we ignore points >5cm away from any vertex
        dists, vert_indices = pcd_tree.query(points_3d, distance_upper_bound=0.05, workers=-1)
        
        # Filter infinite distances (no match found within 5cm)
        valid_hits = dists != float('inf')
        vert_indices = vert_indices[valid_hits]
        labels = labels[valid_hits]
        
        # Cast Votes
        # np.add.at is a fast, unbuffered in-place add
        np.add.at(vote_matrix, (vert_indices, labels), 1)

    # 3. Finalize
    print("Aggregating votes...")
    final_labels = np.argmax(vote_matrix, axis=1)
    
    # Save Colorized Mesh for Visualization
    print(f"Saving {OUTPUT_PLY}...")
    
    # Generate random colors for classes to visualize
    np.random.seed(42)
    color_palette = np.random.rand(num_classes, 3)
    vertex_colors = color_palette[final_labels]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.io.write_triangle_mesh(str(OUTPUT_PLY), mesh)
    
    # Also save the raw IDs (important for Habitat!)
    np.save(str(OUTPUT_IDS), final_labels)
    
    print("Done! You can now view the colored mesh in CloudCompare/MeshLab.")

if __name__ == "__main__":
    main()
