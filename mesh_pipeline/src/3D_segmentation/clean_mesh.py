"""
Clean and refine fused semantic meshes.

Workflow:
1) Load the fused semantic mesh and vertex labels produced by fuse_labels.py.
2) Enforce geometric consistency with DBSCAN super-point clustering.
3) Smooth semantics with KNN voting.
4) Save a cleaned mesh (colorized for quick visualization) and cleaned label .npy.

Inputs/outputs are resolved from config/paths.yml via config_utils.load_paths().
"""

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
import copy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.config_utils import load_paths

# ================= CONFIGURATION =================
paths = load_paths()
INPUT_PLY = paths.fused_mesh
OUTPUT_PLY = paths.fused_mesh_clean
INPUT_LABELS = paths.fused_ids
OUTPUT_LABELS = paths.fused_ids_clean

# 1. Geometric Clustering (Super-points)
# Tuning for LARGE BUILDING (ETH HG):
# EPS=0.05 (5cm): Groups distinct objects.
# Min_Samples=50: Ignores tiny mesh artifacts.
CLUSTER_EPS = 0.05  
MIN_CLUSTER_POINTS = 50 

# 2. Smoothing
# Neighbors=50: Look at a wider area for voting.
# Iterations=5: Run multiple passes to erode noisy boundaries.
KNN_NEIGHBORS = 50
KNN_ITERATIONS = 5
# =================================================

def geometric_refinement(
    mesh: o3d.geometry.TriangleMesh,
    labels: np.ndarray,
    eps: float = 0.05,
    min_points: int = 50
) -> np.ndarray:
    """
    Cluster nearby vertices (position + normals) with DBSCAN and assign each
    cluster the majority semantic label to make objects solid.

    Parameters:
    mesh (o3d.geometry.TriangleMesh): Input 3D mesh.
    labels (np.ndarray): Semantic labels for each vertex in the mesh.
    eps (float): Maximum distance between points in a cluster.
    min_points (int): Minimum number of samples required to form a dense region.

    Returns:
    np.ndarray: New semantic labels for each vertex in the mesh after geometric refinement.
    """
    print(f"Refining: Clustering mesh based on Geometry (eps={eps}, min_samples={min_points})...")
    
    # Check for scikit-learn
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("Error: scikit-learn is required for 6D geometric clustering.")
        print("Please run: pip install scikit-learn")
        sys.exit(1)

    # 1. Prepare Features for Clustering (Position + Normals)
    verts = np.asarray(mesh.vertices)
    
    if not mesh.has_vertex_normals():
        print("Computing vertex normals...")
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    
    # Weight normals heavily so we distinguish floor from wall (90 deg difference)
    # Concatenate [x,y,z] and [nx,ny,nz]*weight
    # A weight of 2.0 ensures that surfaces with different normals are "far apart" mathematically
    features = np.hstack((verts, normals * 2.0))
    
    # 2. Run DBSCAN Clustering (using Scikit-Learn)
    # n_jobs=-1 uses all CPU cores
    print("Running DBSCAN (this may take a moment for large meshes)...")
    clustering = DBSCAN(eps=eps, min_samples=min_points, metric='euclidean', n_jobs=-1)
    cluster_ids = clustering.fit_predict(features)
    
    max_label = cluster_ids.max()
    print(f"Found {max_label + 1} geometric clusters.")
    
    new_labels = labels.copy()
    
    # 3. Majority Vote per Cluster
    print("Voting on clusters...")
    
    unique_clusters = np.unique(cluster_ids)
    
    for cid in unique_clusters:
        if cid == -1: continue # Skip noise
        
        # Get indices of points in this cluster
        indices = np.where(cluster_ids == cid)[0]
        
        # Get their current semantic labels
        current_semantic = labels[indices]
        
        # Find winner
        counts = np.bincount(current_semantic, minlength=152)
        winner = np.argmax(counts)
        
        # Force all points in this geometric cluster to take the winner label
        new_labels[indices] = winner
        
    return new_labels

def knn_smoothing(
    mesh: o3d.geometry.TriangleMesh, 
    labels: np.ndarray, 
    k: int = 50, 
    iterations: int = 5
) -> np.ndarray:
    """
    Run iterative KNN majority voting to erode salt-and-pepper noise between classes.

    Parameters:
    mesh (o3d.geometry.TriangleMesh): Input 3D mesh.
    labels (np.ndarray): Semantic labels for each vertex in the mesh.
    k (int): Number of nearest neighbors to consider for each vertex.
    iterations (int): Number of KNN smoothing iterations to run.

    Returns:
    np.ndarray: Smoothed semantic labels for each vertex in the mesh.
    """
    print(f"Refining: KNN Smoothing (k={k}, iter={iterations})...")
    vertices = np.asarray(mesh.vertices)
    tree = cKDTree(vertices)
    current_labels = labels.copy()
    
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...")
        # Chunk size for memory management
        chunk_size = 50000 
        new_labels = np.zeros_like(current_labels)
        
        for start_idx in range(0, len(vertices), chunk_size):
            end_idx = min(start_idx + chunk_size, len(vertices))
            batch_verts = vertices[start_idx:end_idx]
            
            # Query neighbors
            _, neighbor_indices = tree.query(batch_verts, k=k, workers=-1)
            
            # Vectorized voting for the batch
            for local_idx, row_labels in enumerate(current_labels[neighbor_indices]):
                counts = np.bincount(row_labels, minlength=152)
                new_labels[start_idx + local_idx] = np.argmax(counts)
                
        current_labels = new_labels.copy()
    return current_labels

def main():
    """
    Load fused mesh + labels, refine with geometry-aware clustering and KNN smoothing,
    then save cleaned mesh/labels for downstream export.
    """
    print(f"Loading {INPUT_PLY}...")
    mesh = o3d.io.read_triangle_mesh(str(INPUT_PLY))
    
    # Load raw labels
    try:
        labels = np.load(str(INPUT_LABELS))
        print("Loaded labels from .npy")
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_LABELS}'.")
        print("Please make sure you ran the previous script (fusion) which generates this file.")
        return

    # STEP 1: Geometric Refinement
    # Forces objects to be "solid" based on geometry (Position + Normals)
    # This separates the Floor from the Wall even if they touch.
    labels = geometric_refinement(mesh, labels, eps=CLUSTER_EPS, min_points=MIN_CLUSTER_POINTS)

    # STEP 2: KNN Smoothing 
    # Erodes noisy boundaries between objects (salt-and-pepper noise)
    labels = knn_smoothing(mesh, labels, k=KNN_NEIGHBORS, iterations=KNN_ITERATIONS)
    
    # Save
    print(f"Saving cleaned mesh to {OUTPUT_PLY}...")
    num_classes = 151
    np.random.seed(42)
    color_palette = np.random.rand(num_classes, 3)
    
    # Visualization Colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(color_palette[labels])
    o3d.io.write_triangle_mesh(str(OUTPUT_PLY), mesh)
    
    # Save Clean IDs
    np.save(str(OUTPUT_LABELS), labels)
    print("Done.")

if __name__ == "__main__":
    main()
