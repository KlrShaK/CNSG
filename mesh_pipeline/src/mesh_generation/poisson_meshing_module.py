#!/usr/bin/env python3
"""
Poisson surface reconstruction utilities for turning point clouds into meshes.

Workflow:
1) Clean a point cloud and estimate extent.
2) Run Poisson reconstruction with optional density trimming.
3) Optionally simplify via quadric decimation.

Functions here are imported by run_poisson_meshing.py.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import open3d as o3d
import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# Helper: Clean the mesh (important for Poisson indoors)
# -----------------------------------------------------------
def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Remove topology artifacts and recompute normals for a triangle mesh.

    Args:
        mesh: Input triangle mesh.

    Returns:
        Cleaned mesh with duplicates/degenerates removed and normals computed.
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


# -----------------------------------------------------------
# Poisson Meshing Function
# -----------------------------------------------------------
def poisson_meshing(
    pcd: o3d.geometry.PointCloud,
    depth: int = 12,
    resolution: Optional[float] = None,
    densities_thresh_resolution: Optional[float] = None,
    densities_thresh_ratio: float = 0.01,
) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
    """
    Run Poisson surface reconstruction on a point cloud with optional density trimming.

    Args:
        pcd: Input point cloud.
        depth: Poisson octree depth (higher captures more detail).
        resolution: Desired voxel resolution; overrides depth if provided.
        densities_thresh_resolution: Remove vertices below this density resolution (meters).
        densities_thresh_ratio: Remove the bottom X fraction of densities (0-1).

    Returns:
        mesh: Reconstructed triangle mesh.
        debug: Dict containing intermediate density arrays/masks.
    """
    debug = {}

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent().max()

    # If resolution is provided, derive depth
    if resolution is not None:
        depth = int(np.ceil(np.log2(extent / resolution)))
    else:
        resolution = extent / (2 ** depth)

    logger.info(
        f"[Poisson] Using depth={depth}, voxel resolution={resolution:.4f}m"
    )

    # Run Poisson
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Warning
    ):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth
        )
        debug["psr_densities"] = densities

    # ---------------------------
    # Density-based trimming
    # ---------------------------
    if densities_thresh_resolution is not None:
        # Remove vertices too far from sampled surface
        threshold = np.log2(extent / densities_thresh_resolution)
        mask = densities < threshold
        logger.info(f"[Poisson] Removing vertices below density={threshold:.3f}")
        mesh.remove_vertices_by_mask(mask)
        debug["psr_removed_by_density"] = mask

    elif densities_thresh_ratio:
        # Remove lowest X% of densities
        q = np.quantile(densities, densities_thresh_ratio)
        logger.info(f"[Poisson] Removing bottom {100 * densities_thresh_ratio:.1f}% density (threshold={q:.3f})")
        mask = densities < q
        mesh.remove_vertices_by_mask(mask)
        debug["psr_removed_by_density"] = mask

    # Final clean
    mesh = clean_mesh(mesh)

    return mesh, debug


# -----------------------------------------------------------
# Optional quadric simplification
# -----------------------------------------------------------
def simplify_mesh(mesh: o3d.geometry.TriangleMesh, factor: int, max_error: float) -> o3d.geometry.TriangleMesh:
    """
    Quadric decimation to reduce triangle count.

    Args:
        mesh: Input mesh.
        factor: Divide triangle count by this factor.
        max_error: Maximum allowable error for decimation.

    Returns:
        Simplified mesh with recomputed normals.
    """
    target = max(1, len(mesh.triangles) // factor)
    logger.info(
        f"[Simplify] Quadric decimation to {target} triangles (error={max_error})"
    )
    mesh_s = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target,
        maximum_error=max_error
    )
    mesh_s.compute_vertex_normals()
    return mesh_s


# -----------------------------------------------------------
# Main entry point (replaces advancing_front version)
# -----------------------------------------------------------
def mesh_from_pointcloud(
    pcd: o3d.geometry.PointCloud,
    psr_depth: int = 12,
    psr_resolution: Optional[float] = None,
    psr_densities_thresh_ratio: float = 0.01,
    psr_densities_thresh_res: Optional[float] = None,
    simplify_factor: Optional[int] = None,
    simplify_error: float = 1e-8,
) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
    """
    Top-level helper: run Poisson reconstruction and optional simplification.

    Args:
        pcd: Input point cloud.
        psr_depth: Poisson octree depth.
        psr_resolution: Target resolution (overrides depth if set).
        psr_densities_thresh_ratio: Remove bottom X fraction of densities.
        psr_densities_thresh_res: Remove densities below this absolute threshold.
        simplify_factor: Optional decimation factor (None to skip).
        simplify_error: Max error for decimation.

    Returns:
        mesh: Final (possibly simplified) mesh.
        debug: Dict of intermediate artifacts (densities, masks).
    """
    debug = {}

    # Run Poisson meshing
    mesh, dbg = poisson_meshing(
        pcd,
        depth=psr_depth,
        resolution=psr_resolution,
        densities_thresh_resolution=psr_densities_thresh_res,
        densities_thresh_ratio=psr_densities_thresh_ratio,
    )
    debug.update(dbg)

    simplified = None
    if simplify_factor:
        simplified = simplify_mesh(mesh, simplify_factor, simplify_error)

    return mesh, simplified, debug
