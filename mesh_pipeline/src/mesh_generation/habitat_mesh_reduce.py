#!/usr/bin/env python3
"""
Prepare a Habitat-friendly mesh by simplifying and cleaning a GLB/PLY asset.

Workflow:
1) Load the input mesh.
2) Simplify via voxel clustering (optionally save voxelized intermediate).
3) Clean mesh topology (remove duplicates/degenerates, recompute normals).
4) Export the result for Habitat usage.
"""

import argparse
from pathlib import Path
import open3d as o3d


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Ensure the mesh is suitable for Habitat navmesh generation.

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


def reduce_mesh(
    input_path: Path,
    output_path: Path,
    voxel_size: float = 0.08,
    simplify_factor: int = 5,
    max_error: float = 1e-8,
    save_voxel_mesh: bool = False,
) -> None:
    """
    Simplify and clean a mesh for Habitat consumption.

    Args:
        input_path: Path to the source mesh (PLY/GLB).
        output_path: Path for the reduced mesh (GLB/PLY).
        voxel_size: Voxel grid size for clustering-based simplification.
        simplify_factor: Placeholder for quadric decimation factor (not used currently).
        max_error: Placeholder for quadric decimation max error (not used currently).
        save_voxel_mesh: Whether to also save the voxelized intermediate mesh.

    Returns:
        None
    """
    print("\n=========================================")
    print("    HABITAT MESH REDUCTION PIPELINE       ")
    print("=========================================\n")

    print(f"[INFO] Loading mesh: {input_path}")
    mesh = o3d.io.read_triangle_mesh(str(input_path))

    if not mesh.has_triangles():
        raise RuntimeError("Input mesh has no triangles!")

    mesh.compute_vertex_normals()

    print(f"[INFO] Loaded mesh:")
    print(f"       Vertices:  {len(mesh.vertices):,}")
    print(f"       Triangles: {len(mesh.triangles):,}\n")

    # ----------------------------------------------------
    # STEP 1: Vertex Clustering (Voxel Simplification)
    # ----------------------------------------------------
    print(f"[INFO] Voxel clustering with voxel_size = {voxel_size}")
    mesh_v = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )
    mesh_v = clean_mesh(mesh_v)

    print(f"[INFO] After voxel clustering:")
    print(f"       Vertices:  {len(mesh_v.vertices):,}")
    print(f"       Triangles: {len(mesh_v.triangles):,}\n")

    if save_voxel_mesh:
        voxel_path = output_path.with_suffix(".voxelized.ply")
        o3d.io.write_triangle_mesh(str(voxel_path), mesh_v)
        print(f"[INFO] Saved voxelized mesh to: {voxel_path}\n")

    # ----------------------------------------------------
    # STEP 2: Quadric Decimation (currently disabled; placeholders kept)
    # ----------------------------------------------------
    # target_triangles = max(1, len(mesh_v.triangles) // simplify_factor)
    # print("[INFO] Quadric decimation:")
    # print(f"       Target triangles = {target_triangles:,}")
    # print(f"       Max error = {max_error}")
    # mesh_q = mesh_v.simplify_quadric_decimation(
    #     target_number_of_triangles=target_triangles,
    #     maximum_error=max_error,
    # )
    # mesh_q = clean_mesh(mesh_q)

    # ----------------------------------------------------
    # STEP 3: Export Final GLB
    # ----------------------------------------------------
    print(f"[INFO] Saving final GLB: {output_path}")
    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh_v,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )

    if not success:
        raise RuntimeError("Failed to write GLB output file.")

    print("\n=========================================")
    print("        Mesh Reduction Completed          ")
    print("=========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat-safe mesh reduction")

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--voxel_size", type=float, default=0.08)
    parser.add_argument("--simplify_factor", type=int, default=5)
    parser.add_argument("--max_error", type=float, default=1e-8)
    parser.add_argument("--save_voxel_mesh", action="store_true")

    args = parser.parse_args()

    reduce_mesh(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        simplify_factor=args.simplify_factor,
        max_error=args.max_error,
        save_voxel_mesh=args.save_voxel_mesh,
    )
