#!/usr/bin/env python3

import argparse
from pathlib import Path
import open3d as o3d


def simplify_mesh(
    input_path: Path,
    output_path: Path,
    voxel_size: float = 0.08,
    simplify_factor: int = 5,
    max_error: float = 1e-8,
    save_voxel_mesh: bool = False,
    output_format: str = "glb",
):
    print(f"[INFO] Loading mesh: {input_path}")
    mesh = o3d.io.read_triangle_mesh(str(input_path))

    if not mesh.has_triangles():
        raise RuntimeError("Mesh has no triangles. Aborting.")

    print(f"[INFO] Number of vertices:  {len(mesh.vertices)}")
    print(f"[INFO] Number of triangles: {len(mesh.triangles)}")

    mesh.compute_vertex_normals()

    # -------------------------------------------------------
    # Step 1 — Vertex clustering (voxelization)
    # -------------------------------------------------------
    print(f"[INFO] Voxel clustering with voxel size = {voxel_size}")
    mesh_v = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average
    )
    mesh_v.compute_vertex_normals()

    print("[INFO] After voxel clustering:")
    print(f"[INFO]   Vertices:  {len(mesh_v.vertices)}")
    print(f"[INFO]   Triangles: {len(mesh_v.triangles)}")

    if save_voxel_mesh:
        vox_path = output_path.with_suffix(".voxelized.ply")
        o3d.io.write_triangle_mesh(str(vox_path), mesh_v)
        print(f"[INFO] Saved voxelized mesh: {vox_path}")

    # -------------------------------------------------------
    # Step 2 — Quadric decimation
    # -------------------------------------------------------
    # target_triangles = max(1, len(mesh_v.triangles) // simplify_factor)

    # print(f"[INFO] Quadric decimation: target = {target_triangles}, max_error = {max_error}")
    # mesh_q = mesh_v.simplify_quadric_decimation(
    #     target_number_of_triangles=target_triangles,
    #     maximum_error=max_error
    # )
    # mesh_q.compute_vertex_normals()

    # print("[INFO] After quadric decimation:")
    # print(f"[INFO]   Vertices:  {len(mesh_q.vertices)}")
    # print(f"[INFO]   Triangles: {len(mesh_q.triangles)}")

    # -------------------------------------------------------
    # Save final file (GLB or PLY)
    # -------------------------------------------------------
    # Fix output extension if needed
    output_path = output_path.with_suffix(f".{output_format}")

    print(f"[INFO] Saving final mesh as {output_format.upper()} → {output_path}")

    if output_format.lower() == "glb":
        o3d.io.write_triangle_mesh(
            str(output_path),
            mesh_v,
            write_ascii=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )
    elif output_format.lower() == "ply":
        o3d.io.write_triangle_mesh(
            str(output_path),
            mesh_v,
            write_ascii=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh simplification using Open3D")

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--voxel_size", type=float, default=0.08)
    parser.add_argument("--simplify_factor", type=int, default=5)
    parser.add_argument("--max_error", type=float, default=1e-8)
    parser.add_argument("--save_voxel_mesh", action="store_true")

    parser.add_argument(
        "--format",
        choices=["glb", "ply"],
        default="glb",
        help="Output mesh format (default: glb)"
    )

    args = parser.parse_args()

    simplify_mesh(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        simplify_factor=args.simplify_factor,
        max_error=args.max_error,
        save_voxel_mesh=args.save_voxel_mesh,
        output_format=args.format,
    )
