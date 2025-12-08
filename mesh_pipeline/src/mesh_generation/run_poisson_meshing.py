#!/usr/bin/env python3
"""
CLI entrypoint for running Poisson surface reconstruction on a point cloud.

Workflow:
1) Load a PLY point cloud.
2) Run Poisson reconstruction with optional density trimming and simplification.
3) Save the reconstructed mesh (and optionally a simplified mesh).
"""

import argparse
import logging
from pathlib import Path
import open3d as o3d

from poisson_meshing_module import mesh_from_pointcloud


def setup_logging(logfile: Path | None = None, verbose: bool = True) -> logging.Logger:
    """
    Configure logging to console and optional file.

    Args:
        logfile: Optional path to save logs.
        verbose: Whether to emit info-level logs (True) or only warnings (False).

    Returns:
        Configured logger instance.
    """
    level = logging.INFO if verbose else logging.WARNING
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"

    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    logger = logging.getLogger(__name__)
    return logger


def main(args: argparse.Namespace) -> None:
    """
    Run Poisson reconstruction on the provided point cloud and save outputs.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None
    """
    logger = setup_logging(args.log, verbose=not args.quiet)

    logger.info("---------------------------------------------------")
    logger.info("     Poisson Surface Reconstruction (Open3D)       ")
    logger.info("---------------------------------------------------")
    logger.info(f"Input pointcloud:  {args.input}")
    logger.info(f"Output mesh:       {args.output}")
    logger.info(f"Depth:             {args.depth}")
    logger.info(f"Density thresh:    ratio={args.density_ratio}, res={args.density_res}")
    logger.info(f"Simplify factor:   {args.simplify}")
    logger.info("---------------------------------------------------")

    # -------------------------------------------------------
    # Load pointcloud
    # -------------------------------------------------------
    logger.info("[INFO] Loading pointcloud...")
    pcd = o3d.io.read_point_cloud(str(args.input))
    logger.info(f"[INFO] Loaded pointcloud with {len(pcd.points):,} points")

    # -------------------------------------------------------
    # Run Poisson Reconstruction
    # -------------------------------------------------------
    logger.info("[INFO] Running Poisson meshing...")
    mesh, simplified, debug = mesh_from_pointcloud(
        pcd,
        psr_depth=args.depth,
        psr_densities_thresh_ratio=args.density_ratio,
        psr_densities_thresh_res=args.density_res,
        simplify_factor=args.simplify,
        simplify_error=args.simplify_error,
    )

    logger.info(f"[INFO] Final mesh:")
    logger.info(f"       Vertices:  {len(mesh.vertices):,}")
    logger.info(f"       Triangles: {len(mesh.triangles):,}")

    # -------------------------------------------------------
    # Save mesh
    # -------------------------------------------------------
    logger.info("[INFO] Saving output mesh...")
    o3d.io.write_triangle_mesh(str(args.output), mesh)
    logger.info(f"[INFO] Mesh saved to {args.output}")

    # -------------------------------------------------------
    # Save simplified mesh if required
    # -------------------------------------------------------
    if simplified is not None:
        simplified_path = Path(args.output).with_suffix(".simplified.ply")
        logger.info(f"[INFO] Saving simplified mesh → {simplified_path}")
        o3d.io.write_triangle_mesh(str(simplified_path), simplified)

    logger.info("---------------------------------------------------")
    logger.info("        Poisson reconstruction completed.          ")
    logger.info("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Poisson surface reconstruction on a PLY pointcloud.")

    parser.add_argument("--input", required=True, type=Path,
                        help="Input PLY pointcloud")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output mesh .ply")

    parser.add_argument("--depth", type=int, default=12,
                        help="Poisson octree depth (default: 12)")

    parser.add_argument("--density_ratio", type=float, default=0.01,
                        help="Remove bottom X percent of density (default: 0.01 = 1%)")

    parser.add_argument("--density_res", type=float, default=None,
                        help="Alternative absolute density resolution threshold")

    parser.add_argument("--simplify", type=int, default=None,
                        help="Quadric decimation factor (e.g. 5 means /5 triangles)")

    parser.add_argument("--simplify_error", type=float, default=1e-8,
                        help="Max error for quadric decimation")

    parser.add_argument("--log", type=Path, default=None,
                        help="Optional log file to save logs")

    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console logging (only warnings)")

    args = parser.parse_args()
    main(args)
