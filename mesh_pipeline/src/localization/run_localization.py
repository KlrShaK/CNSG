#!/usr/bin/env python3
"""
Run single image localization using LaMAR Docker container.

This script automatically loads NavVis data paths from the project config
and runs the LaMAR localization pipeline in Docker.

Usage:
    python run_localization.py --query_image /path/to/image.jpg
    python run_localization.py --query_image /path/to/image.jpg --output_dir ./my_outputs
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path to import config utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_utils import load_paths


def check_docker_image(image_name: str = "lamar:lamar") -> bool:
    """Check if the required Docker image exists."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def build_docker_command(
    repo_path: Path,
    navvis_data_path: Path,
    query_image_path: Path,
    map_session_path: Path,
    output_dir: Path,
    docker_image: str = "lamar:lamar"
) -> list:
    """Build the Docker command for localization."""

    # Get the directory containing the query image for volume mounting
    query_image_dir = query_image_path.parent.absolute()

    docker_cmd = [
        "docker", "run",
        "-it", "--rm", "--init",
        "--shm-size=16g",
        # Mount the LaMAR repository
        "-v", f"{repo_path}:{repo_path}",
        # Mount the NavVis data directory
        "-v", f"{navvis_data_path}:{navvis_data_path}",
        # Mount the query image directory
        "-v", f"{query_image_dir}:{query_image_dir}",
        # Set working directory
        "-w", str(repo_path),
        # Docker image
        docker_image,
        # Python script and arguments
        "python", "localize_single_image.py",
        "--map_path", str(map_session_path),
        "--query_image", str(query_image_path.absolute()),
        "--output_dir", "./outputs",
    ]

    return docker_cmd


def run_localization(query_image: Path, output_dir: Path = None):
    """Run the localization pipeline."""

    # Load paths from config
    print("Loading configuration...")
    try:
        paths = load_paths()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Make sure config/paths.yml is properly configured.")
        sys.exit(1)

    # Validate query image exists
    if not query_image.exists():
        print(f"Error: Query image not found: {query_image}")
        sys.exit(1)

    # Validate session directory exists
    if not paths.session_dir.exists():
        print(f"Error: NavVis session directory not found: {paths.session_dir}")
        print(f"Expected session: {paths.session_dir}")
        sys.exit(1)

    # Set up paths
    repo_root = paths.root
    lamar_repo = repo_root / "third_party" / "lamar-benchmark"

    if not lamar_repo.exists():
        print(f"Error: LaMAR repository not found: {lamar_repo}")
        print("Make sure the lamar-benchmark submodule is initialized.")
        sys.exit(1)

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = paths.outputs_root
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check if Docker image exists
    docker_image = "lamar:lamar"
    print(f"\nChecking for Docker image '{docker_image}'...")
    if not check_docker_image(docker_image):
        print(f"Warning: Docker image '{docker_image}' not found.")
        print("\nTo build the image, run:")
        print(f"  cd {lamar_repo}")
        print(f"  docker build --target lamar -t {docker_image} -f Dockerfile ./")

        response = input("\nWould you like to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Build Docker command
    print(f"\n{'='*60}")
    print("Running LaMAR Localization")
    print(f"{'='*60}")
    print(f"Query image:    {query_image}")
    print(f"NavVis session: {paths.session_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"{'='*60}\n")

    docker_cmd = build_docker_command(
        repo_path=lamar_repo,
        navvis_data_path=paths.data_root,
        query_image_path=query_image,
        map_session_path=paths.session_dir,
        output_dir=output_dir,
        docker_image=docker_image
    )

    # Print the command for debugging
    print("Docker command:")
    print(" ".join(docker_cmd))
    print()

    # Run the Docker command
    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Docker command failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nLocalization interrupted by user.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Localization complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Localize a single image using LaMAR and NavVis data"
    )
    parser.add_argument(
        "--query_image",
        type=Path,
        required=True,
        help="Path to the query image to localize"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for results (default: outputs/localization)"
    )

    args = parser.parse_args()

    run_localization(args.query_image, args.output_dir)


if __name__ == "__main__":
    main()
