#!/usr/bin/env python3
"""Build a COLMAP map from the NavVis session and localize query images.

Mapping (session images):
  - Feature extraction using intrinsics from sensors.txt.
  - Sequential matching across the capture order.
  - Incremental mapping + sparse model export (.ply).
  - Vocabulary tree built for fast retrieval (or reuse a pre-trained tree with --vocab-tree).

Localization (query images in data/test_images by default):
  - Feature extraction on queries.
  - Vocabulary-tree-based pair selection + matching against the map.
  - Absolute pose estimation via image_registrator; exports a localized model.

Run from the repository root, e.g.:
  python scripts/run_colmap_navvis.py
  python scripts/run_colmap_navvis.py --skip-map        # only localize into an existing map
  python scripts/run_colmap_navvis.py --skip-localize   # only build the map
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm


def _run(cmd: List[str]) -> None:
    """Log and run a shell command, raising on failure."""
    logging.info("[cmd] %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_defaults() -> Dict[str, Path]:
    cfg_path = Path("config/paths.yml")
    session_dir = Path("data/navvis_2022-02-06_12.55.11")
    image_list = session_dir / "images.txt"
    sensors = session_dir / "sensors.txt"
    query_dir = Path("data/test_images")

    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())["paths"]
        session_dir = Path(cfg["data_root"]) / cfg["session_name"]
        image_list = session_dir / cfg["images_file"]
        sensors = session_dir / cfg["sensors_file"]
        query_dir = Path(cfg.get("query_dir", query_dir))

    output_root = session_dir / "colmap"
    return {
        "session_dir": session_dir,
        "image_list": image_list,
        "sensors": sensors,
        "output_root": output_root,
        "query_dir": query_dir,
    }


def _load_image_names(image_list_path: Path) -> List[str]:
    names: List[str] = []
    with image_list_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading image list", leave=False):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            names.append(parts[2])
    # preserve order while removing duplicates
    seen = set()
    unique_names = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
    return unique_names


def _parse_cameras(sensors_path: Path) -> Dict[str, Dict[str, object]]:
    cameras: Dict[str, Dict[str, object]] = {}
    with sensors_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 10 or row[2].strip() != "camera":
                continue
            sensor_id = row[0].strip()
            model = row[3].strip()
            width = int(row[4].strip())
            height = int(row[5].strip())
            fx, fy, cx, cy = map(float, row[6:10])
            cameras[sensor_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": (fx, fy, cx, cy),
            }
    return cameras


def _write_list(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _fmt_params(params: Iterable[float]) -> str:
    return ",".join(f"{p:.8f}".rstrip("0").rstrip(".") if "." in f"{p:.8f}" else f"{p:.8f}" for p in params)


def _resolve_image_root(session_dir: Path, image_names: List[str]) -> Tuple[Path, List[str]]:
    """Find an image root so listed images exist; try session_dir and session_dir/raw_data."""
    candidates = [(session_dir, image_names), (session_dir / "raw_data", image_names)]

    # Also try prefixing raw_data/ into the list if that matches the disk layout.
    prefixed = [f"raw_data/{name}" for name in image_names]
    candidates.append((session_dir, prefixed))

    first_missing = None
    for root, names in candidates:
        missing = [rel for rel in names if not (root / rel).exists()]
        if not missing:
            return root, names
        if first_missing is None and missing:
            first_missing = root / missing[0]
    raise FileNotFoundError(f"Could not resolve image paths; first missing file: {first_missing}")


def run_mapping(
    session_dir: Path,
    image_list_path: Path,
    sensors_path: Path,
    output_root: Path,
    num_threads: int,
    seq_overlap: int,
    use_gpu: bool,
    reset: bool,
    vocab_tree_override: Optional[Path],
    match_use_gpu: bool,
    max_matches: int,
) -> Tuple[Path, Path]:
    map_root = output_root / "map"
    sparse_dir = map_root / "sparse"
    model_dir = sparse_dir / "0"
    vocab_tree_path = map_root / "vocab_tree.bin" if vocab_tree_override is None else vocab_tree_override
    db_path = map_root / "database.db"
    image_list_out = map_root / "mapping_image_list.txt"

    if reset and map_root.exists():
        shutil.rmtree(map_root)
    map_root.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_names = _load_image_names(image_list_path)
    image_root, resolved_names = _resolve_image_root(session_dir, image_names)
    logging.info("Resolved image root: %s", image_root)
    logging.info("Images to process: %d", len(resolved_names))
    logging.debug("First 5 images: %s", resolved_names[:5])
    logging.info(
        "Matching mode: %s, max matches: %d",
        "GPU" if match_use_gpu else "CPU",
        max_matches,
    )
    _write_list(image_list_out, resolved_names)

    cameras = _parse_cameras(sensors_path)
    if not cameras:
        logging.warning("No camera entries found in %s, using COLMAP defaults.", sensors_path)
        camera_model = "PINHOLE"
        camera_params = None
    else:
        # Use the first camera as representative (NavVis cameras share intrinsics).
        cam = cameras[sorted(cameras.keys())[0]]
        camera_model = str(cam["model"])
        camera_params = _fmt_params(cam["params"])
        logging.info(
            "Using camera model %s with intrinsics fx=%.3f, fy=%.3f, cx=%.3f, cy=%.3f",
            camera_model,
            cam["params"][0],
            cam["params"][1],
            cam["params"][2],
            cam["params"][3],
        )
        logging.debug("Full camera record: %s", cam)

    feature_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_root),
        "--image_list_path",
        str(image_list_out),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.num_threads",
        str(num_threads),
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
    ]
    if camera_params:
        feature_cmd += ["--ImageReader.camera_params", camera_params]
    _run(feature_cmd)

    # Build a vocab tree so localization can use fast retrieval.
    if vocab_tree_override is None:
        _run(
            [
                "colmap",
                "vocab_tree_builder",
                "--database_path",
                str(db_path),
                "--vocab_tree_path",
                str(vocab_tree_path),
            ]
        )
    else:
        logging.info("Using pre-trained vocab tree: %s", vocab_tree_override)

    _run(
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(db_path),
            "--SequentialMatching.overlap",
            str(seq_overlap),
            "--SiftMatching.num_threads",
            str(num_threads),
            "--SiftMatching.use_gpu",
            "1" if match_use_gpu else "0",
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.max_num_matches",
            str(max_matches),
        ]
    )

    _run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(image_root),
            "--output_path",
            str(sparse_dir),
            "--Mapper.num_threads",
            str(num_threads),
            "--Mapper.min_model_size",
            "3",
        ]
    )

    ply_path = map_root / "map.ply"
    if model_dir.exists():
        _run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(ply_path),
                "--output_type",
                "PLY",
            ]
        )
    else:
        logging.warning("No model found at %s, mapper may have failed.", model_dir)

    return model_dir, vocab_tree_path


def run_localization(
    map_model_dir: Path,
    map_db_path: Path,
    vocab_tree_path: Path,
    query_dir: Path,
    output_root: Path,
    num_threads: int,
    vocab_neighbors: int,
    use_gpu: bool,
    reset: bool,
    match_use_gpu: bool,
    max_matches: int,
) -> Path:
    loc_root = output_root / "localization"
    loc_db = loc_root / "database.db"
    loc_model_dir = loc_root / "registered"

    if reset and loc_root.exists():
        shutil.rmtree(loc_root)
    loc_root.mkdir(parents=True, exist_ok=True)

    if not map_db_path.exists():
        raise FileNotFoundError(f"Mapping database missing: {map_db_path}")
    if not vocab_tree_path.exists():
        raise FileNotFoundError(f"Vocab tree missing: {vocab_tree_path}")
    if not query_dir.exists():
        raise FileNotFoundError(f"Query images folder missing: {query_dir}")

    shutil.copy(map_db_path, loc_db)
    logging.info(
        "Localization matching mode: %s, max matches: %d",
        "GPU" if match_use_gpu else "CPU",
        max_matches,
    )

    _run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(loc_db),
            "--image_path",
            str(query_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--SiftExtraction.num_threads",
            str(num_threads),
            "--SiftExtraction.use_gpu",
            "1" if use_gpu else "0",
        ]
    )

    def run_matcher(vt_path: Path) -> None:
        _run(
            [
                "colmap",
                "vocab_tree_matcher",
                "--database_path",
                str(loc_db),
                "--VocabTreeMatching.vocab_tree_path",
                str(vt_path),
                "--VocabTreeMatching.num_images",
                str(vocab_neighbors),
                "--SiftMatching.num_threads",
                str(num_threads),
                "--SiftMatching.use_gpu",
                "1" if match_use_gpu else "0",
                "--SiftMatching.guided_matching",
                "1",
                "--SiftMatching.max_num_matches",
                str(max_matches),
            ]
        )

    try:
        run_matcher(vocab_tree_path)
    except subprocess.CalledProcessError:
        fallback_tree = loc_root / "vocab_tree_auto.bin"
        logging.warning(
            "Failed to use vocab tree at %s (likely legacy FLANN format). Rebuilding a FAISS tree at %s.",
            vocab_tree_path,
            fallback_tree,
        )
        _run(
            [
                "colmap",
                "vocab_tree_builder",
                "--database_path",
                str(loc_db),
                "--vocab_tree_path",
                str(fallback_tree),
            ]
        )
        run_matcher(fallback_tree)

    _run(
        [
            "colmap",
            "image_registrator",
            "--database_path",
            str(loc_db),
            "--input_path",
            str(map_model_dir),
            "--output_path",
            str(loc_model_dir),
            "--Mapper.num_threads",
            str(num_threads),
        ]
    )

    ply_path = loc_root / "localized.ply"
    if loc_model_dir.exists():
        _run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(loc_model_dir),
                "--output_path",
                str(ply_path),
                "--output_type",
                "PLY",
            ]
        )
    else:
        logging.warning("Localization output missing at %s", loc_model_dir)

    return loc_model_dir


def parse_args() -> argparse.Namespace:
    defaults = _load_defaults()
    parser = argparse.ArgumentParser(
        description="COLMAP mapping + localization for the NavVis LaMAR session."
    )
    parser.add_argument("--session-dir", type=Path, default=defaults["session_dir"])
    parser.add_argument("--image-list", type=Path, default=defaults["image_list"])
    parser.add_argument("--sensors", type=Path, default=defaults["sensors"])
    parser.add_argument("--query-dir", type=Path, default=defaults["query_dir"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--seq-overlap", type=int, default=4, help="Sequential matcher overlap window.")
    parser.add_argument(
        "--vocab-neighbors",
        type=int,
        default=80,
        help="Nearest neighbors per image for vocab_tree_matcher during localization.",
    )
    parser.add_argument(
        "--vocab-tree",
        type=Path,
        default=None,
        help="Path to a pre-trained vocab tree (skips building a new one).",
    )
    parser.add_argument(
        "--match-use-gpu",
        action="store_true",
        help="Use GPU for feature matching (may OOM on small GPUs). Default: CPU matching.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=8192,
        help="SiftMatching.max_num_matches to limit GPU memory usage (default 8192).",
    )
    parser.add_argument("--skip-map", action="store_true", help="Skip mapping stage.")
    parser.add_argument("--skip-localize", action="store_true", help="Skip localization stage.")
    parser.add_argument("--reset", action="store_true", help="Remove prior outputs before running.")
    parser.add_argument("--cpu", dest="use_gpu", action="store_false", help="Disable GPU acceleration.")
    parser.set_defaults(use_gpu=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    if not shutil.which("colmap"):
        logging.error("COLMAP binary not found in PATH.")
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    map_root = args.output_root / "map"
    map_db = map_root / "database.db"
    vocab_tree_path = args.vocab_tree if args.vocab_tree else map_root / "vocab_tree.bin"
    model_dir = map_root / "sparse" / "0"

    if not args.skip_map:
        model_dir, vocab_tree_path = run_mapping(
            session_dir=args.session_dir,
            image_list_path=args.image_list,
            sensors_path=args.sensors,
            output_root=args.output_root,
            num_threads=args.num_threads,
            seq_overlap=args.seq_overlap,
            use_gpu=args.use_gpu,
            reset=args.reset,
            vocab_tree_override=args.vocab_tree,
            match_use_gpu=args.match_use_gpu,
            max_matches=args.max_matches,
        )
        map_db = args.output_root / "map" / "database.db"
    elif args.vocab_tree:
        logging.info("Using supplied vocab tree for localization: %s", args.vocab_tree)

    if not args.skip_localize:
        run_localization(
            map_model_dir=model_dir,
            map_db_path=map_db,
            vocab_tree_path=vocab_tree_path,
            query_dir=args.query_dir,
            output_root=args.output_root,
            num_threads=args.num_threads,
            vocab_neighbors=args.vocab_neighbors,
            use_gpu=args.use_gpu,
            reset=args.reset,
            match_use_gpu=args.match_use_gpu,
            max_matches=args.max_matches,
        )


if __name__ == "__main__":
    main()
