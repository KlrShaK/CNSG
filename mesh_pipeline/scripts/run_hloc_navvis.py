#!/usr/bin/env python3
"""Headless NavVis HLoc pipeline: build SfM with HLoc (Option B) and localize queries.


python scripts/run_hloc_navvis.py --save-query-plot

"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Resolve repository root so the script works regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(REPO_ROOT / "third_party" / "Hierarchical-Localization"))
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction  # type: ignore  # noqa
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster  # type: ignore  # noqa
from hloc.utils.io import write_poses  # type: ignore  # noqa
from hloc.utils.parsers import parse_retrieval  # type: ignore  # noqa
from hloc.utils import viz_3d  # type: ignore  # noqa
import pycolmap  # type: ignore  # noqa

LOGGER = logging.getLogger("hloc_navvis")


def _log_skip(path: Path, action: str) -> bool:
    if path.exists():
        LOGGER.info("Found %s, skipping %s.", path, action)
        return True
    return False


def _load_sensors(path: Path) -> Dict[str, Dict[str, object]]:
    cameras: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return cameras
    with path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 10 or row[2].strip() != "camera":
                continue
            sensor_id = row[0].strip()
            model = row[3].strip()
            width, height = int(row[4]), int(row[5])
            fx, fy, cx, cy = map(float, row[6:10])
            cameras[sensor_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": [fx, fy, cx, cy],
            }
    return cameras


def _load_image_entries(image_list_path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    if not image_list_path.exists():
        return entries
    with image_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            entries.append((parts[1], parts[2]))
    seen = set()
    unique: List[Tuple[str, str]] = []
    for sensor_id, rel_path in entries:
        if rel_path in seen:
            continue
        seen.add(rel_path)
        unique.append((sensor_id, rel_path))
    return unique


def _resolve_image_root(
    session_dir: Path, entries: Sequence[Tuple[str, str]], default_dir: Path
) -> Tuple[Path, List[Tuple[str, str]]]:
    rel_paths = [rel for _, rel in entries]
    candidates = [
        (session_dir, rel_paths),
        (session_dir / "raw_data", rel_paths),
        (session_dir, [f"raw_data/{p}" for p in rel_paths]),
    ]
    if default_dir:
        base = [Path(p).name for p in rel_paths]
        candidates.append((default_dir.parent, [f"{default_dir.name}/{b}" for b in base]))
    for root, names in candidates:
        if names and all((root / n).exists() for n in names):
            resolved = list(zip([e[0] for e in entries], names))
            return root, resolved
    raise FileNotFoundError(
        f"Could not resolve mapping image root; first missing: {rel_paths[0] if rel_paths else 'n/a'}"
    )


def _list_images(image_root: Path) -> List[str]:
    files: List[Path] = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
        files += list(image_root.rglob(ext))
    files = sorted(set(files))
    return [p.relative_to(image_root).as_posix() for p in files]


def _build_sequential_pairs(names: Sequence[str], overlap: int) -> set[Tuple[str, str]]:
    pairs = set()
    for i, name in enumerate(names):
        for j in range(i + 1, min(i + overlap + 1, len(names))):
            pairs.add((name, names[j]))
    return pairs


def _save_pairs(pairs: Iterable[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")


def _read_pairs(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()
    pairs = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a, b = line.strip().split()
            pairs.add((a, b))
    return pairs


def _infer_camera(qpath: Path, fallback: Optional[Dict[str, object]]) -> pycolmap.Camera:
    try:
        return pycolmap.infer_camera_from_image(qpath)
    except Exception:
        if fallback is None:
            raise
        LOGGER.warning("EXIF intrinsics missing for %s, using fallback.", qpath.name)
        return pycolmap.Camera(
            model=fallback["model"],  # type: ignore[index]
            width=int(fallback["width"]),  # type: ignore[index]
            height=int(fallback["height"]),  # type: ignore[index]
            params=fallback["params"],  # type: ignore[index]
        )


def _write_loc_logs(
    results_path: Path,
    loc_logs: Dict[str, dict],
    retrieval: Dict[str, List[str]],
    features: Path,
    matches: Path,
) -> None:
    with open(f"{results_path}_logs.pkl", "wb") as f:
        pickle.dump(
            {"loc": loc_logs, "retrieval": retrieval, "features": features, "matches": matches},
            f,
        )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=REPO_ROOT / "data/navvis_2022-02-06_12.55.11",
    )
    parser.add_argument(
        "--mapping-subdir",
        type=str,
        default="raw_data/images_undistr_center",
        help="Relative to session-dir.",
    )
    parser.add_argument("--query-dir", type=Path, default=REPO_ROOT / "data/test_images")
    parser.add_argument(
        "--output-root",
        type=Path,
        # help="Defaults to <repo_root>/outputs/hloc/<session>",
        help="Defaults to <repo_root>/outputs/hloc/<session>",
    )
    parser.add_argument("--top-k-retrieval", type=int, default=20)
    parser.add_argument("--top-k-map", type=int, default=16)
    parser.add_argument("--seq-overlap", type=int, default=8)
    parser.add_argument("--ransac", type=float, default=12.0)
    parser.add_argument(
        "--save-query-plot",
        action="store_true",
        help="Save an HTML plot with only query poses visualized in the map.",
    )
    parser.add_argument(
        "--query-plot-path",
        type=Path,
        help="Optional output path for the query pose plot (HTML).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Device: %s", device)

    session_dir = args.session_dir if args.session_dir.is_absolute() else REPO_ROOT / args.session_dir
    mapping_subdir = Path(args.mapping_subdir)
    mapping_dir_default = mapping_subdir if mapping_subdir.is_absolute() else session_dir / mapping_subdir
    image_list_path = session_dir / "images.txt"
    sensors_path = session_dir / "sensors.txt"
    if args.output_root is not None:
        output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    else:
        output_root = REPO_ROOT / "outputs/hloc" / session_dir.name

    paths = {
        "features": output_root / "features",
        "global": output_root / "global",
        "pairs": output_root / "pairs",
        "matches": output_root / "matches",
        "sfm": output_root / "sfm",
        "loc": output_root / "localization",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    map_features_path = paths["features"] / "feats-superpoint-n4096-r1024_mapping.h5"
    query_features_path = paths["features"] / "feats-superpoint-n4096-r1024_queries.h5"
    map_global_path = paths["global"] / "global-feats-netvlad_mapping.h5"
    query_global_path = paths["global"] / "global-feats-netvlad_queries.h5"
    map_pairs_seq_path = paths["pairs"] / f"pairs-map-seq-overlap{args.seq_overlap}.txt"
    map_pairs_retr_path = paths["pairs"] / f"pairs-map-retrieval-top{args.top_k_map}.txt"
    map_pairs_path = paths["pairs"] / "pairs-map-combined.txt"
    map_matches_path = paths["matches"] / "matches-superglue_map.h5"
    loc_pairs_path = paths["pairs"] / f"pairs-query-top{args.top_k_retrieval}.txt"
    loc_matches_path = paths["matches"] / "matches-superglue_loc.h5"
    loc_results_path = paths["loc"] / "hloc_navvis_results.txt"

    feature_conf = deepcopy(extract_features.confs["superpoint_aachen"])
    feature_conf["model"]["device"] = device
    matcher_conf = deepcopy(match_features.confs["superglue"])
    matcher_conf["model"]["device"] = device
    global_conf = deepcopy(extract_features.confs["netvlad"])
    global_conf["model"]["device"] = device

    cameras = _load_sensors(sensors_path)
    image_entries = _load_image_entries(image_list_path)
    if image_entries:
        image_root, resolved = _resolve_image_root(session_dir, image_entries, mapping_dir_default)
        mapping_names = [name for _, name in resolved]
        LOGGER.info("Loaded %d mapping images from images.txt.", len(mapping_names))
    else:
        image_root = mapping_dir_default if mapping_dir_default.exists() else session_dir / "raw_data"
        mapping_names = _list_images(image_root)
        LOGGER.info("No images.txt found, using %d images from %s.", len(mapping_names), image_root)

    query_dir = args.query_dir if args.query_dir.is_absolute() else REPO_ROOT / args.query_dir
    query_root = query_dir.parent
    query_names = sorted(
        [p.relative_to(query_root).as_posix() for p in query_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    LOGGER.info("Queries: %d from %s", len(query_names), query_dir)

    shared_camera = None
    if cameras:
        uniq = {
            (cam["model"], cam["width"], cam["height"], tuple(cam["params"])) for cam in cameras.values()
        }
        if len(uniq) == 1:
            shared_camera = list(cameras.values())[0]
            LOGGER.info("Using shared intrinsics from sensors.txt: %s", shared_camera)
        else:
            LOGGER.info("Multiple distinct sensors detected; per-image intrinsics will be kept.")

    # Mapping features
    if not _log_skip(map_features_path, "mapping feature extraction"):
        extract_features.main(
            feature_conf,
            image_root,
            image_list=mapping_names,
            feature_path=map_features_path,
        )

    if not _log_skip(map_global_path, "mapping global descriptor extraction"):
        extract_features.main(
            global_conf,
            image_root,
            image_list=mapping_names,
            feature_path=map_global_path,
        )

    seq_pairs = _build_sequential_pairs(mapping_names, args.seq_overlap)
    _save_pairs(seq_pairs, map_pairs_seq_path)
    if args.top_k_map > 0 and not _log_skip(map_pairs_retr_path, "retrieval mapping pairs"):
        pairs_from_retrieval.main(
            descriptors=map_global_path,
            output=map_pairs_retr_path,
            num_matched=args.top_k_map,
            query_list=mapping_names,
            db_list=mapping_names,
        )
    retr_pairs = _read_pairs(map_pairs_retr_path)
    all_pairs = seq_pairs | retr_pairs
    _save_pairs(all_pairs, map_pairs_path)
    LOGGER.info("Mapping pairs: %d (seq=%d, retr=%d)", len(all_pairs), len(seq_pairs), len(retr_pairs))

    if not _log_skip(map_matches_path, "mapping matching"):
        match_features.main(
            matcher_conf,
            pairs=map_pairs_path,
            features=map_features_path,
            matches=map_matches_path,
        )

    camera_mode = pycolmap.CameraMode.AUTO
    image_options: Dict[str, object] = {}
    if shared_camera:
        camera_mode = pycolmap.CameraMode.SINGLE
        image_options = {
            "camera_model": shared_camera["model"],
            # pycolmap expects a comma-separated string for camera_params.
            "camera_params": ",".join(str(p) for p in shared_camera["params"]),
        }
    elif cameras:
        camera_mode = pycolmap.CameraMode.PER_IMAGE

    if (paths["sfm"] / "images.bin").exists():
        LOGGER.info("Reusing existing SfM at %s", paths["sfm"])
        model = pycolmap.Reconstruction(paths["sfm"])
    else:
        model = reconstruction.main(
            sfm_dir=paths["sfm"],
            image_dir=image_root,
            pairs=map_pairs_path,
            features=map_features_path,
            matches=map_matches_path,
            camera_mode=camera_mode,
            image_list=mapping_names,
            image_options=image_options,
            mapper_options={"ba_refine_principal_point": False},
        )
        if model is None:
            raise RuntimeError("Reconstruction failed")
    LOGGER.info("SfM summary: %s", model.summary())

    if not _log_skip(query_global_path, "query global descriptor extraction"):
        extract_features.main(
            global_conf,
            query_root,
            image_list=query_names,
            feature_path=query_global_path,
        )

    if not _log_skip(loc_pairs_path, "query retrieval"):
        pairs_from_retrieval.main(
            descriptors=query_global_path,
            db_descriptors=map_global_path,
            output=loc_pairs_path,
            num_matched=args.top_k_retrieval,
            db_list=mapping_names,
        )
    retrieval = parse_retrieval(loc_pairs_path)

    if not _log_skip(query_features_path, "query feature extraction"):
        extract_features.main(
            feature_conf,
            query_root,
            image_list=query_names,
            feature_path=query_features_path,
        )

    if not _log_skip(loc_matches_path, "query-map matching"):
        match_features.main(
            matcher_conf,
            pairs=loc_pairs_path,
            features=query_features_path,
            features_ref=map_features_path,
            matches=loc_matches_path,
        )

    db_name_to_id = {img.name: i for i, img in model.images.items()}
    localizer = QueryLocalizer(
        model,
        {
            "estimation": {"ransac": {"max_error": args.ransac}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        },
    )
    fallback_cam = shared_camera
    results = {}
    loc_logs: Dict[str, dict] = {}
    query_cameras: Dict[str, pycolmap.Camera] = {}
    for qname, refs in retrieval.items():
        db_ids = [db_name_to_id[n] for n in refs if n in db_name_to_id]
        if not db_ids:
            continue
        camera = _infer_camera(query_root / qname, fallback_cam)
        ret, log = pose_from_cluster(
            localizer,
            qname,
            camera,
            db_ids,
            query_features_path,
            loc_matches_path,
        )
        log["covisibility_clustering"] = False
        loc_logs[qname] = log
        if ret is not None:
            results[qname] = ret["cam_from_world"]
            query_cameras[qname] = camera
    LOGGER.info("Localized %d/%d queries.", len(results), len(query_names))

    write_poses(results, loc_results_path, prepend_camera_name=False)
    _write_loc_logs(loc_results_path, loc_logs, retrieval, query_features_path, loc_matches_path)

    # Print poses to stdout
    if results:
        print("Predicted poses (cam_from_world) for localized queries (wxyz | tx ty tz):")
        for qname, pose in results.items():
            qvec = pose.rotation.quat[[3, 0, 1, 2]]
            tvec = pose.translation
            print(
                f"  {qname}: q=({qvec[0]:.6f}, {qvec[1]:.6f}, {qvec[2]:.6f}, {qvec[3]:.6f}), "
                f"t=({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})"
            )

    # Save a lightweight 3D plot with only query poses (no mapping cameras)
    if results and args.save_query_plot:
        plot_path = (
            args.query_plot_path
            if args.query_plot_path is not None
            else paths["loc"] / "query_poses.html"
        )
        plot_path = plot_path if plot_path.is_absolute() else REPO_ROOT / plot_path
        fig = viz_3d.init_figure(height=600)
        viz_3d.plot_reconstruction(fig, model, max_reproj_error=10.0, cs=1.2, cameras=False)
        for qname, pose in results.items():
            cam = query_cameras.get(qname)
            if cam is None:
                continue
            viz_3d.plot_camera_colmap(fig, pose, cam, name=f"query: {qname}")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(plot_path)
        LOGGER.info("Saved query-only pose plot to %s", plot_path)


if __name__ == "__main__":
    run()
