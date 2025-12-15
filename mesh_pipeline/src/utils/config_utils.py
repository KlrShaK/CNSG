from dataclasses import dataclass
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to read the path configuration. "
        "Install with `pip install pyyaml`."
    ) from exc

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/paths.yml"


def _resolve_path(value: str, base: Path) -> Path:
    """Resolve a possibly relative path against a base directory."""
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass
class SegmentationPaths:
    root: Path
    data_root: Path
    outputs_root: Path
    session_dir: Path
    raw_images_dir: Path
    raw_pointcloud: Path
    depth_maps_dir: Path
    semantic_masks_dir: Path
    trajectories_file: Path
    images_file: Path
    sensors_file: Path
    global_alignment_file: Path
    poisson_mesh: Path
    mesh_path: Path
    fused_mesh: Path
    fused_ids: Path
    fused_mesh_clean: Path
    fused_ids_clean: Path
    trimesh_labels: Path
    hm3d_glb: Path
    hm3d_txt: Path
    hm3d_scn: Path
    segmentation_config: Path


def load_paths(config_path: Path = CONFIG_PATH) -> SegmentationPaths:
    """Load shared segmentation paths and resolve them relative to the repo root."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    paths_cfg = config.get("paths")
    if not paths_cfg:
        raise ValueError("config/paths.yml is missing the 'paths' section.")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = _resolve_path(paths_cfg.get("data_root", "data"), repo_root)
    outputs_root = _resolve_path(paths_cfg.get("outputs_root", "outputs"), repo_root)

    session_name = paths_cfg.get("session_name")
    if not session_name:
        raise ValueError("config/paths.yml must define 'session_name' under 'paths'.")
    session_dir = data_root / session_name

    raw_images_dir = _resolve_path(
        paths_cfg.get("raw_images_subdir", "raw_data/images_undistr_center"),
        session_dir,
    )
    raw_pointcloud = _resolve_path(
        paths_cfg.get("raw_pointcloud_filename", "raw_data/pointcloud.ply"),
        session_dir,
    )
    depth_maps_dir = _resolve_path(
        paths_cfg.get("depth_maps_subdir", "depth_maps"),
        session_dir,
    )
    semantic_masks_dir = _resolve_path(
        paths_cfg.get("semantic_masks_subdir", "semantic_masks"),
        session_dir,
    )

    trajectories_file = _resolve_path(
        paths_cfg.get("trajectories_file", "trajectories.txt"),
        session_dir,
    )
    images_file = _resolve_path(
        paths_cfg.get("images_file", "images.txt"),
        session_dir,
    )
    sensors_file = _resolve_path(
        paths_cfg.get("sensors_file", "sensors.txt"),
        session_dir,
    )
    global_alignment_file = _resolve_path(
        paths_cfg.get("global_alignment_file", "proc/alignment_global.txt"),
        session_dir,
    )

    poisson_mesh = _resolve_path(
        paths_cfg.get(
            "poisson_mesh_filename",
            paths_cfg.get("mesh_filename", "HGE_poisson.ply"),
        ),
        data_root,
    )
    mesh_path = _resolve_path(paths_cfg.get("mesh_filename", "HGE_cut.voxelized.ply"), data_root)
    fused_mesh = _resolve_path(paths_cfg.get("fused_mesh_filename", "HGE_semantic.ply"), data_root)
    fused_ids = _resolve_path(paths_cfg.get("fused_ids_filename", "HGE_semantic_ids.npy"), data_root)
    fused_mesh_clean = _resolve_path(
        paths_cfg.get("fused_mesh_clean_filename", "HGE_semantic_clean.ply"),
        data_root,
    )
    fused_ids_clean = _resolve_path(
        paths_cfg.get("fused_ids_clean_filename", "HGE_semantic_ids_clean.npy"),
        data_root,
    )
    trimesh_labels = _resolve_path(
        paths_cfg.get("trimesh_labels_filename", "HGE_cut_ids_trimesh.npy"),
        data_root,
    )

    hm3d_glb = _resolve_path(paths_cfg.get("hm3d_glb_filename", "HGE.semantic.glb"), data_root)
    hm3d_txt = _resolve_path(paths_cfg.get("hm3d_txt_filename", "HGE.semantic.txt"), data_root)
    hm3d_scn = _resolve_path(paths_cfg.get("hm3d_scn_filename", "HGE.semantic.scn"), data_root)
    segmentation_config = _resolve_path(
        paths_cfg.get("segmentation_config", "config/segmentation_config.json"),
        repo_root,
    )

    return SegmentationPaths(
        root=repo_root,
        data_root=data_root,
        outputs_root=outputs_root,
        session_dir=session_dir,
        raw_images_dir=raw_images_dir,
        raw_pointcloud=raw_pointcloud,
        depth_maps_dir=depth_maps_dir,
        semantic_masks_dir=semantic_masks_dir,
        trajectories_file=trajectories_file,
        images_file=images_file,
        sensors_file=sensors_file,
        global_alignment_file=global_alignment_file,
        poisson_mesh=poisson_mesh,
        mesh_path=mesh_path,
        fused_mesh=fused_mesh,
        fused_ids=fused_ids,
        fused_mesh_clean=fused_mesh_clean,
        fused_ids_clean=fused_ids_clean,
        trimesh_labels=trimesh_labels,
        hm3d_glb=hm3d_glb,
        hm3d_txt=hm3d_txt,
        hm3d_scn=hm3d_scn,
        segmentation_config=segmentation_config,
    )
