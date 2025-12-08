# CNSG Mesh Pipeline

End-to-end mesh and semantic processing for NavVis LaMAR sessions: Poisson meshing, depth rendering, semantic segmentation, label fusion/cleanup, and Habitat export. NavVis data comes from the LaMAR benchmark; Poisson meshing is adapted from LaMAR (see `third_party`). OpenYOLO3D is vendored for future integration.

## Environment
Use conda (Python 3.10.6 recommended):
```bash
conda create -n CNSG-meshing python=3.10.6
conda activate CNSG-meshing
pip install -r requirements.txt
```
CUDA is recommended for faster segmentation; install the appropriate PyTorch build if needed.

## Data
Download NavVis LaMAR session and processed add-ons:
```bash
./scripts/download_data.sh
```
This fetches `navvis_2022-02-06_12.55.11` into `data/` and auto-extracts provided `depth_maps` and `semantic_masks` from the Google Drive bundle.

## Pipelines
Run commands from the repo root.

### Mesh generation
**NOTE**: If you already ran `./scripts/download_data.sh`, a mesh and voxelized mesh are provided; **you can skip this part unless you want to tweak parameters and regenerate**.
```bash
./scripts/generate_mesh.sh
```

The script prompts for:
- Poisson mesh only
- Poisson + voxelized mesh
- Full pipeline (Poisson + voxelize + depth maps)

Underlying components (`src/mesh_generation`):
- `run_poisson_meshing.py` — build Poisson mesh from point cloud.
- `habitat_mesh_reduce.py` — voxel-based simplification for Habitat.
- `generate_depth.py` — raycast depth maps using trajectories.

### 3D segmentation
```bash
./scripts/run_segmentation_pipeline.sh
```
Behavior:
- Verifies `depth_maps`; offers to run `generate_depth.py` if missing.
- Executes in order: `run_segmentation.py` → `fuse_labels.py` → `clean_mesh.py` → `transfer_labels_to_trimesh.py` → `export_hm3d.py`.

Underlying components (`src/3D_segmentation`):
- `run_segmentation.py` — Mask2Former semantic inference on NavVis images.
- `fuse_labels.py` — project per-frame masks onto the mesh via depth + poses.
- `clean_mesh.py` — DBSCAN + KNN smoothing of vertex labels.
- `transfer_labels_to_trimesh.py` — align labels to trimesh vertex order.
- `export_hm3d.py` — Habitat GLB + TXT semantic export.

## Configuration
`config/paths.yml` centralizes dataset and asset paths (session name, point cloud, images, depth, meshes, exports) and is loaded via `src/utils/config_utils.py`. Adjust entries if your data lives elsewhere.

## Third-Party
- LaMAR benchmark assets and adapted Poisson tools in `third_party/`.
- OpenYOLO3D (not yet used) is vendored under `third_party/OpenYOLO3D`; integration planned.

## Contact
Questions: rzendehdel@ethz.ch

## Citations
If you use this pipeline or data, please cite:
```
@inproceedings{sarlin2022lamar,
  author    = {Paul-Edouard Sarlin and
               Mihai Dusmanu and
               Johannes L. Schönberger and
               Pablo Speciale and
               Lukas Gruber and
               Viktor Larsson and
               Ondrej Miksik and
               Marc Pollefeys},
  title     = {{LaMAR: Benchmarking Localization and Mapping for Augmented Reality}},
  booktitle = {ECCV},
  year      = {2022},
}

@inproceedings{
boudjoghra2025openyolo,
title={Open-{YOLO} 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation},
author={Mohamed El Amine Boudjoghra and Angela Dai and Jean Lahoud and Hisham Cholakkal and Rao Muhammad Anwer and Salman Khan and Fahad Shahbaz Khan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```
