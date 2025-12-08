"""
Export a cleaned semantic mesh to Habitat-compatible artifacts (GLB + TXT).

Workflow:
1) Load the target mesh (trimesh) and semantic labels (.npy).
2) Map class IDs to deterministic colors and human-readable names.
3) Paint vertex colors and export a GLB plus a Habitat semantic TXT manifest.

All paths are resolved from config/paths.yml via config_utils.load_paths().
"""

from typing import Dict
import numpy as np
import trimesh
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.config_utils import load_paths

# ================= CONFIGURATION =================
# Inputs
paths = load_paths()
INPUT_MESH = paths.mesh_path
CONFIG_JSON = paths.segmentation_config     # Your model config

# Label candidates: prefer the trimesh-aligned labels, fallback to cleaned fusion labels
TRIMESH_LABELS = paths.trimesh_labels
FALLBACK_LABELS = paths.fused_ids_clean

# Outputs
OUTPUT_GLB = paths.hm3d_glb
OUTPUT_TXT = paths.hm3d_txt
# =================================================

def id_to_color(obj_id: int) -> np.ndarray:
    """
    Generate a deterministic RGB color for a class/instance ID.

    Habitat expects unique colors per instance; we hash via a seeded RNG.

    Args:
        obj_id: Semantic class or instance identifier.

    Returns:
        np.ndarray of shape (3,) with uint8-compatible RGB values in [0, 255].
    """
    np.random.seed(obj_id)
    # Generate 3 random integers 0-255
    return np.random.randint(0, 255, size=3)

def main() -> None:
    """
    Load mesh + labels, colorize vertices, and export Habitat GLB/TXT artifacts.
    """
    print(f"Loading mesh: {INPUT_MESH}...")
    try:
        # Trimesh handles GLB export better than Open3D for this specific task
        mesh = trimesh.load(str(INPUT_MESH))
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # Prefer labels aligned to trimesh vertex ordering; otherwise fall back to cleaned fusion labels
    label_path = TRIMESH_LABELS if TRIMESH_LABELS.exists() else FALLBACK_LABELS
    print(f"Loading labels: {label_path}...")
    try:
        labels = np.load(label_path)
    except FileNotFoundError:
        print("Error: Label file not found. Run clean_mesh.py first.")
        return

    # Validation
    if len(labels) != len(mesh.vertices):
        print(f"CRITICAL ERROR: Mismatch! Mesh has {len(mesh.vertices)} vertices, Labels have {len(labels)}.")
        return

    # Load Class Names
    print(f"Loading config: {CONFIG_JSON}...")
    with open(CONFIG_JSON, 'r') as f:
        config_data = json.load(f)
    
    # Handle config format (id2label might be string keys)
    raw_id2label = config_data.get("id2label", {})
    # Convert keys to integers for safe lookup
    id2label: Dict[int, str] = {int(k): v for k, v in raw_id2label.items()}

    print("Generating HM3D semantic data (colors + TXT manifest)...")
    
    # 1. Prepare Vertex Color Array (N, 4) -> RGBA
    # Default alpha = 255
    vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255 
    
    # 2. Prepare TXT content
    # Header required by Habitat
    txt_lines = ["HM3D Semantic Annotations"]
    
    # Get all unique classes found in the mesh
    unique_ids = np.unique(labels)
    print(f"Found {len(unique_ids)} unique classes in the mesh.")
    
    for uid in unique_ids:
        # A. Generate Unique Color for this ID
        color = id_to_color(uid) # [R, G, B]
        
        # B. Paint Vertices
        # Find all vertices belonging to this class ID
        mask = (labels == uid)
        vertex_colors[mask, 0:3] = color
        
        # C. Lookup Name
        # Get raw name (e.g. "chair, seat") -> take first part "chair"
        raw_name = id2label.get(uid, "unknown")
        clean_name = raw_name.split(',')[0].strip()
        
        # D. Format Line for TXT
        # ID,HEX,"Name",16
        # Note: HM3D usually indexes instances 1..N. 
        # We are mapping Semantic Class ID directly to Instance ID for simplicity.
        # This means all "chairs" will be one single "chair instance" visually in the semantic view.
        # If you want individual instances (Chair #1, Chair #2), you need instance segmentation (Mask3D).
        # Since we used Semantic Segmentation (Mask2Former), grouping by class is the correct behavior.
        
        hex_code = "{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])
        line = f'{uid},{hex_code},"{clean_name}",16'
        txt_lines.append(line)

    # 3. Assign Colors to Mesh
    mesh.visual.vertex_colors = vertex_colors

    # 4. Export GLB
    print(f"Exporting {OUTPUT_GLB}...")
    # Trimesh automatically exports vertex colors as COLOR_0 accessor
    mesh.export(str(OUTPUT_GLB))

    # 5. Export TXT
    print(f"Exporting {OUTPUT_TXT}...")
    with open(OUTPUT_TXT, 'w') as f:
        f.write("\n".join(txt_lines))

    print("Success! Pipeline Complete.")
    print("Files ready for Habitat:")
    print(f"  - {OUTPUT_GLB}")
    print(f"  - {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
