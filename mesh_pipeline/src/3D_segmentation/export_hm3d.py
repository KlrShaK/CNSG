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
OUTPUT_SCN = paths.hm3d_scn
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

    # 3. Assign Colors to Mesh (ensure ColorVisuals are used)
    try:
        # Prefer explicit ColorVisuals to guarantee vertex color accessors
        from trimesh.visual import ColorVisuals

        mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    except Exception:
        # Fallback: set the vertex_colors attribute directly
        mesh.visual.vertex_colors = vertex_colors

    # 4. Export GLB (force file_type to 'glb' so exporter uses correct writer)
    print(f"Exporting {OUTPUT_GLB}...")
    try:
        mesh.export(str(OUTPUT_GLB), file_type='glb')
    except Exception:
        # older trimesh versions may ignore file_type, try default export
        mesh.export(str(OUTPUT_GLB))

    # Quick diagnostic (optional): check whether exported GLB contains COLOR_0
    try:
        from pygltflib import GLTF2
        g = GLTF2().load(str(OUTPUT_GLB))
        prim_attrs = {}
        if g.meshes and len(g.meshes) > 0 and g.meshes[0].primitives:
            prim_attrs = g.meshes[0].primitives[0].attributes
        if not prim_attrs or 'COLOR_0' not in prim_attrs:
            print('Warning: exported GLB does not contain COLOR_0 attribute (vertex colors).')
        else:
            print('DEBUG: exported GLB contains COLOR_0 vertex color attribute.')
    except Exception:
        # pygltflib not installed or parse error; skip diagnostic
        pass

    # Post-process exported GLB to ensure viewers use vertex colors
    # - remove any baseColorTexture references from materials (so textures don't override vertex colors)
    # - ensure there's at least one material and assign a fallback material to primitives missing `material`
    try:
        import struct
        glb_path = str(OUTPUT_GLB)
        with open(glb_path, 'rb') as f:
            header = f.read(12)
            if len(header) >= 12:
                # parse GLB header to extract version
                magic, version, length = struct.unpack('<4sII', header)
                # read first chunk header
                chunk_header = f.read(8)
                if len(chunk_header) >= 8:
                    json_len, json_type = struct.unpack('<I4s', chunk_header)
                    json_bytes = f.read(json_len)
                    rest = f.read()
                    try:
                        js = json.loads(json_bytes.decode('utf-8'))
                    except Exception:
                        js = json.loads(json_bytes.decode('utf-8', 'ignore'))

                    meshes = js.get('meshes', [])
                    materials = js.get('materials')
                    # sanitize existing materials: drop baseColorTexture if present
                    if materials:
                        for m in materials:
                            pbr = m.get('pbrMetallicRoughness')
                            if isinstance(pbr, dict) and 'baseColorTexture' in pbr:
                                del pbr['baseColorTexture']
                                if 'baseColorFactor' not in pbr:
                                    pbr['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]

                    # ensure materials array exists and add fallback material
                    if 'materials' not in js or not js.get('materials'):
                        js['materials'] = []
                    # create fallback material
                    fallback = {
                        'name': 'vertexcolor_fallback',
                        'pbrMetallicRoughness': {
                            'baseColorFactor': [1.0, 1.0, 1.0, 1.0],
                            'metallicFactor': 0.0
                        }
                    }
                    js['materials'].append(fallback)
                    fallback_index = len(js['materials']) - 1

                    # assign fallback material to any primitive missing 'material'
                    for mesh in meshes:
                        for prim in mesh.get('primitives', []):
                            if 'material' not in prim:
                                prim['material'] = fallback_index

                    # write back modified GLB (keep binary chunk(s) unchanged)
                    new_json = json.dumps(js, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
                    pad_len = (4 - (len(new_json) % 4)) % 4
                    new_json += b' ' * pad_len
                    total_len = 12 + 8 + len(new_json) + len(rest)
                    with open(glb_path, 'wb') as out_f:
                        out_f.write(struct.pack('<4sII', b'glTF', version, total_len))
                        out_f.write(struct.pack('<I4s', len(new_json), b'JSON'))
                        out_f.write(new_json)
                        out_f.write(rest)
                    print('Post-processed GLB: ensured fallback material and removed baseColorTexture refs')
    except Exception as e:
        print('Warning: GLB post-process failed:', e)

    # 5. Export TXT
    print(f"Exporting {OUTPUT_TXT}...")
    with open(OUTPUT_TXT, 'w') as f:
        f.write("\n".join(txt_lines))
    
    print(f"Exporting {OUTPUT_SCN}...")
    with open(OUTPUT_SCN, 'w') as f:
        f.write("\n".join(txt_lines))

    print("Success! Pipeline Complete.")
    print("Files ready for Habitat:")
    print(f"  - {OUTPUT_GLB}")
    print(f"  - {OUTPUT_TXT}")
    print(f"  - {OUTPUT_SCN}")

if __name__ == "__main__":
    main()
