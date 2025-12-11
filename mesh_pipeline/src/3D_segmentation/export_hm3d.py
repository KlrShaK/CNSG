"""
Export a cleaned semantic mesh to Habitat-compatible artifacts (GLB + TXT).

Key Features:
1. "Explodes" the mesh by label to prevent GPU color interpolation (fixes rainbow banding).
2. Patches the GLB to force "doubleSided" rendering (fixes invisible walls).
3. Exports standard HM3D vertex-color format.

Usage:
  python src/3D_segmentation/export_hm3d.py
"""

from typing import Dict, List
import numpy as np
import trimesh
import json
import struct
import sys
from pathlib import Path

# --- Configuration Setup ---
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.config_utils import load_paths

paths = load_paths()
INPUT_MESH = paths.mesh_path
CONFIG_JSON = paths.segmentation_config
TRIMESH_LABELS = paths.trimesh_labels
FALLBACK_LABELS = paths.fused_ids_clean

OUTPUT_GLB = paths.hm3d_glb
OUTPUT_TXT = paths.hm3d_txt
OUTPUT_SCN = paths.hm3d_scn
# ---------------------------

def id_to_color(obj_id: int) -> np.ndarray:
    """Generate a deterministic RGB color [0-255] for an ID."""
    np.random.seed(obj_id)
    return np.random.randint(0, 255, size=3)

def explode_mesh_by_label(mesh, labels):
    """
    Physically separates the mesh into sub-meshes based on labels.
    This prevents the GPU from interpolating colors between different objects,
    fixing the 'rainbow banding' artifact at boundaries.
    """
    print("Exploding mesh by semantic ID to prevent color interpolation...")
    
    unique_ids = np.unique(labels)
    sub_meshes = []
    
    # We will rebuild the mesh as a concatenation of separate objects
    # This ensures vertices are duplicated at boundaries, breaking the smoothing.
    
    for uid in unique_ids:
        # Create a boolean mask for faces that belong to this label
        # We assume a face belongs to a label if its first vertex does.
        # (Since we cleaned the mesh, faces should be mostly uniform).
        
        # Get labels for all 3 vertices of every face
        # We assume vertices and labels are aligned.
        # Note: Trimesh doesn't have "face labels" by default, we derive them from vertices.
        
        # Strategy: A face belongs to label L if vertex 0 of that face has label L.
        # This is fast and generally accurate for dense meshes.
        face_mask = (labels[mesh.faces[:, 0]] == uid)
        
        if not np.any(face_mask):
            continue
            
        # Create a submesh of just these faces
        # process=False prevents merging vertices back together
        sub = mesh.submesh([face_mask], append=True)
        
        # Now we paint this entire submesh with the solid color
        # This is much safer than per-vertex painting on a shared mesh
        color = id_to_color(uid)
        
        # Create color array (N, 4)
        # We use 'visual.vertex_colors' on the submesh
        vertex_colors = np.zeros((len(sub.vertices), 4), dtype=np.uint8)
        vertex_colors[:, 0:3] = color
        vertex_colors[:, 3] = 255 # Alpha
        
        sub.visual.vertex_colors = vertex_colors
        
        # Optional: Store the ID in metadata
        sub.metadata['semantic_id'] = uid
        
        sub_meshes.append(sub)
        
    print(f"Split into {len(sub_meshes)} separate sub-objects.")
    
    # Combine them back into one "Scene" mesh (vertices are now duplicated at seams)
    if not sub_meshes:
        print("Warning: No submeshes created. Returning original mesh.")
        return mesh

    combined_mesh = trimesh.util.concatenate(sub_meshes)
    return combined_mesh

def patch_glb_doublesided(glb_path: str):
    """
    Directly modifies the binary GLB file to ensure all materials 
    have "doubleSided": true. This fixes visibility issues (invisible walls) in Habitat.
    """
    print(f"Patching {glb_path} to force double-sided rendering...")
    
    with open(glb_path, 'rb') as f:
        data = f.read()

    # Parse GLB Header
    if len(data) < 12: 
        print("Error: File too short.")
        return
    magic, version, length = struct.unpack('<4sII', data[:12])
    if magic != b'glTF': 
        print("Error: Not a GLB file.")
        return

    # Parse Chunk 0 (JSON)
    if len(data) < 20: 
        print("Error: Incomplete header.")
        return
    chunk_len, chunk_type = struct.unpack('<I4s', data[12:20])
    if chunk_type != b'JSON': 
        print("Error: First chunk not JSON.")
        return

    # Extract JSON
    json_bytes = data[20:20+chunk_len]
    try:
        json_str = json_bytes.decode('utf-8')
        gltf_data = json.loads(json_str)
        
        # --- THE PATCH ---
        if "materials" in gltf_data:
            for mat in gltf_data["materials"]:
                mat["doubleSided"] = True
                # Ensure neutral base color so vertex colors show through
                if "pbrMetallicRoughness" in mat:
                     mat["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        else:
            gltf_data["materials"] = [{"name": "default", "doubleSided": True}]
        # -----------------

        # Re-pack
        new_json_bytes = json.dumps(gltf_data).encode('utf-8')
        pad = (4 - (len(new_json_bytes) % 4)) % 4
        new_json_bytes += b' ' * pad
        
        # Original binary chunk starts after the old JSON chunk
        binary_start_idx = 20 + chunk_len
        binary_data = data[binary_start_idx:]
        
        # New total length
        total_len = 12 + 8 + len(new_json_bytes) + len(binary_data)
        
        with open(glb_path, 'wb') as f:
            f.write(struct.pack('<4sII', magic, version, total_len))
            f.write(struct.pack('<I4s', len(new_json_bytes), chunk_type))
            f.write(new_json_bytes)
            f.write(binary_data)
            
        print("Success: GLB patched for double-sided rendering.")
        
    except Exception as e:
        print(f"Failed to patch GLB: {e}")

def main() -> None:
    print(f"Loading mesh: {INPUT_MESH}...")
    try:
        # process=False is crucial to keep vertex order aligned with labels
        mesh = trimesh.load(str(INPUT_MESH), process=False) 
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    label_path = TRIMESH_LABELS if TRIMESH_LABELS.exists() else FALLBACK_LABELS
    print(f"Loading labels: {label_path}...")
    try:
        labels = np.load(label_path)
    except FileNotFoundError:
        print("Error: Label file not found.")
        return

    if len(labels) != len(mesh.vertices):
        print(f"CRITICAL: Mismatch! Verts={len(mesh.vertices)}, Labels={len(labels)}.")
        return

    print(f"Loading config: {CONFIG_JSON}...")
    with open(CONFIG_JSON, 'r') as f:
        config_data = json.load(f)
    
    raw_id2label = config_data.get("id2label", {})
    id2label: Dict[int, str] = {int(k): v for k, v in raw_id2label.items()}

    # --- STEP 1: EXPLODE AND COLOR THE MESH ---
    # This splits the mesh into separate pieces for each ID
    # Colors are assigned per-piece inside this function
    colored_mesh = explode_mesh_by_label(mesh, labels)

    # --- STEP 2: GENERATE MANIFEST ---
    print("Generating HM3D semantic manifest...")
    txt_lines = ["HM3D Semantic Annotations"]
    unique_ids = np.unique(labels)
    
    for uid in unique_ids:
        # We need to regenerate the color here just for the text file
        # The mesh already has the color baked in from the explode step
        color = id_to_color(uid) 
        
        raw_name = id2label.get(uid, "unknown")
        clean_name = raw_name.split(',')[0].strip()
        
        hex_code = "{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])
        line = f'{uid},{hex_code},"{clean_name}",16'
        txt_lines.append(line)

    # --- STEP 3: EXPORT GLB ---
    print(f"Exporting {OUTPUT_GLB}...")
    try:
        # Force GLB format. Trimesh handles the vertex colors we assigned in `explode_mesh_by_label`
        colored_mesh.export(str(OUTPUT_GLB), file_type='glb')
    except Exception as e:
        print(f"Export failed: {e}")
        return

    # --- STEP 4: PATCH GLB (Fix Visibility) ---
    patch_glb_doublesided(str(OUTPUT_GLB))

    # --- STEP 5: EXPORT MANIFEST ---
    print(f"Exporting {OUTPUT_TXT}...")
    with open(OUTPUT_TXT, 'w') as f:
        f.write("\n".join(txt_lines))
    
    # Also save .scn for compatibility if needed
    with open(OUTPUT_SCN, 'w') as f:
        f.write("\n".join(txt_lines))

    print("Success! Pipeline Complete.")
    print("Files ready for Habitat:")
    print(f"  - {OUTPUT_GLB}")
    print(f"  - {OUTPUT_TXT}")

if __name__ == "__main__":
    main()