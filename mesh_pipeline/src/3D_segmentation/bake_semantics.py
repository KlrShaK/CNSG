"""
GPU-Accelerated Semantic Baking using PyRender.

This script replaces the slow KD-Tree approach with hardware rasterization.
It "unwraps" the mesh and renders the vertex colors directly into a texture.

Workflow:
1. Unwrap Mesh (CPU - xatlas).
2. Create a scene where the mesh is flat in UV space (Z=0).
3. Render the scene with an orthographic camera capturing the [0,1] UV square.
4. The resulting image IS your semantic texture.
"""

import numpy as np
import trimesh
import xatlas
import pyrender
import sys
import os
from pathlib import Path
from PIL import Image

# --- Configuration ---
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
TEXTURE_FILENAME = "semantic_texture.png"
TEXTURE_SIZE = 4096  # GPU can handle 4K easily
# ---------------------

def id_to_color(obj_id: int) -> np.ndarray:
    np.random.seed(obj_id)
    return np.random.randint(0, 255, size=3)

def bake_texture_gpu(mesh, new_colors, width=4096, height=4096):
    """
    Bakes vertex colors to texture using OpenGL rasterization.
    It effectively "draws" the mesh flat using its UV coordinates as XY positions.
    """
    print("   Setting up GPU rasterizer...")
    
    # 1. Create a flat mesh where XY = UV coordinates
    # We ignore the original 3D positions and use UVs as geometry.
    # UVs are 0..1, we map them to world coordinates for the camera.
    flat_vertices = np.zeros((len(mesh.vertices), 3))
    flat_vertices[:, 0] = mesh.visual.uv[:, 0]
    flat_vertices[:, 1] = mesh.visual.uv[:, 1]
    flat_vertices[:, 2] = 0.0  # Flat on Z plane

    # 2. Create PyRender Mesh
    # We assign the vertex colors we want to bake
    m = pyrender.Mesh.from_trimesh(trimesh.Trimesh(
        vertices=flat_vertices,
        faces=mesh.faces,
        vertex_colors=new_colors
    ), smooth=False)

    # 3. Setup Scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(m)

    # 4. Setup Orthographic Camera
    # We want to capture exactly the [0,1] x [0,1] square.
    # Orthographic scale is half-size, so for 1.0 width we use xmag=0.5
    camera = pyrender.OrthographicCamera(xmag=0.5, ymag=0.5)
    
    # Position camera centered at (0.5, 0.5) looking down -Z
    camera_pose = np.eye(4)
    camera_pose[0, 3] = 0.5  # X center
    camera_pose[1, 3] = 0.5  # Y center
    camera_pose[2, 3] = 1.0  # Z position (above plane)
    scene.add(camera, pose=camera_pose)

    # 5. Render
    print(f"   Rendering {width}x{height} texture on GPU...")
    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT) # FLAT = no lighting, pure color
    
    # 6. Cleanup
    r.delete()
    
    # PyRender outputs flipped Y relative to standard UVs usually, but since
    # we mapped UV y=0 to world y=0, and camera y-axis is up, it should match.
    # However, image origins are top-left. We often need a flip.
    # Let's verify: World (0,0) is bottom-left. Image (0,0) is top-left.
    # So we flip.
    return Image.fromarray(np.flipud(color))

def main():
    print(f"Loading mesh: {INPUT_MESH}...")
    mesh = trimesh.load(str(INPUT_MESH))
    
    # Load labels
    label_path = TRIMESH_LABELS if TRIMESH_LABELS.exists() else FALLBACK_LABELS
    print(f"Loading labels: {label_path}...")
    labels = np.load(label_path)
    
    # --- Prepare Colors & Manifest ---
    print("Preparing semantic colors...")
    vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255 
    unique_ids = np.unique(labels)
    
    import json
    txt_lines = ["HM3D Semantic Annotations"]
    with open(CONFIG_JSON, 'r') as f:
        config_data = json.load(f)
    raw_id2label = config_data.get("id2label", {})
    id2label = {int(k): v for k, v in raw_id2label.items()}

    for uid in unique_ids:
        color = id_to_color(uid)
        mask = (labels == uid)
        vertex_colors[mask, 0:3] = color
        
        # Manifest
        raw_name = id2label.get(uid, "unknown")
        clean_name = raw_name.split(',')[0].strip()
        hex_code = "{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])
        line = f'{uid},{hex_code},"{clean_name}",16'
        txt_lines.append(line)

    # --- Unwrap ---
    print("Unwrapping mesh with xatlas...")
    v_mapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    
    # Remap colors
    new_colors = vertex_colors[v_mapping]
    
    # Create UV mesh
    uv_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[v_mapping],
        faces=indices,
        visual=trimesh.visual.TextureVisuals(uv=uvs),
        process=False
    )

    # --- GPU Bake ---
    texture_image = bake_texture_gpu(uv_mesh, new_colors, width=TEXTURE_SIZE, height=TEXTURE_SIZE)
    
    # Save Texture
    tex_path = OUTPUT_GLB.parent / TEXTURE_FILENAME
    texture_image.save(tex_path)
    print(f"Texture saved: {tex_path}")

    # --- Export GLB ---
    print("Exporting GLB...")
    material = trimesh.visual.material.PBRMaterial(
        name='SemanticMaterial',
        baseColorTexture=Image.open(tex_path),
        doubleSided=True,  # Crucial for visibility
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    uv_mesh.visual.material = material
    uv_mesh.export(str(OUTPUT_GLB), file_type='glb')

    # --- Export Manifest ---
    print("Exporting Manifest...")
    with open(OUTPUT_TXT, 'w') as f:
        f.write("\n".join(txt_lines))

    print("Done. GPU Bake complete.")

if __name__ == "__main__":
    main()