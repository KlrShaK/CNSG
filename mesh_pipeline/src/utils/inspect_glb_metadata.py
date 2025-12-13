"""
Inspect GLB file metadata to understand structure and attributes.

Usage:
    python src/3D_segmentation/inspect_glb_metadata.py <path_to_glb>

    Or to compare two files:
    python src/3D_segmentation/inspect_glb_metadata.py <your_glb> <hm3d_glb>
"""

import struct
import json
import sys
from pathlib import Path
from typing import Dict, Any


def inspect_glb(glb_path: Path) -> Dict[str, Any]:
    """Extract and return metadata from a GLB file."""

    print(f"\n{'='*70}")
    print(f"Inspecting: {glb_path.name}")
    print(f"{'='*70}")

    with open(glb_path, 'rb') as f:
        data = f.read()

    # Parse GLB header
    if len(data) < 12:
        print("ERROR: File too short")
        return {}

    magic, version, length = struct.unpack('<4sII', data[:12])
    print(f"\n📦 GLB Header:")
    print(f"   Magic: {magic}")
    print(f"   Version: {version}")
    print(f"   Total Length: {length:,} bytes")

    # Parse JSON chunk
    if len(data) < 20:
        print("ERROR: Incomplete GLB header")
        return {}

    json_len, json_type = struct.unpack('<I4s', data[12:20])
    print(f"\n📄 JSON Chunk:")
    print(f"   Type: {json_type}")
    print(f"   Length: {json_len:,} bytes")

    json_bytes = data[20:20 + json_len]
    gltf = json.loads(json_bytes.decode('utf-8'))

    # Parse binary chunk
    binary_start = 20 + json_len
    if len(data) > binary_start + 8:
        bin_len, bin_type = struct.unpack('<I4s', data[binary_start:binary_start + 8])
        print(f"\n📦 Binary Chunk:")
        print(f"   Type: {bin_type}")
        print(f"   Length: {bin_len:,} bytes")

    # Analyze GLTF JSON structure
    print(f"\n🔍 GLTF Structure:")
    print(f"   Asset: {gltf.get('asset', {})}")
    print(f"   Buffers: {len(gltf.get('buffers', []))}")
    print(f"   BufferViews: {len(gltf.get('bufferViews', []))}")
    print(f"   Accessors: {len(gltf.get('accessors', []))}")
    print(f"   Materials: {len(gltf.get('materials', []))}")
    print(f"   Meshes: {len(gltf.get('meshes', []))}")

    # Analyze meshes and primitives
    meshes = gltf.get('meshes', [])
    print(f"\n🎨 Meshes ({len(meshes)}):")
    for mesh_idx, mesh in enumerate(meshes):
        print(f"\n   Mesh {mesh_idx}: {mesh.get('name', 'unnamed')}")
        primitives = mesh.get('primitives', [])
        print(f"      Primitives: {len(primitives)}")

        for prim_idx, prim in enumerate(primitives):
            print(f"\n      Primitive {prim_idx}:")
            attrs = prim.get('attributes', {})
            print(f"         Attributes: {list(attrs.keys())}")

            # Detailed attribute info
            accessors = gltf.get('accessors', [])
            for attr_name, accessor_idx in attrs.items():
                if accessor_idx < len(accessors):
                    accessor = accessors[accessor_idx]
                    comp_type = accessor.get('componentType')
                    comp_type_name = {
                        5120: 'BYTE',
                        5121: 'UNSIGNED_BYTE (u8)',
                        5122: 'SHORT',
                        5123: 'UNSIGNED_SHORT (u16)',
                        5125: 'UNSIGNED_INT',
                        5126: 'FLOAT'
                    }.get(comp_type, f'Unknown ({comp_type})')

                    acc_type = accessor.get('type', 'UNKNOWN')
                    normalized = accessor.get('normalized', False)
                    count = accessor.get('count', 0)

                    print(f"         {attr_name}:")
                    print(f"            Accessor: {accessor_idx}")
                    print(f"            Type: {acc_type}")
                    print(f"            ComponentType: {comp_type_name}")
                    print(f"            Normalized: {normalized}")
                    print(f"            Count: {count:,}")

                    # For COLOR_0, this is critical
                    if attr_name == 'COLOR_0':
                        if comp_type == 5121:
                            format_str = "u8_norm (8-bit normalized)"
                        elif comp_type == 5123:
                            format_str = "u16_norm (16-bit normalized)"
                        else:
                            format_str = "UNKNOWN FORMAT!"
                        print(f"            ⚠️  COLOR FORMAT: {format_str}")

            # Material info
            mat_idx = prim.get('material')
            if mat_idx is not None:
                print(f"         Material Index: {mat_idx}")

    # Analyze materials
    materials = gltf.get('materials', [])
    if materials:
        print(f"\n🎨 Materials ({len(materials)}):")
        for mat_idx, mat in enumerate(materials):
            print(f"\n   Material {mat_idx}: {mat.get('name', 'unnamed')}")
            print(f"      doubleSided: {mat.get('doubleSided', False)}")

            pbr = mat.get('pbrMetallicRoughness', {})
            if pbr:
                print(f"      PBR:")
                print(f"         baseColorFactor: {pbr.get('baseColorFactor', 'not set')}")
                print(f"         metallicFactor: {pbr.get('metallicFactor', 'not set')}")
                print(f"         roughnessFactor: {pbr.get('roughnessFactor', 'not set')}")

                if 'baseColorTexture' in pbr:
                    print(f"         baseColorTexture: {pbr['baseColorTexture']}")
                else:
                    print(f"         baseColorTexture: None (good for vertex colors)")

    # Analyze accessors in detail
    accessors = gltf.get('accessors', [])
    print(f"\n📊 Accessors Summary ({len(accessors)}):")

    accessor_types = {}
    for accessor in accessors:
        comp_type = accessor.get('componentType')
        acc_type = accessor.get('type', 'UNKNOWN')
        key = f"{acc_type}:{comp_type}"
        accessor_types[key] = accessor_types.get(key, 0) + 1

    for key, count in accessor_types.items():
        acc_type, comp_type = key.split(':')
        comp_type = int(comp_type)
        comp_type_name = {
            5120: 'BYTE',
            5121: 'u8',
            5122: 'SHORT',
            5123: 'u16',
            5125: 'UINT',
            5126: 'FLOAT'
        }.get(comp_type, f'{comp_type}')
        print(f"   {acc_type}:{comp_type_name} → {count} accessors")

    return gltf


def compare_glbs(glb1_path: Path, glb2_path: Path):
    """Compare two GLB files."""

    print("\n" + "="*70)
    print("COMPARISON MODE")
    print("="*70)

    gltf1 = inspect_glb(glb1_path)
    gltf2 = inspect_glb(glb2_path)

    print("\n" + "="*70)
    print("🔍 KEY DIFFERENCES")
    print("="*70)

    # Compare COLOR_0 formats
    def find_color_format(gltf_data):
        """Find the COLOR_0 format in a GLTF structure."""
        meshes = gltf_data.get('meshes', [])
        accessors = gltf_data.get('accessors', [])

        for mesh in meshes:
            for prim in mesh.get('primitives', []):
                color_idx = prim.get('attributes', {}).get('COLOR_0')
                if color_idx is not None and color_idx < len(accessors):
                    accessor = accessors[color_idx]
                    comp_type = accessor.get('componentType')
                    normalized = accessor.get('normalized', False)

                    if comp_type == 5121:
                        return "u8_norm" if normalized else "u8"
                    elif comp_type == 5123:
                        return "u16_norm" if normalized else "u16"
                    else:
                        return f"unknown_{comp_type}"
        return "NO COLOR_0 FOUND"

    format1 = find_color_format(gltf1)
    format2 = find_color_format(gltf2)

    print(f"\n📊 COLOR_0 Format:")
    print(f"   {glb1_path.name}: {format1}")
    print(f"   {glb2_path.name}: {format2}")

    if format1 != format2:
        print(f"   ⚠️  MISMATCH! This will cause color matching issues!")
    else:
        print(f"   ✓ Same format")

    # Compare material counts
    mat_count1 = len(gltf1.get('materials', []))
    mat_count2 = len(gltf2.get('materials', []))

    print(f"\n🎨 Materials:")
    print(f"   {glb1_path.name}: {mat_count1}")
    print(f"   {glb2_path.name}: {mat_count2}")

    # Compare double-sided settings
    def check_doublesided(gltf_data):
        materials = gltf_data.get('materials', [])
        if not materials:
            return "NO MATERIALS"
        double_sided_count = sum(1 for m in materials if m.get('doubleSided', False))
        return f"{double_sided_count}/{len(materials)} are double-sided"

    ds1 = check_doublesided(gltf1)
    ds2 = check_doublesided(gltf2)

    print(f"\n🔄 Double-Sided:")
    print(f"   {glb1_path.name}: {ds1}")
    print(f"   {glb2_path.name}: {ds2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_glb_metadata.py <glb_file> [<comparison_glb>]")
        sys.exit(1)

    glb1 = Path(sys.argv[1])

    if not glb1.exists():
        print(f"ERROR: File not found: {glb1}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        # Comparison mode
        glb2 = Path(sys.argv[2])
        if not glb2.exists():
            print(f"ERROR: File not found: {glb2}")
            sys.exit(1)
        compare_glbs(glb1, glb2)
    else:
        # Single file mode
        inspect_glb(glb1)

    print("\n" + "="*70)
    print("✓ Inspection complete")
    print("="*70 + "\n")
