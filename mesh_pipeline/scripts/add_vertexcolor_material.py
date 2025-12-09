#!/usr/bin/env python3
"""
Add a simple material (no baseColorTexture) to a GLB and assign it to primitives
that lack a `material` index so viewers will use vertex colors (`COLOR_0`).

Usage: python3 scripts/add_vertexcolor_material.py input.glb output.glb

The script:
- extracts the JSON chunk
- ensures there's a `materials` array and appends a fallback material
  with `pbrMetallicRoughness.baseColorFactor=[1,1,1,1]`
- assigns the new material index to any primitive missing `material`
- writes a new GLB with modified JSON and original binary chunks
"""
import sys, json, struct

def usage():
    print("Usage: add_vertexcolor_material.py input.glb output.glb")

def main():
    if len(sys.argv) != 3:
        usage(); sys.exit(2)
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp, 'rb') as f:
        header = f.read(12)
        if len(header) < 12:
            raise SystemExit('Not a valid GLB (header too short)')
        magic, version, length = struct.unpack('<4sII', header)
        if magic != b'glTF':
            raise SystemExit('Not a GLB (magic mismatch)')
        # read first chunk (JSON)
        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            raise SystemExit('No chunks in GLB')
        json_len, json_type = struct.unpack('<I4s', chunk_header)
        json_bytes = f.read(json_len)
        rest = f.read()

    js = json.loads(json_bytes.decode('utf-8'))
    meshes = js.setdefault('meshes', [])
    mats = js.get('materials')
    if not mats:
        js['materials'] = []
        mats = js['materials']

    # create fallback material
    fallback = {
        'name': 'vertexcolor_fallback',
        'pbrMetallicRoughness': {
            'baseColorFactor': [1.0, 1.0, 1.0, 1.0],
            'metallicFactor': 0.0
        }
    }
    mats.append(fallback)
    fallback_index = len(mats) - 1

    # assign to primitives missing material
    for mesh in meshes:
        for prim in mesh.get('primitives', []):
            if 'material' not in prim:
                prim['material'] = fallback_index

    new_json = json.dumps(js, separators=(',',':'), ensure_ascii=False).encode('utf-8')
    pad_len = (4 - (len(new_json) % 4)) % 4
    new_json += b' ' * pad_len

    with open(outp, 'wb') as f:
        total_len = 12 + 8 + len(new_json) + len(rest)
        f.write(struct.pack('<4sII', b'glTF', version, total_len))
        f.write(struct.pack('<I4s', len(new_json), b'JSON'))
        f.write(new_json)
        f.write(rest)

if __name__ == '__main__':
    main()
