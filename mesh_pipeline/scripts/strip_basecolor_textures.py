#!/usr/bin/env python3
"""
Strip material baseColorTexture references from a GLB so viewers will use vertex colors.

Usage: python3 scripts/strip_basecolor_textures.py input.glb output.glb

This script:
- extracts the JSON chunk from the GLB
- removes `pbrMetallicRoughness.baseColorTexture` from all materials
- sets `pbrMetallicRoughness.baseColorFactor` to [1,1,1,1] if missing
- writes a new GLB with the modified JSON chunk and original BIN chunk(s) unchanged

It does not attempt to remove embedded images/buffers — they will remain in the new GLB but unreferenced.
"""
import sys, json, struct

def usage():
    print("Usage: strip_basecolor_textures.py input.glb output.glb")

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
        json_type = json_type.decode('utf-8')
        json_bytes = f.read(json_len)
        # read remaining as binary chunk(s)
        rest = f.read()

    js = json.loads(json_bytes.decode('utf-8'))
    mats = js.get('materials')
    if not mats:
        print('No materials found; writing copy')
    else:
        for m in mats:
            pbr = m.get('pbrMetallicRoughness')
            if isinstance(pbr, dict):
                if 'baseColorTexture' in pbr:
                    del pbr['baseColorTexture']
                # ensure baseColorFactor present so material isn't fully white
                if 'baseColorFactor' not in pbr:
                    pbr['baseColorFactor'] = [1.0,1.0,1.0,1.0]

    new_json = json.dumps(js, separators=(',',':'), ensure_ascii=False).encode('utf-8')
    # pad to 4-byte alignment
    pad_len = (4 - (len(new_json) % 4)) % 4
    new_json += b' ' * pad_len

    # rebuild GLB: header + JSON chunk header + JSON chunk + rest unchanged
    with open(outp, 'wb') as f:
        total_len = 12 + 8 + len(new_json) + len(rest)
        f.write(struct.pack('<4sII', b'glTF', version, total_len))
        f.write(struct.pack('<I4s', len(new_json), b'JSON'))
        f.write(new_json)
        f.write(rest)

if __name__ == '__main__':
    main()
