#!/usr/bin/env python3
"""
Split a semantic GLB (single mesh with COLOR_0 per-vertex) into a new GLB
containing one mesh/node per color group (i.e. semantic id). This script
computes per-group bounding boxes and centroid, and prints diagnostic info
before and after.

Usage:
    python tools/split_semantic_glb.py input.semantic.glb output_split.glb

Dependencies: numpy

This script does not require external GLTF libs; it reads/parses the GLB JSON
chunk and binary chunk directly.
"""
import sys
import os
import json
import struct
import argparse
from collections import defaultdict
import numpy as np

# helpers for glTF component type -> numpy dtype
COMPONENT_TYPE_TO_DTYPE = {
    5120: np.int8,    # BYTE
    5121: np.uint8,   # UNSIGNED_BYTE
    5122: np.int16,   # SHORT
    5123: np.uint16,  # UNSIGNED_SHORT
    5125: np.uint32,  # UNSIGNED_INT
    5126: np.float32, # FLOAT
}

TYPE_TO_NUM = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16,
}


def load_glb(path):
    with open(path, 'rb') as f:
        header = f.read(12)
        if len(header) != 12:
            raise RuntimeError('Invalid GLB header')
        magic, version, length = struct.unpack('<4sII', header)
        if magic != b'glTF':
            raise RuntimeError('Not a glTF binary')
        data = f.read()

    idx = 0
    json_chunk = None
    bin_chunk = None
    while idx < len(data):
        if idx + 8 > len(data):
            break
        chunk_len, chunk_type = struct.unpack('<I4s', data[idx:idx+8])
        idx += 8
        chunk_data = data[idx:idx+chunk_len]
        idx += chunk_len
        chunk_type_s = chunk_type.decode('ascii', errors='ignore')
        if chunk_type_s.strip() == 'JSON':
            json_chunk = chunk_data.decode('utf-8')
        elif chunk_type_s.strip() == 'BIN\x00' or chunk_type_s.strip() == 'BIN':
            bin_chunk = chunk_data
    if json_chunk is None:
        raise RuntimeError('No JSON chunk in GLB')
    if bin_chunk is None:
        # Some GLBs embed buffer as separate external file; we expect BIN chunk
        bin_chunk = b''
    gltf = json.loads(json_chunk)
    return gltf, bin_chunk


def get_accessor_data(gltf, bin_chunk, accessor_idx):
    accessors = gltf.get('accessors', [])
    bufferViews = gltf.get('bufferViews', [])

    acc = accessors[accessor_idx]
    bv_index = acc.get('bufferView')
    if bv_index is None:
        # sparse or empty? not supported
        raise RuntimeError('Accessor without bufferView not supported')
    bv = bufferViews[bv_index]
    buffer_byte_offset = bv.get('byteOffset', 0) or 0
    accessor_byte_offset = acc.get('byteOffset', 0) or 0
    byte_offset = buffer_byte_offset + accessor_byte_offset
    count = acc['count']
    comp_type = acc['componentType']
    dtype = COMPONENT_TYPE_TO_DTYPE.get(comp_type)
    if dtype is None:
        raise RuntimeError(f'Unsupported componentType: {comp_type}')
    type_str = acc['type']
    num = TYPE_TO_NUM[type_str]

    # handle byteStride if present
    bv_byte_stride = bv.get('byteStride')
    if bv_byte_stride:
        # Need to read per-entry stride
        itemsize = np.dtype(dtype).itemsize * num
        data = np.zeros((count, num), dtype=dtype)
        for i in range(count):
            off = byte_offset + i * bv_byte_stride
            arr = np.frombuffer(bin_chunk, dtype=dtype, count=num, offset=off)
            data[i, :] = arr
        return data
    else:
        total_elems = count * num
        arr = np.frombuffer(bin_chunk, dtype=dtype, count=total_elems, offset=byte_offset)
        if num == 1:
            return arr.copy()
        else:
            return arr.reshape((count, num)).copy()


def pack_vec(arr):
    return arr.tobytes()


def pad_to_4(b):
    pad = (4 - (len(b) % 4)) % 4
    if pad:
        return b + (b' ' * pad)
    return b


def build_new_glb(out_path, groups):
    # groups: list of dicts {name, positions(np.float32 Nx3), normals(opt), indices(np.uint32 M), bbox (min,max), centroid}
    gltf = {
        'asset': {'version': '2.0'},
        'scenes': [{'nodes': list(range(len(groups)))}],
        'nodes': [],
        'meshes': [],
        'accessors': [],
        'bufferViews': [],
        'buffers': [],
        'materials': [{'name': 'semantic_material'}],
    }

    bin_parts = []
    # Build nodes and meshes
    for gi, g in enumerate(groups):
        mesh_idx = len(gltf['meshes'])
        # Prepare position buffer
        positions = g['positions'].astype(np.float32)
        normals = g.get('normals')
        indices = g['indices'].astype(np.uint32)

        # align each bufferView at 4-byte boundary
        pos_offset = sum(len(p) for p in bin_parts)
        pos_bytes = pad_to_4(positions.tobytes())
        bin_parts.append(pos_bytes)
        pos_bv_index = len(gltf['bufferViews'])
        gltf['bufferViews'].append({'buffer': 0, 'byteOffset': pos_offset, 'byteLength': len(pos_bytes)})
        pos_accessor_index = len(gltf['accessors'])
        min_vals = positions.min(axis=0).astype(float).tolist()
        max_vals = positions.max(axis=0).astype(float).tolist()
        gltf['accessors'].append({'bufferView': pos_bv_index, 'byteOffset': 0, 'componentType': 5126, 'count': positions.shape[0], 'type': 'VEC3', 'min': min_vals, 'max': max_vals})

        # normals if present
        norm_accessor_index = None
        if normals is not None:
            norm_offset = sum(len(p) for p in bin_parts)
            norm_bytes = pad_to_4(normals.astype(np.float32).tobytes())
            bin_parts.append(norm_bytes)
            norm_bv_index = len(gltf['bufferViews'])
            gltf['bufferViews'].append({'buffer': 0, 'byteOffset': norm_offset, 'byteLength': len(norm_bytes)})
            norm_accessor_index = len(gltf['accessors'])
            gltf['accessors'].append({'bufferView': norm_bv_index, 'byteOffset': 0, 'componentType': 5126, 'count': normals.shape[0], 'type': 'VEC3'})

        # indices
        idx_offset = sum(len(p) for p in bin_parts)
        idx_bytes = pad_to_4(indices.astype(np.uint32).tobytes())
        bin_parts.append(idx_bytes)
        idx_bv_index = len(gltf['bufferViews'])
        gltf['bufferViews'].append({'buffer': 0, 'byteOffset': idx_offset, 'byteLength': len(idx_bytes)})
        idx_accessor_index = len(gltf['accessors'])
        gltf['accessors'].append({'bufferView': idx_bv_index, 'byteOffset': 0, 'componentType': 5125, 'count': indices.shape[0], 'type': 'SCALAR'})

        # mesh primitive
        prim = {'attributes': {'POSITION': pos_accessor_index}}
        if norm_accessor_index is not None:
            prim['attributes']['NORMAL'] = norm_accessor_index
        prim['indices'] = idx_accessor_index
        prim['material'] = 0
        gltf['meshes'].append({'primitives': [prim], 'name': g['name']})

        # node
        gltf['nodes'].append({'mesh': mesh_idx, 'name': g['name']})

    # finalize buffer
    bin_blob = b''.join(bin_parts)
    gltf['buffers'].append({'byteLength': len(bin_blob)})

    json_bytes = json.dumps(gltf, indent=2).encode('utf-8')
    json_padded = pad_to_4(json_bytes)
    bin_padded = pad_to_4(bin_blob)

    # construct GLB
    magic = b'glTF'
    version = 2
    total_length = 12 + 8 + len(json_padded) + 8 + len(bin_padded)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<4sII', magic, version, total_length))
        # JSON chunk
        f.write(struct.pack('<I4s', len(json_padded), b'JSON'))
        f.write(json_padded)
        # BIN chunk
        f.write(struct.pack('<I4s', len(bin_padded), b'BIN\x00'))
        f.write(bin_padded)


def analyze_gltf_summary(gltf):
    nodes = gltf.get('nodes', [])
    meshes = gltf.get('meshes', [])
    materials = gltf.get('materials', [])
    print('GLTF summary: nodes=%d meshes=%d materials=%d' % (len(nodes), len(meshes), len(materials)))
    for i, m in enumerate(meshes[:20]):
        prims = m.get('primitives', [])
        print(f' Mesh {i}: name={m.get("name")} primitives={len(prims)}')
        for j, p in enumerate(prims):
            attrs = p.get('attributes', {})
            print('  Prim', j, 'attributes', list(attrs.keys()), 'indices', p.get('indices'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_glb')
    parser.add_argument('output_glb')
    args = parser.parse_args()

    in_path = args.input_glb
    out_path = args.output_glb

    print('Loading input GLB:', in_path)
    gltf, bin_chunk = load_glb(in_path)
    print('\n--- BEFORE ---')
    analyze_gltf_summary(gltf)

    # locate a single mesh primitive
    meshes = gltf.get('meshes', [])
    if len(meshes) == 0:
        print('No meshes found in input GLB')
        sys.exit(1)
    # For simplicity take first mesh and first primitive
    primitive = meshes[0].get('primitives', [])[0]
    attributes = primitive.get('attributes', {})
    if 'POSITION' not in attributes:
        print('No POSITION attribute in primitive')
        sys.exit(1)
    position_acc_idx = attributes['POSITION']
    pos = get_accessor_data(gltf, bin_chunk, position_acc_idx)
    # normals optional
    normals = None
    if 'NORMAL' in attributes:
        normals = get_accessor_data(gltf, bin_chunk, attributes['NORMAL'])
    # colors
    if 'COLOR_0' not in attributes:
        print('No COLOR_0 attribute found; cannot split by semantic color')
        sys.exit(1)
    colors = get_accessor_data(gltf, bin_chunk, attributes['COLOR_0'])
    # indices
    if 'indices' in primitive:
        indices_raw = get_accessor_data(gltf, bin_chunk, primitive['indices'])
        # indices array may be (count,) ; ensure shape
        indices = indices_raw.reshape((-1,))
    else:
        # create sequential
        indices = np.arange(pos.shape[0], dtype=np.uint32)
    # convert indices to triangles
    if indices.size % 3 != 0:
        print('Warning: indices length not multiple of 3; attempting to reshape by floor')
    faces = indices.reshape((-1, 3))

    # Normalize colors to uint8 tuple
    if colors.dtype == np.float32 or colors.dtype == np.float64:
        # assume floats in [0,1]
        if colors.max() <= 1.1:
            cols_u8 = np.clip((colors * 255.0).round(), 0, 255).astype(np.uint8)
        else:
            cols_u8 = np.clip(colors.round(), 0, 255).astype(np.uint8)
    else:
        # integer types
        cols_u8 = colors.astype(np.uint8)
    # If COLOR_0 has 4 components, drop alpha for grouping
    if cols_u8.shape[1] >= 3:
        cols3 = cols_u8[:, :3]
    else:
        # pad
        cols3 = np.pad(cols_u8, ((0,0),(0,3-cols_u8.shape[1])), constant_values=0)[:,:3]

    # map faces to color groups using first vertex color
    face_colors = [tuple(cols3[f[0]].tolist()) for f in faces]
    groups_idx = defaultdict(list)
    for fi, fc in enumerate(face_colors):
        groups_idx[fc].append(fi)

    print(f'Found {len(groups_idx)} color groups (semantic ids) from {faces.shape[0]} faces')

    groups = []
    for color, face_list in groups_idx.items():
        # collect vertex indices used
        used_face_idx = np.array(face_list, dtype=np.int32)
        used_vertex_indices = np.unique(faces[used_face_idx].reshape(-1))
        old_to_new = {old: new for new, old in enumerate(used_vertex_indices)}
        positions_sub = pos[used_vertex_indices]
        normals_sub = None
        if normals is not None:
            normals_sub = normals[used_vertex_indices]
        # remap faces
        faces_sub = faces[used_face_idx]
        remapped = np.vectorize(lambda x: old_to_new[x])(faces_sub)
        # flatten indices
        indices_sub = remapped.reshape(-1).astype(np.uint32)
        # compute bbox
        bbox_min = positions_sub.min(axis=0)
        bbox_max = positions_sub.max(axis=0)
        centroid = (bbox_min + bbox_max) / 2.0
        name = f'semantic_{color[0]:03d}_{color[1]:03d}_{color[2]:03d}'
        groups.append({'name': name, 'positions': positions_sub, 'normals': normals_sub, 'indices': indices_sub, 'bbox_min': bbox_min, 'bbox_max': bbox_max, 'centroid': centroid, 'color': color})

    # print some group stats
    print('\nGroup stats (first 20):')
    for g in groups[:20]:
        print(f" {g['name']}: verts={g['positions'].shape[0]} faces={g['indices'].size//3} bbox_min={g['bbox_min']} bbox_max={g['bbox_max']}")

    # build new glb
    print('\nBuilding new GLB with separate meshes/nodes...')
    build_new_glb(out_path, groups)

    # analyze output
    out_gltf, out_bin = load_glb(out_path)
    print('\n--- AFTER ---')
    analyze_gltf_summary(out_gltf)
    print('Wrote output GLB:', out_path)


if __name__ == '__main__':
    main()
