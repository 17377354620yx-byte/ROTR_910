"""Auditable DePoLL geometry and metrics, all physical coordinates in mm.

The local release's marker YAMLs are in CT coordinates despite the prose in
its description PDF. M maps each state's CT markers to reconstruction space.
The released surface_CT and video_reconstruction are already in that space.
"""
import re
from pathlib import Path
import numpy as np


def read_matrix(path, key):
    text = Path(path).read_text()
    match = re.search(r'\b' + re.escape(key) + r':\s*!!opencv-matrix\s+rows:\s*(\d+)\s+cols:\s*(\d+)\s+dt:\s*\w+\s+data:\s*\[([^]]+)\]', text)
    if match is None:
        raise ValueError(f'Missing OpenCV matrix {key}: {path}')
    rows, cols, data = match.groups()
    result = np.fromstring(data, sep=',').reshape(int(rows), int(cols))
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite matrix: {path}')
    return result


def read_markers(path):
    result = {}
    for group, count in [('balls', 45), ('clips', 15)]:
        points = read_matrix(path, 'centroids' + group.capitalize())
        if points.shape == (3, count):
            points = points.T
        if points.shape != (count, 3):
            raise ValueError(f'Unexpected {group} shape in {path}: {points.shape}')
        result[group] = points.copy()
    return result


def read_ply(path):
    """Read scalar vertex properties from ASCII/binary PLY; ignore mesh faces."""
    types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
             'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4'}
    with open(path, 'rb') as handle:
        if handle.readline().strip() != b'ply':
            raise ValueError(f'Not PLY: {path}')
        properties, count, vertex, fmt = [], None, False, None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f'Truncated PLY: {path}')
            fields = line.decode('ascii').split()
            if fields[:1] == ['format']:
                fmt = fields[1]
            elif fields[:1] == ['element']:
                vertex = fields[1] == 'vertex'
                if vertex:
                    count = int(fields[2])
            elif fields[:1] == ['property'] and vertex:
                if fields[1] not in types:
                    raise ValueError(f'Unsupported vertex property: {line!r}')
                properties.append((fields[2], types[fields[1]]))
            elif fields[:1] == ['end_header']:
                break
        if not count:
            raise ValueError(f'Empty/missing vertices: {path}')
        if fmt == 'ascii':
            raw = np.loadtxt(handle, max_rows=count, ndmin=2)
            points = raw[:, [[k for k, _ in properties].index(a) for a in 'xyz']]
        elif fmt in ('binary_little_endian', 'binary_big_endian'):
            endian = '<' if fmt == 'binary_little_endian' else '>'
            raw = np.fromfile(handle, dtype=np.dtype([(k, endian + t) for k, t in properties]), count=count)
            points = np.column_stack([raw[a] for a in 'xyz'])
        else:
            raise ValueError(f'Unsupported PLY format: {fmt}')
    if len(points) != count or not np.isfinite(points).all():
        raise ValueError(f'Invalid vertices: {path}')
    return points.astype(np.float64)


def apply(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def marker_errors(source, target, transform):
    distances = np.linalg.norm(apply(source, transform) - target, axis=1)
    return {'tre_mm': float(distances.mean()),
            'rms_tre_mm': float(np.sqrt(np.mean(distances ** 2))),
            'distances_mm': distances.tolist()}


def physical_transform(normalized_transform, source_center, target_center, scale):
    result = np.asarray(normalized_transform, dtype=np.float64).copy()
    result[:3, 3] = scale * result[:3, 3] + target_center - result[:3, :3] @ source_center
    return result


def load_case(root, case, protocol):
    root = Path(root)
    M = read_matrix(root / 'coordinate_transforms' / case / 'M.yml', 'M')
    if M.shape != (4, 4) or not np.allclose(M[3], [0, 0, 0, 1]):
        raise ValueError(f'Invalid M for {case}')
    if not np.allclose(M[:3, :3].T @ M[:3, :3], np.eye(3), atol=2e-5) or np.linalg.det(M[:3, :3]) < 0:
        raise ValueError(f'Nonrigid M for {case}')
    intra_ct = read_markers(root / case / 'markers/markers.yml')
    target_markers = {k: apply(v, M) for k, v in intra_ct.items()}
    target = read_ply(root / case / 'surfaces/video_reconstruction.ply')
    if protocol == 'intraop':
        surface_path = root / case / 'surfaces/surface_CT.ply'
        source_registered_markers = target_markers
    elif protocol == 'preop':
        surface_path = root / 'reference/referenceModel/segmentedLobes/surface.ply'
        if not surface_path.is_file():
            surface_path = surface_path.with_name('surfaceFull.ply')
        reference = read_markers(root / 'reference/referenceModel/markers/markers.yml')
        association = read_matrix(root / case / 'markers/associations.yml', 'Association').ravel()
        if not np.array_equal(np.sort(association), np.arange(1, 61)):
            raise ValueError(f'Expected a 1-based 60-marker permutation: {case}')
        if not ((association[:45] <= 45).all() and (association[45:] > 45).all()):
            raise ValueError(f'Cross-group associations: {case}')
        reference_all = np.concatenate([reference['balls'], reference['clips']])
        # Local release maps reference index -> intraoperative index (1-based).
        # Invert to arrange reference markers in intraoperative YAML order.
        # This reproduces paper Table 4 Ground-Truth to both decimal places.
        ordered = reference_all[np.argsort(association.astype(int) - 1)]
        source_registered_markers = {'balls': ordered[:45], 'clips': ordered[45:]}
    else:
        raise ValueError(protocol)
    source_registered = read_ply(surface_path)
    inverse = np.linalg.inv(M)
    # Undo released alignment on the source AND its markers. Never feed M or
    # marker positions into the model. Output maps this restored source to video.
    source = apply(source_registered, inverse)
    source_markers = {k: apply(v, inverse) for k, v in source_registered_markers.items()}
    return dict(source=source, target=target, source_markers=source_markers,
                target_markers=target_markers, released_transform=M,
                source_path=str(surface_path))
