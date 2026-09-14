"""Experimental partial-to-complete point-to-plane refinement (no markers/GT)."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def refine_surface(source, reference, initial, iterations=15, radius=0.1, kernel=0.02):
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    transform = np.array(initial, dtype=np.float64, copy=True)
    if min(len(source), len(reference)) < 20:
        return transform
    tree = cKDTree(source)
    neighbours = source[tree.query(source, k=min(20, len(source)))[1]]
    centered = neighbours - neighbours.mean(1, keepdims=True)
    _, vectors = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered)
    normals = vectors[:, :, 0]
    for _ in range(iterations):
        rot, trans = transform[:3, :3], transform[:3, 3]
        distances, indices = tree.query((reference - trans) @ rot)
        valid = distances < radius
        if valid.sum() < 20:
            break
        p = source[indices[valid]] @ rot.T + trans
        n = normals[indices[valid]] @ rot.T
        residual = ((reference[valid] - p) * n).sum(1)
        weights = 1 / (1 + (residual / kernel)**2)
        jac = np.concatenate([np.cross(p, n), n], axis=1)
        hessian = jac.T @ (weights[:, None] * jac)
        if np.linalg.cond(hessian) > 1e7:
            break
        step = np.linalg.solve(hessian + np.eye(6)*1e-6, jac.T @ (weights * residual))
        # A local corrector must not turn into an unrestricted pose search.
        if np.linalg.norm(step[:3]) > 0.1 or np.linalg.norm(step[3:]) > 0.05:
            break
        update = np.eye(4)
        update[:3, :3] = Rotation.from_rotvec(step[:3]).as_matrix()
        update[:3, 3] = step[3:]
        candidate = update @ transform
        total = candidate @ np.linalg.inv(initial)
        if (np.linalg.norm(Rotation.from_matrix(total[:3, :3]).as_rotvec()) > 0.2
                or np.linalg.norm(total[:3, 3]) > 0.1):
            break
        transform = candidate
        if np.linalg.norm(step) < 1e-6:
            break
    return transform
