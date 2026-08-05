"""Pure PyTorch geometry helpers used by DiceDial and its CPU-only tests."""

import math

import torch


# Standard opposite-face convention: opposite faces sum to seven.
# The vectors are expressed in the die's local coordinate frame.
FACE_NORMALS = torch.tensor(
    [
        [0.0, 0.0, 1.0],   # 1: +Z
        [1.0, 0.0, 0.0],   # 2: +X
        [0.0, 1.0, 0.0],   # 3: +Y
        [0.0, -1.0, 0.0],  # 4: -Y
        [-1.0, 0.0, 0.0],  # 5: -X
        [0.0, 0.0, -1.0],  # 6: -Z
    ],
    dtype=torch.float32,
)


# Canonical orientations that place each numbered face upward. Quaternions use
# Isaac Lab's scalar-first convention: (w, x, y, z).
_s = math.sqrt(0.5)
FACE_UP_QUATERNIONS = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [_s, 0.0, -_s, 0.0],
        [_s, _s, 0.0, 0.0],
        [_s, -_s, 0.0, 0.0],
        [_s, 0.0, _s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def normalize_quaternion(quaternion):
    """Normalize scalar-first quaternions along the final dimension."""

    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def rotate_vector_wxyz(quaternion, vector):
    """Rotate vectors by scalar-first quaternions without requiring Isaac Lab.

    The implementation uses the equivalent cross-product form and supports
    arbitrary leading batch dimensions.
    """

    quaternion = normalize_quaternion(quaternion)
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    first_cross = torch.cross(xyz, vector, dim=-1)
    second_cross = torch.cross(xyz, first_cross, dim=-1)
    return vector + 2.0 * (scalar * first_cross + second_cross)


def target_face_alignment(object_quaternion, target_faces):
    """Return cosine alignment between the commanded face and world up."""

    normals = FACE_NORMALS.to(object_quaternion.device)[target_faces.long() - 1]
    world_normals = rotate_vector_wxyz(object_quaternion, normals)
    return world_normals[..., 2]


def top_face(object_quaternion):
    """Return the currently uppermost numbered face for each quaternion."""

    batch_shape = object_quaternion.shape[:-1]
    flat_quaternion = object_quaternion.reshape(-1, 4)
    count = flat_quaternion.shape[0]
    normals = FACE_NORMALS.to(object_quaternion.device)
    expanded_quaternion = flat_quaternion[:, None, :].expand(-1, 6, -1)
    expanded_normals = normals[None, :, :].expand(count, -1, -1)
    world_normals = rotate_vector_wxyz(expanded_quaternion, expanded_normals)
    faces = world_normals[..., 2].argmax(dim=-1) + 1
    return faces.reshape(batch_shape)


def canonical_goal_quaternion(target_faces, device=None):
    """Return a canonical die orientation for each requested face."""

    table = FACE_UP_QUATERNIONS.to(device or target_faces.device)
    return table[target_faces.long() - 1]
