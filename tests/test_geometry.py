import torch

from dicedial.geometry import (
    FACE_NORMALS,
    canonical_goal_quaternion,
    target_face_alignment,
    top_face,
)


def test_opposite_faces_sum_to_seven():
    opposite_pairs = [(1, 6), (2, 5), (3, 4)]
    for first, second in opposite_pairs:
        assert torch.allclose(FACE_NORMALS[first - 1], -FACE_NORMALS[second - 1])
        assert first + second == 7


def test_canonical_quaternion_places_requested_face_up():
    faces = torch.arange(1, 7)
    quaternions = canonical_goal_quaternion(faces)
    alignment = target_face_alignment(quaternions, faces)
    assert torch.allclose(alignment, torch.ones(6), atol=1.0e-5)


def test_top_face_matches_canonical_goal():
    faces = torch.arange(1, 7)
    quaternions = canonical_goal_quaternion(faces)
    assert torch.equal(top_face(quaternions), faces)


def test_identity_has_face_one_up():
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert top_face(quaternion).item() == 1
