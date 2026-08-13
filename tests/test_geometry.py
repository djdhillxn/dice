"""Unit tests for geometry calculations and die face math."""

import unittest

import torch

from dicedial.geometry import (
    FACE_UP_QUATERNIONS,
    normalize_quaternion,
    rotate_vector_wxyz,
    rotation_6d_from_quaternion_wxyz,
    target_face_alignment,
    top_face,
)


class TestGeometry(unittest.TestCase):
    def test_normalize_quaternion(self):
        q = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
        q_norm = normalize_quaternion(q)
        norms = q_norm.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))

    def test_rotate_vector_identity(self):
        q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        vec = torch.tensor([[1.0, 2.0, 3.0]])
        rotated = rotate_vector_wxyz(q_identity, vec)
        self.assertTrue(torch.allclose(rotated, vec))

    def test_rotation_6d_and_matrix(self):
        q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        rot_6d = rotation_6d_from_quaternion_wxyz(q_identity)
        self.assertEqual(rot_6d.shape, (1, 6))
        # For identity, first two columns are [1,0,0] and [0,1,0]
        expected_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        self.assertTrue(torch.allclose(rot_6d, expected_6d))

    def test_top_face_and_alignment(self):
        # Face 1 canonical quat is identity -> +Z points up -> top face 1
        q_face1 = FACE_UP_QUATERNIONS[0:1]
        top1 = top_face(q_face1)
        self.assertEqual(top1.item(), 1)

        align1 = target_face_alignment(q_face1, torch.tensor([1]))
        self.assertAlmostEqual(align1.item(), 1.0, places=5)

    def test_opposite_face_sum_rule(self):
        # Opposite faces in standard dice: (1,6), (2,5), (3,4) all sum to 7
        for f1 in range(1, 7):
            for f2 in range(1, 7):
                is_opp = f1 + f2 == 7
                if (f1, f2) in [(1, 6), (6, 1), (2, 5), (5, 2), (3, 4), (4, 3)]:
                    self.assertTrue(is_opp, f"Expected {f1} and {f2} to be opposite")
                else:
                    self.assertFalse(
                        is_opp, f"Expected {f1} and {f2} not to be opposite"
                    )


if __name__ == "__main__":
    unittest.main()
