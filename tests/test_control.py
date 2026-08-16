"""Unit tests for controller tensor selection and normalization."""

import unittest

import torch

from dicedial.control import normalize_selected_joint_targets


class TestControl(unittest.TestCase):
    def setUp(self):
        # Use different limits in every environment so indexing an environment
        # dimension with joint identifiers cannot accidentally pass.
        offsets = torch.tensor([[0.0], [10.0], [20.0], [30.0]])
        self.lower = offsets + torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        self.upper = self.lower + torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0]])
        self.midpoint = 0.5 * (self.lower + self.upper)
        self.joint_indices = [4, 1, 3]

    def test_all_environments_select_rows_then_joint_columns(self):
        normalized = normalize_selected_joint_targets(
            self.midpoint,
            self.lower,
            self.upper,
            self.joint_indices,
        )

        self.assertEqual(normalized.shape, (4, 3))
        self.assertTrue(torch.allclose(normalized, torch.zeros_like(normalized)))

    def test_environment_subset_preserves_requested_order(self):
        env_ids = torch.tensor([3, 1])
        targets = self.midpoint.clone()
        targets[3, self.joint_indices] = self.upper[3, self.joint_indices]
        targets[1, self.joint_indices] = self.lower[1, self.joint_indices]

        normalized = normalize_selected_joint_targets(
            targets,
            self.lower,
            self.upper,
            self.joint_indices,
            env_ids,
        )

        expected = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        self.assertEqual(normalized.shape, (2, 3))
        self.assertTrue(torch.allclose(normalized, expected))

    def test_out_of_range_targets_are_clamped(self):
        targets = self.midpoint.clone()
        targets[:, self.joint_indices] = self.upper[:, self.joint_indices] + 100.0

        normalized = normalize_selected_joint_targets(
            targets,
            self.lower,
            self.upper,
            self.joint_indices,
        )

        self.assertTrue(torch.equal(normalized, torch.ones_like(normalized)))


if __name__ == "__main__":
    unittest.main()
