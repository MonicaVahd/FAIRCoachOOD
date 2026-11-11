"""
tests/test_preprocessing.py — Shape and value tests for preprocessing utilities.

Runs on synthetic data only (no datasets required).  Verifies that the
seven-step pipeline produces the correct output shapes and normalization range.
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.common import (
    normalize_xy,
    pad_or_trim,
    pad_or_interpolate,
    ntu25_to_openpose18,
    rehab26_to_openpose18,
    kinect22_to_openpose18,
)


class TestNormalizeXY(unittest.TestCase):
    def test_shape_preserved(self):
        arr = np.random.randn(3, 100, 18).astype(np.float32)
        out = normalize_xy(arr.copy())
        self.assertEqual(out.shape, arr.shape)

    def test_channels_0_1_in_range(self):
        arr = np.random.randn(3, 100, 18).astype(np.float32) * 10
        out = normalize_xy(arr.copy())
        self.assertAlmostEqual(float(out[0].min()), 0.0, places=5)
        self.assertAlmostEqual(float(out[0].max()), 1.0, places=5)
        self.assertAlmostEqual(float(out[1].min()), 0.0, places=5)
        self.assertAlmostEqual(float(out[1].max()), 1.0, places=5)

    def test_constant_channel_unchanged(self):
        arr = np.zeros((3, 100, 18), dtype=np.float32)
        arr[2] = 5.0
        out = normalize_xy(arr.copy())
        # Channel 0 and 1 are all-zero; v_max == v_min so not normalized
        np.testing.assert_allclose(out[2], arr[2])

    def test_channel_2_not_renormalized(self):
        """normalize_xy only iterates over ch in range(2)."""
        arr = np.ones((3, 100, 18), dtype=np.float32)
        arr[2] = np.arange(18, dtype=np.float32)  # varying channel 2
        out = normalize_xy(arr.copy())
        # Channel 2 must NOT be changed
        np.testing.assert_allclose(out[2], arr[2])


class TestPadOrTrim(unittest.TestCase):
    def test_trim(self):
        seq = np.ones((150, 3, 18), dtype=np.float32)
        out = pad_or_trim(seq, 100)
        self.assertEqual(out.shape, (100, 3, 18))

    def test_pad(self):
        seq = np.ones((60, 3, 18), dtype=np.float32)
        out = pad_or_trim(seq, 100)
        self.assertEqual(out.shape, (100, 3, 18))
        # Padded frames should be zero
        np.testing.assert_allclose(out[60:], 0)

    def test_exact(self):
        seq = np.ones((100, 3, 18), dtype=np.float32)
        out = pad_or_trim(seq, 100)
        self.assertEqual(out.shape, (100, 3, 18))


class TestPadOrInterpolate(unittest.TestCase):
    def test_exact(self):
        seq = np.random.randn(100, 18, 3).astype(np.float32)
        out = pad_or_interpolate(seq, 100)
        np.testing.assert_allclose(out, seq, atol=1e-6)

    def test_upsample(self):
        seq = np.random.randn(50, 18, 3).astype(np.float32)
        out = pad_or_interpolate(seq, 100)
        self.assertEqual(out.shape, (100, 18, 3))

    def test_downsample(self):
        seq = np.random.randn(200, 18, 3).astype(np.float32)
        out = pad_or_interpolate(seq, 100)
        self.assertEqual(out.shape, (100, 18, 3))

    def test_single_frame(self):
        seq = np.ones((1, 18, 3), dtype=np.float32) * 7.0
        out = pad_or_interpolate(seq, 100)
        self.assertEqual(out.shape, (100, 18, 3))
        np.testing.assert_allclose(out, 7.0)


class TestJointRemapping(unittest.TestCase):
    def test_ntu25_shape(self):
        seq = np.random.randn(100, 3, 25).astype(np.float32)
        out = ntu25_to_openpose18(seq)
        self.assertEqual(out.shape, (100, 3, 18))

    def test_ntu25_confidence_before_centering(self):
        """Channel 2 of real joints is set to 1.0 by ntu25_to_openpose18."""
        seq = np.random.randn(10, 3, 25).astype(np.float32)
        out = ntu25_to_openpose18(seq)
        # Only mapped joints (NTU_TO_OPENPOSE = 14 joints → positions 0-13)
        from preprocessing.common import NTU_TO_OPENPOSE
        mapped_positions = list(range(len(NTU_TO_OPENPOSE)))
        for pos in mapped_positions:
            np.testing.assert_allclose(
                out[:, 2, pos], 1.0,
                err_msg=f"confidence at position {pos} should be 1.0"
            )

    def test_rehab26_shape(self):
        seq = np.random.randn(80, 26, 2).astype(np.float32)
        out = rehab26_to_openpose18(seq)
        self.assertEqual(out.shape, (80, 3, 18))

    def test_kinect22_shape(self):
        seq = np.random.randn(120, 22, 3).astype(np.float32)
        out = kinect22_to_openpose18(seq)
        self.assertEqual(out.shape, (120, 18, 3))


class TestFullPipeline(unittest.TestCase):
    """End-to-end shape test for the EC3D-style pipeline."""

    def _run_ec3d_pipeline(self, T_raw: int) -> np.ndarray:
        from preprocessing.common import ntu25_to_openpose18, pad_or_trim, normalize_xy
        seq = np.random.randn(T_raw, 3, 25).astype(np.float32)
        seq18 = ntu25_to_openpose18(seq)                           # (T, 3, 18)
        seq18 = seq18 - seq18[:, :, 8:9]                          # hip-center
        torso = np.linalg.norm(seq18[:, :, 1] - seq18[:, :, 8], axis=1).mean()
        if torso > 1e-6:
            seq18 = seq18 / torso
        fixed = pad_or_trim(seq18, 100)                            # (100, 3, 18)
        arr = np.transpose(fixed, (1, 0, 2))                      # (3, 100, 18)
        arr = normalize_xy(arr)
        arr = arr[:, :, :, np.newaxis]                             # (3, 100, 18, 1)
        return arr

    def test_short_sequence(self):
        arr = self._run_ec3d_pipeline(40)
        self.assertEqual(arr.shape, (3, 100, 18, 1))

    def test_long_sequence(self):
        arr = self._run_ec3d_pipeline(200)
        self.assertEqual(arr.shape, (3, 100, 18, 1))

    def test_exact_sequence(self):
        arr = self._run_ec3d_pipeline(100)
        self.assertEqual(arr.shape, (3, 100, 18, 1))

    def test_dtype(self):
        arr = self._run_ec3d_pipeline(80)
        self.assertEqual(arr.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
