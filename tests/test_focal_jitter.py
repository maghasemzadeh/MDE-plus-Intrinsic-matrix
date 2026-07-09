"""Unit tests for the focal-depth rescaling ("focal jitter") augmentation.

Pure-array tests of NYUTrainingDataset._apply_focal_jitter — no .mat file,
no GPU needed.
"""

import numpy as np
import pytest

from datasets.training_datasets import NYUTrainingDataset

apply_jitter = NYUTrainingDataset._apply_focal_jitter


def make_inputs():
    depth = np.linspace(0.5, 9.5, 12, dtype=np.float32).reshape(3, 4)
    K = np.array([
        [518.8579, 0.0, 325.5824],
        [0.0, 519.4696, 253.7362],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return depth, K


class TestApplyFocalJitter:
    def test_focal_entries_scaled_by_alpha(self):
        depth, K = make_inputs()
        d2, K2, alpha = apply_jitter(depth, K, 1.6)
        assert np.isclose(K2[0, 0], K[0, 0] * alpha)
        assert np.isclose(K2[1, 1], K[1, 1] * alpha)

    def test_principal_point_unchanged(self):
        depth, K = make_inputs()
        _, K2, _ = apply_jitter(depth, K, 1.6)
        assert K2[0, 2] == K[0, 2]
        assert K2[1, 2] == K[1, 2]
        assert K2[2, 2] == 1.0
        assert K2[0, 1] == 0.0 and K2[1, 0] == 0.0

    def test_depth_scaled_by_same_alpha(self):
        depth, K = make_inputs()
        d2, K2, alpha = apply_jitter(depth, K, 1.6)
        np.testing.assert_allclose(d2, depth * alpha, rtol=1e-6)
        # the focal/depth ratio structure is preserved exactly
        np.testing.assert_allclose(K2[0, 0] / d2, K[0, 0] / depth, rtol=1e-5)

    def test_inputs_not_mutated(self):
        depth, K = make_inputs()
        depth_orig, K_orig = depth.copy(), K.copy()
        apply_jitter(depth, K, 1.6)
        np.testing.assert_array_equal(depth, depth_orig)
        np.testing.assert_array_equal(K, K_orig)

    def test_alpha_within_range(self):
        depth, K = make_inputs()
        m = 1.6
        for _ in range(500):
            _, _, alpha = apply_jitter(depth, K, m)
            assert 1.0 / m <= alpha <= m

    def test_log_uniform_symmetry(self):
        depth, K = make_inputs()
        rng_state = np.random.get_state()
        try:
            np.random.seed(1234)
            logs = [np.log(apply_jitter(depth, K, 1.6)[2]) for _ in range(20000)]
        finally:
            np.random.set_state(rng_state)
        # E[log alpha] = 0 for a log-uniform symmetric draw; std of the mean
        # is log(1.6)/sqrt(3)/sqrt(N) ~= 0.0019, so 0.01 is a ~5-sigma bound.
        assert abs(np.mean(logs)) < 0.01
        # and the draws actually spread over the range, not collapsed
        assert np.std(logs) > 0.1


class TestValidMaskSemantics:
    """The valid mask must reflect the UNSCALED (sensor) depth: threshold
    scales with alpha, so points valid before jitter stay valid after."""

    def test_mask_preserved_under_jitter(self):
        depth = np.array([[0.0, 0.5], [9.9, 10.5]], dtype=np.float32)
        pre_mask = (depth <= 10.0) & (depth > 0)
        K = np.eye(3, dtype=np.float32)
        for _ in range(100):
            d2, _, alpha = apply_jitter(depth, K, 1.6)
            post_mask = (d2 <= 10.0 * alpha) & (d2 > 0)
            np.testing.assert_array_equal(post_mask, pre_mask)

    def test_naive_mask_would_clip_far_points(self):
        # documents WHY the threshold must scale: with alpha > 1 a fixed 10 m
        # ceiling would drop exactly the far points where alpha is informative
        depth = np.full((2, 2), 9.0, dtype=np.float32)
        K = np.eye(3, dtype=np.float32)
        clipped = False
        for _ in range(200):
            d2, _, alpha = apply_jitter(depth, K, 1.6)
            if alpha > 10.0 / 9.0 and not ((d2 <= 10.0) & (d2 > 0)).all():
                clipped = True
                break
        assert clipped


class TestConstructorValidation:
    def test_rejects_m_below_one(self):
        # the validation runs before any file I/O, so no .mat file is needed
        with pytest.raises(ValueError):
            NYUTrainingDataset('does_not_exist.mat', mode='train', focal_jitter=0.5)

    def test_zero_and_one_disable(self):
        # 0 and 1 must not raise in the validation branch (file error comes later)
        for m in (0.0, 1.0):
            with pytest.raises(FileNotFoundError):
                NYUTrainingDataset('does_not_exist.mat', mode='train', focal_jitter=m)
