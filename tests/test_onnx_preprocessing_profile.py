from __future__ import annotations

import unittest

import numpy as np

from benchmark_onnx_preprocessing import (
    _latency_statistics,
    profile_preprocessing,
)
from edgeai.onnx_detector import prepare_onnx_input


class OnnxPreprocessingProfileTests(unittest.TestCase):
    def test_profile_matches_production_tensor_and_reports_all_stages(self):
        image = np.zeros((90, 150, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(150, dtype=np.uint8)
        image[:, :, 1] = 127
        image[:, :, 2] = 240

        profile = profile_preprocessing(
            image,
            input_size=320,
            warmup_runs=1,
            measured_runs=2,
            trials=2,
        )
        tensor, _, _, _ = prepare_onnx_input(image, 320)

        self.assertEqual(profile["sample_count"], 4)
        self.assertTrue(profile["input_contract"]["verified_exact_tensor_equality"])
        self.assertEqual(profile["input_contract"]["tensor_shape"], [1, 3, 320, 320])
        self.assertTrue(profile["input_contract"]["tensor_c_contiguous"])
        self.assertTrue(profile["input_contract"]["tensor_all_finite"])
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(
            set(profile["stage_statistics_ms"]),
            {
                "input_size_validation",
                "letterbox_geometry",
                "resize",
                "border_padding",
                "channel_reversal_transpose_batch_view",
                "contiguous_float32_conversion",
                "in_place_normalization",
            },
        )
        self.assertEqual(len(profile["trial_summaries"]), 2)
        self.assertEqual(len(profile["raw_samples"]), 4)
        self.assertTrue(
            all(
                row["prepare_onnx_input_total_ms"] >= 0
                and row["sum_stage_latencies_ms"] >= 0
                for row in profile["raw_samples"]
            )
        )

    def test_latency_statistics_reject_invalid_samples(self):
        for samples in ([], [1.0, float("nan")], [1.0, -0.1]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                _latency_statistics(samples)

    def test_production_validation_error_is_preserved(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            profile_preprocessing(
                image,
                input_size=500,
                warmup_runs=1,
                measured_runs=1,
                trials=1,
            )


if __name__ == "__main__":
    unittest.main()
