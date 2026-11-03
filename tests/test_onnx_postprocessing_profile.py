from __future__ import annotations

import unittest

import numpy as np

from benchmark_onnx_postprocessing import _latency_statistics, profile_postprocessing
from edgeai.onnx_detector import _decode_yolo_output


class OnnxPostprocessingProfileTests(unittest.TestCase):
    def test_profile_preserves_detections_and_reports_actual_stages(self):
        # [cx, cy, width, height, person score, bicycle score].
        raw = np.asarray(
            [
                [50, 50, 40, 40, 0.90, 0.01],
                [51, 51, 40, 40, 0.80, 0.01],
                [120, 50, 30, 30, 0.01, 0.88],
            ],
            dtype=np.float32,
        )[None]
        kwargs = {
            "scale": 1.0,
            "pad_left": 0,
            "pad_top": 0,
            "image_width": 160,
            "image_height": 100,
            "names": {0: "person", 1: "bicycle"},
            "confidence": 0.25,
            "iou_threshold": 0.5,
        }

        profile = profile_postprocessing(
            raw,
            **kwargs,
            warmup_runs=1,
            measured_runs=2,
            trials=2,
        )
        expected = _decode_yolo_output(raw, **kwargs)

        self.assertEqual(profile["sample_count"], 4)
        self.assertEqual(profile["reference_detection_count"], 2)
        self.assertTrue(profile["outputs_consistent"])
        self.assertEqual(profile["measured_output_consistency_checks"], 8)
        self.assertEqual(len(profile["raw_samples"]), 4)
        self.assertEqual(len(profile["trial_summaries"]), 2)
        self.assertEqual(
            set(profile["stage_statistics_ms"]),
            {
                "output_conversion_shape_validation_orientation",
                "class_score_selection",
                "confidence_filtering",
                "box_conversion",
                "class_aware_greedy_nms",
                "maximum_detection_cap",
                "coordinate_rescaling_clipping_and_detection_construction",
            },
        )
        self.assertEqual(
            [(d.class_id, d.class_name) for d in expected],
            [(0, "person"), (1, "bicycle")],
        )
        self.assertAlmostEqual(expected[0].confidence, 0.9)
        self.assertEqual(expected[0].bbox_xyxy, (30.0, 30.0, 70.0, 70.0))
        self.assertAlmostEqual(expected[1].confidence, 0.88)
        self.assertEqual(expected[1].bbox_xyxy, (105.0, 35.0, 135.0, 65.0))
        self.assertTrue(
            all(
                row["reference_detection_count"]
                == row["profiled_detection_count"]
                == row["whole_function_detection_count"]
                for row in profile["raw_samples"]
            )
        )

    def test_empty_confidence_filter_path_matches_production(self):
        raw = np.asarray([[[1, 1, 2, 2, 0.01]]], dtype=np.float32)
        profile = profile_postprocessing(
            raw,
            scale=1.0,
            pad_left=0,
            pad_top=0,
            image_width=10,
            image_height=10,
            names={0: "object"},
            confidence=0.25,
            iou_threshold=0.7,
            warmup_runs=1,
            measured_runs=1,
            trials=1,
        )

        self.assertEqual(profile["reference_detection_count"], 0)
        self.assertTrue(profile["outputs_consistent"])
        self.assertEqual(
            profile["stage_statistics_ms"]["class_aware_greedy_nms"]["mean_ms"],
            0.0,
        )

    def test_latency_statistics_reject_invalid_samples(self):
        for samples in ([], [1.0, float("nan")], [1.0, -0.1]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                _latency_statistics(samples)


if __name__ == "__main__":
    unittest.main()
