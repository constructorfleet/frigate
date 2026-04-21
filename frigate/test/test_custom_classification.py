import json
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Mock native/optional modules before any Frigate imports so the module can be
# imported in environments where cv2 or TFLite are unavailable (e.g. CI).
_SYS_MOCKS = [
    "cv2",
    "tflite_runtime",
    "tflite_runtime.interpreter",
    "ai_edge_litert",
    "ai_edge_litert.interpreter",
]
for _mod in _SYS_MOCKS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

WIDTH = 720
HEIGHT = 1280


class Contains:
    def __init__(self, needle):
        self.needle = needle

    def __eq__(self, other):
        return self.needle in other


class TestCustomObjectClassificationZones(unittest.TestCase):
    """Test that zone information is correctly added to custom classification MQTT messages"""

    def _build_classification_data(
        self, obj_data, classification_type="sub_label", label="person_walking"
    ):
        """Helper method to build classification data with conditional zones.

        Args:
            obj_data: Object data dictionary containing id, camera, and optionally current_zones
            classification_type: Either "sub_label" or "attribute"
            label: The classification label

        Returns:
            Dictionary with classification data, including zones if applicable
        """
        classification_data = {
            "type": "classification",
            "id": obj_data["id"],
            "camera": obj_data["camera"],
            "timestamp": 1234567890.0,
            "model": "test_classifier",
            "score": 0.89,
        }

        if classification_type == "sub_label":
            classification_data["sub_label"] = label
        else:
            classification_data["attribute"] = label

        if obj_data.get("current_zones"):
            classification_data["zones"] = obj_data["current_zones"]

        return classification_data

    def test_sub_label_message_includes_zones_when_present(self):
        """Test that zones are included in sub_label classification messages when object is in zones"""
        # Create a simple mock requestor
        requestor = MagicMock()

        # Create mock obj_data with zones
        obj_data = {
            "id": "test_object_123",
            "camera": "front_door",
            "current_zones": ["driveway", "front_yard"],
        }

        # Build classification data using helper
        classification_data = self._build_classification_data(
            obj_data, "sub_label", "person_walking"
        )

        requestor.send_data("tracked_object_update", json.dumps(classification_data))

        # Verify that send_data was called
        requestor.send_data.assert_called_once()

        # Get the actual call arguments
        call_args = requestor.send_data.call_args
        topic = call_args[0][0]
        data_json = call_args[0][1]

        # Verify the topic
        self.assertEqual(topic, "tracked_object_update")

        # Parse and verify the data
        data = json.loads(data_json)
        self.assertEqual(data["type"], "classification")
        self.assertEqual(data["id"], "test_object_123")
        self.assertEqual(data["camera"], "front_door")
        self.assertEqual(data["model"], "test_classifier")
        self.assertEqual(data["sub_label"], "person_walking")
        self.assertIn("zones", data)
        self.assertEqual(data["zones"], ["driveway", "front_yard"])

    def test_sub_label_message_excludes_zones_when_empty(self):
        """Test that zones are not included when object is not in any zones"""
        requestor = MagicMock()

        # Create mock obj_data without zones
        obj_data = {
            "id": "test_object_456",
            "camera": "back_door",
            "current_zones": [],
        }

        # Build classification data using helper
        classification_data = self._build_classification_data(
            obj_data, "sub_label", "person_running"
        )
        classification_data["score"] = 0.87

        requestor.send_data("tracked_object_update", json.dumps(classification_data))

        # Get the actual call arguments
        call_args = requestor.send_data.call_args
        data_json = call_args[0][1]

        # Parse and verify the data
        data = json.loads(data_json)
        self.assertNotIn("zones", data)

    def test_attribute_message_includes_zones_when_present(self):
        """Test that zones are included in attribute classification messages when object is in zones"""
        requestor = MagicMock()

        # Create mock obj_data with zones
        obj_data = {
            "id": "test_object_789",
            "camera": "construction_site",
            "current_zones": ["site_entrance"],
        }

        # Build classification data using helper
        classification_data = self._build_classification_data(
            obj_data, "attribute", "wearing_helmet"
        )
        classification_data["score"] = 0.92
        classification_data["model"] = "helmet_detector"

        requestor.send_data("tracked_object_update", json.dumps(classification_data))

        # Get the actual call arguments
        call_args = requestor.send_data.call_args
        data_json = call_args[0][1]

        # Parse and verify the data
        data = json.loads(data_json)
        self.assertEqual(data["type"], "classification")
        self.assertEqual(data["id"], "test_object_789")
        self.assertEqual(data["camera"], "construction_site")
        self.assertEqual(data["model"], "helmet_detector")
        self.assertEqual(data["attribute"], "wearing_helmet")
        self.assertIn("zones", data)
        self.assertEqual(data["zones"], ["site_entrance"])

    def test_attribute_message_excludes_zones_when_missing(self):
        """Test that zones are not included when current_zones key is missing"""
        requestor = MagicMock()

        # Create mock obj_data without current_zones key
        obj_data = {
            "id": "test_object_999",
            "camera": "parking_lot",
        }

        # Build classification data using helper
        classification_data = self._build_classification_data(
            obj_data, "attribute", "sedan"
        )
        classification_data["score"] = 0.95
        classification_data["model"] = "vehicle_type"

        requestor.send_data("tracked_object_update", json.dumps(classification_data))

        # Get the actual call arguments
        call_args = requestor.send_data.call_args
        data_json = call_args[0][1]

        # Parse and verify the data
        data = json.loads(data_json)
        self.assertNotIn("zones", data)


class TestCustomObjectClassificationIntegration(unittest.TestCase):
    """
    Integration tests that call process_frame() on the actual processor.
    These tests exercise the full pipeline from process_frame() through the
    deferred worker thread to drain_results(), verifying that zone information
    is carried through to the result dicts that the maintainer publishes.
    """

    def setUp(self):
        import numpy as np

        self.np = np

        # Import the module first so patch() can resolve its attributes.
        try:
            import frigate.data_processing.real_time.custom_classification  # noqa: F401
            from frigate.data_processing.real_time.custom_classification import (
                CustomObjectClassificationProcessor,
            )
        except ImportError as e:
            self.skipTest(f"Requires full Frigate environment: {e}")
            return

        self.ProcessorClass = CustomObjectClassificationProcessor

        # Patch out heavy I/O helpers on the already-imported module object.
        for target in [
            "frigate.data_processing.real_time.custom_classification.write_classification_attempt",
            "frigate.data_processing.real_time.custom_classification.suppress_stderr_during",
        ]:
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _make_processor(self, classification_type):
        """Return a processor with a live interpreter stub and pre-loaded history."""
        config = MagicMock()
        model_config = MagicMock()
        model_config.name = "test_model"
        model_config.threshold = 0.7
        model_config.save_attempts = 100
        model_config.object_config.objects = ["person"]
        model_config.object_config.classification_type = classification_type

        sub_label_publisher = MagicMock()
        requestor = MagicMock()
        metrics = MagicMock()
        metrics.classification_speeds = {}
        metrics.classification_cps = {}

        with patch.object(
            self.ProcessorClass,
            "_CustomObjectClassificationProcessor__build_detector",
        ):
            processor = self.ProcessorClass(
                config, model_config, sub_label_publisher, requestor, metrics
            )

        return processor

    def _run_and_drain(self, processor, obj_data, label, score=0.92):
        """
        Run process_frame() with a stubbed interpreter and return drain_results().
        Pre-loads 3 identical history entries so consensus is reached immediately.
        """
        import numpy as np

        # Pre-load history so get_weighted_score returns consensus on the first call.
        processor.classification_history[obj_data["id"]] = [
            (label, score, 1234567890.0),
            (label, score, 1234567891.0),
            (label, score, 1234567892.0),
        ]

        processor.tensor_input_details = [{"index": 0}]
        processor.tensor_output_details = [{"index": 0}]
        processor.labelmap = {0: label}

        mock_interp = MagicMock()
        mock_interp.get_tensor.return_value = np.array([[score, 1.0 - score]])
        processor.interpreter = mock_interp

        frame = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
        processor.process_frame(obj_data, frame)

        # Give the worker thread time to process the enqueued task.
        time.sleep(0.2)

        return processor.drain_results()

    def test_process_frame_with_zones_includes_zones_in_mqtt(self):
        """process_frame() with non-empty current_zones must emit a result with zones."""
        from frigate.config.classification import ObjectClassificationType

        processor = self._make_processor(ObjectClassificationType.sub_label)

        obj_data = {
            "id": "test_123",
            "camera": "front_door",
            "label": "person",
            "false_positive": False,
            "end_time": None,
            "box": [100, 100, 200, 200],
            "current_zones": ["driveway", "porch"],
        }

        results = self._run_and_drain(processor, obj_data, "walking")

        self.assertTrue(results, "process_frame must produce at least one result")
        result = results[0]
        self.assertEqual(result["type"], "classification")
        self.assertIn(
            "zones", result, "Result must include zones when object is in zones"
        )
        self.assertEqual(result["zones"], ["driveway", "porch"])
        self.assertEqual(result["label"], "walking")

    def test_process_frame_without_zones_excludes_zones_from_mqtt(self):
        """process_frame() with empty current_zones must emit a result without zones."""
        from frigate.config.classification import ObjectClassificationType

        processor = self._make_processor(ObjectClassificationType.sub_label)

        obj_data = {
            "id": "test_456",
            "camera": "backyard",
            "label": "person",
            "false_positive": False,
            "end_time": None,
            "box": [150, 150, 250, 250],
            "current_zones": [],
        }

        results = self._run_and_drain(processor, obj_data, "running")

        self.assertTrue(results, "process_frame must produce at least one result")
        result = results[0]
        self.assertNotIn("zones", result, "Empty zones should be excluded from result")

    def test_process_frame_attribute_type_includes_zones(self):
        """process_frame() with attribute classification type must include zones."""
        from frigate.config.classification import ObjectClassificationType

        processor = self._make_processor(ObjectClassificationType.attribute)

        obj_data = {
            "id": "test_789",
            "camera": "garage",
            "label": "person",
            "false_positive": False,
            "end_time": None,
            "box": [200, 200, 300, 300],
            "current_zones": ["parking_lot"],
        }

        results = self._run_and_drain(processor, obj_data, "hat")

        self.assertTrue(results, "process_frame must produce at least one result")
        result = results[0]
        self.assertIn("zones", result, "Result must include zones for attribute type")
        self.assertEqual(result["zones"], ["parking_lot"])
        self.assertEqual(result["label"], "hat")


if __name__ == "__main__":
    unittest.main()
