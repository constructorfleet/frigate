"""Tests for CameraActivityManager zone label and attribute MQTT publishing."""

import sys
import unittest
from unittest.mock import MagicMock


# Only mock modules that are genuinely unavailable in the current environment.
# Critically, we must NEVER permanently replace frigate.config or other core
# modules that other test files depend on — doing so would corrupt the entire
# test process (alphabetical discovery means this file runs first).
#
# In the CI devcontainer all deps (zmq, etc.) are present, so no mocking is
# needed there.  For lightweight local runs, only mock the two leaf modules
# that have C-extension requirements, and only when they are not already
# importable.
def _maybe_mock(module_name: str) -> None:
    """Insert a MagicMock stub for *module_name* only when it cannot be imported."""
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
    except ImportError:
        sys.modules[module_name] = MagicMock()


_maybe_mock("zmq")
_maybe_mock("frigate.comms.zmq_proxy")
_maybe_mock("frigate.comms.event_metadata_updater")

from frigate.camera.activity_manager import CameraActivityManager  # noqa: E402


def _make_config(zone_name="driveway", zone_objects=None, track_objects=None):
    """Build a minimal mock FrigateConfig with one camera and one zone."""
    if zone_objects is None:
        zone_objects = ["person", "car"]
    if track_objects is None:
        track_objects = ["person", "car"]

    zone_config = MagicMock()
    zone_config.objects = zone_objects

    camera_config = MagicMock()
    camera_config.name = "front"
    camera_config.enabled_in_config = True
    camera_config.zones = {zone_name: zone_config}
    camera_config.objects.track = track_objects

    config = MagicMock()
    config.cameras = {"front": camera_config}
    config.model.non_logo_attributes = ["face", "license_plate"]

    return config


def _make_object(
    obj_id,
    object_type,
    label,
    sub_label=None,
    stationary=False,
    current_zones=None,
):
    """Build a minimal activity object dict matching the structure from state.py."""
    return {
        "id": obj_id,
        "object_type": object_type,
        "label": label,
        "stationary": stationary,
        "area": 10000,
        "ratio": 1.0,
        "score": 0.9,
        "sub_label": sub_label,
        "current_zones": current_zones or [],
    }


class TestZoneLabelPublishing(unittest.TestCase):
    """Tests that custom classification sub-label counts are published to
    {zone}/{object_type}/label/{sub_label} and {zone}/{object_type}/label/{sub_label}/active.
    """

    def setUp(self):
        self.publish = MagicMock()
        self.manager = CameraActivityManager(_make_config(), self.publish)

    def test_sub_label_total_published(self):
        """Publish {zone}/{object_type}/label/{sub_label} for a classified object."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person-verified",
                        sub_label="running",
                        stationary=True,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        self.publish.assert_any_call("driveway/person/label/running", 1)

    def test_sub_label_active_published(self):
        """Publish {zone}/{object_type}/label/{sub_label}/active for active classified object."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person-verified",
                        sub_label="running",
                        stationary=False,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        self.publish.assert_any_call("driveway/person/label/running", 1)
        self.publish.assert_any_call("driveway/person/label/running/active", 1)

    def test_sub_label_active_zero_when_stationary(self):
        """Active count is 0 when classified object is stationary."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person-verified",
                        sub_label="walking",
                        stationary=True,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        self.publish.assert_any_call("driveway/person/label/walking", 1)
        self.publish.assert_any_call("driveway/person/label/walking/active", 0)

    def test_sub_label_count_published_zero_when_object_leaves(self):
        """Count drops to 0 and is published when a classified object leaves the zone."""
        activity_with = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person-verified",
                        sub_label="running",
                        stationary=False,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        activity_without = {"front": {"motion": False, "objects": []}}

        self.manager.update_activity(activity_with)
        self.publish.reset_mock()
        self.manager.update_activity(activity_without)

        self.publish.assert_any_call("driveway/person/label/running", 0)
        self.publish.assert_any_call("driveway/person/label/running/active", 0)

    def test_multiple_sub_labels_counted_separately(self):
        """Different sub_labels under the same object_type are counted separately."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person-verified",
                        sub_label="running",
                        stationary=False,
                        current_zones=["driveway"],
                    ),
                    _make_object(
                        "obj2",
                        "person",
                        "person-verified",
                        sub_label="sitting",
                        stationary=True,
                        current_zones=["driveway"],
                    ),
                ],
            }
        }
        self.manager.update_activity(activity)

        self.publish.assert_any_call("driveway/person/label/running", 1)
        self.publish.assert_any_call("driveway/person/label/running/active", 1)
        self.publish.assert_any_call("driveway/person/label/sitting", 1)
        self.publish.assert_any_call("driveway/person/label/sitting/active", 0)

    def test_sub_label_not_published_for_object_outside_zone(self):
        """Sub-label counts are not published for objects outside the zone."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person-verified",
                        sub_label="running",
                        stationary=False,
                        current_zones=[],  # not in any zone
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        published_topics = [c.args[0] for c in self.publish.call_args_list]
        self.assertNotIn("driveway/person/label/running", published_topics)
        self.assertNotIn("driveway/person/label/running/active", published_topics)


class TestZoneAttributePublishing(unittest.TestCase):
    """Tests that attribute counts are published to
    {zone}/{object_type}/attribute/{attribute} and
    {zone}/{object_type}/attribute/{attribute}/active.
    """

    def setUp(self):
        self.publish = MagicMock()
        self.manager = CameraActivityManager(_make_config(), self.publish)

    def test_attribute_total_published(self):
        """Publish {zone}/{object_type}/attribute/{attribute} for an attribute object."""
        # Attributes: label IS the attribute (e.g. "amazon"), object_type is the parent ("car")
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "car",
                        "amazon",
                        sub_label=None,
                        stationary=True,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        self.publish.assert_any_call("driveway/car/attribute/amazon", 1)

    def test_attribute_active_published(self):
        """Publish {zone}/{object_type}/attribute/{attribute}/active for active attribute object."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "car",
                        "amazon",
                        sub_label=None,
                        stationary=False,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        self.publish.assert_any_call("driveway/car/attribute/amazon", 1)
        self.publish.assert_any_call("driveway/car/attribute/amazon/active", 1)

    def test_attribute_active_zero_when_stationary(self):
        """Active count is 0 when attribute object is stationary."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "car",
                        "license_plate",
                        sub_label=None,
                        stationary=True,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        self.publish.assert_any_call("driveway/car/attribute/license_plate", 1)
        self.publish.assert_any_call("driveway/car/attribute/license_plate/active", 0)

    def test_attribute_count_drops_to_zero_when_object_leaves(self):
        """Count drops to 0 and is published when an attribute object leaves the zone."""
        activity_with = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "car",
                        "amazon",
                        sub_label=None,
                        stationary=False,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        activity_without = {"front": {"motion": False, "objects": []}}

        self.manager.update_activity(activity_with)
        self.publish.reset_mock()
        self.manager.update_activity(activity_without)

        self.publish.assert_any_call("driveway/car/attribute/amazon", 0)
        self.publish.assert_any_call("driveway/car/attribute/amazon/active", 0)

    def test_base_object_not_treated_as_attribute(self):
        """A plain object without sub_label whose label == object_type is NOT published as attribute."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person",
                        sub_label=None,
                        stationary=False,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)

        published_topics = [c.args[0] for c in self.publish.call_args_list]
        attribute_topics = [t for t in published_topics if "/attribute/" in t]
        self.assertEqual([], attribute_topics)


class TestNoRepublishUnchanged(unittest.TestCase):
    """Tests that counts are only re-published when they actually change."""

    def setUp(self):
        self.publish = MagicMock()
        self.manager = CameraActivityManager(_make_config(), self.publish)

    def test_no_republish_when_sub_label_count_unchanged(self):
        """Sub-label topics are not re-published when the counts haven't changed."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "person",
                        "person-verified",
                        sub_label="running",
                        stationary=False,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)
        self.publish.reset_mock()

        # Same activity again — counts are identical, nothing should be re-published
        self.manager.update_activity(activity)

        published_topics = [c.args[0] for c in self.publish.call_args_list]
        self.assertNotIn("driveway/person/label/running", published_topics)
        self.assertNotIn("driveway/person/label/running/active", published_topics)

    def test_no_republish_when_attribute_count_unchanged(self):
        """Attribute topics are not re-published when the counts haven't changed."""
        activity = {
            "front": {
                "motion": False,
                "objects": [
                    _make_object(
                        "obj1",
                        "car",
                        "amazon",
                        sub_label=None,
                        stationary=False,
                        current_zones=["driveway"],
                    )
                ],
            }
        }
        self.manager.update_activity(activity)
        self.publish.reset_mock()

        self.manager.update_activity(activity)

        published_topics = [c.args[0] for c in self.publish.call_args_list]
        self.assertNotIn("driveway/car/attribute/amazon", published_topics)
        self.assertNotIn("driveway/car/attribute/amazon/active", published_topics)
