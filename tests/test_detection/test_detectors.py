"""Tests for face detector implementations."""

import os
import pytest

from tests.conftest import sample_multifaceimage
from face_descriptor.detection import (
    SCRFDDetector
)
from face_descriptor.detection import models

def test_load_scrfd_detector():
    detector = SCRFDDetector(
        model_path=os.path.join(os.path.dirname(models.__file__), "scrfd_2.5g_gnkps.onnx")
    )


def test_scrfd_inference(sample_multifaceimage):
    detector = SCRFDDetector(
        model_path=os.path.join(os.path.dirname(models.__file__), "scrfd_2.5g_gnkps.onnx"),
        conf_threshold=0.5
    )
    detections = detector.detect(
        image=sample_multifaceimage
    )
    # 4 faces in multiple_face_1.jpg
    assert len(detections) == 4
