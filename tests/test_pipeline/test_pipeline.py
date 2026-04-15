"""Basic pipeline integration test.

Requires:
    - A test image/s
    - Aface detection model

Run with:
    poetry run pytest tests/test_pipeline/test_pipeline.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from face_descriptor.core.types import PipelineResult
from face_descriptor.detection.detectors import SCRFDDetector
from face_descriptor.detection import models
from face_descriptor.io.readers import FileImageReader
from face_descriptor.pipeline.pipeline import FaceDescriptorPipeline
from face_descriptor.preprocessing.preprocessors import AffineAlignPreprocessor
from face_descriptor.reporting.reporters import VisualReporter
from tests.conftest import multifaceimage_path

SCRFD_MODEL_PATH = os.path.join(os.path.dirname(models.__file__), "scrfd_2.5g_gnkps.onnx")


@pytest.fixture()
def pipeline() -> FaceDescriptorPipeline:
    return FaceDescriptorPipeline(
        reader=FileImageReader(),
        detector=SCRFDDetector(model_path=SCRFD_MODEL_PATH),
        preprocessor=AffineAlignPreprocessor(),
    )


def test_returns_results(
    pipeline: FaceDescriptorPipeline,
    multifaceimage_path: str,
) -> None:
    results = pipeline.run([multifaceimage_path])
    assert isinstance(results, list)
    assert len(results) > 0


def test_result_structure(
    pipeline: FaceDescriptorPipeline,
    multifaceimage_path: str,
) -> None:
    results = pipeline.run([multifaceimage_path])
    r = results[0]
    assert isinstance(r, PipelineResult)
    assert r.source == str(multifaceimage_path)
    assert r.image is not None
    assert r.preprocessed_face is not None
    assert r.embedding is None
    assert r.face.bbox is not None
    assert r.face.confidence > 0
    assert r.preprocessed_face.data.shape[:2] == (112, 112)


def test_visual_report(
    pipeline: FaceDescriptorPipeline,
    multifaceimage_path: str,
    tmp_path: Path
) -> None:
    results = pipeline.run([multifaceimage_path])
    assert len(results) > 0

    reporter = VisualReporter(save_dir=tmp_path)
    reporter.report(results)

    saved = list(tmp_path.glob("*.png"))
    assert len(saved) == 1
