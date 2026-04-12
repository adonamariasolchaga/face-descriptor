"""Shared fixtures for the face_descriptor test suite."""

from __future__ import annotations
import os

import numpy as np
import pytest

from face_descriptor.core.types import (
    BoundingBox,
    Embedding,
    Face,
    Image,
    PreprocessedFace,
)
from face_descriptor.io import(
    FileImageReader
)
from tests import resources

@pytest.fixture()
def synthetic_image() -> Image:
    """A small synthetic 64×64 RGB image."""
    data = np.zeros((64, 64, 3), dtype=np.uint8)
    return Image(data=data, source="test.png")


@pytest.fixture()
def sample_faceimage() -> Image:
    "AN image with a face"
    reader = FileImageReader()
    return reader.read(
        os.path.join(os.path.dirname(resources.__file__), "face_images/single_face_1.jpeg")
    )


@pytest.fixture()
def multifaceimage_path() -> str:
    return os.path.join(os.path.dirname(resources.__file__), "face_images/multiple_face_1.jpg")


@pytest.fixture()
def sample_multifaceimage() -> Image:
    "AN image with a face"
    reader = FileImageReader()
    return reader.read(
        os.path.join(os.path.dirname(resources.__file__), "face_images/multiple_face_1.jpg")
    )


@pytest.fixture()
def sample_face() -> Face:
    """A face bounding box covering the top-left quadrant."""
    return Face(
        bbox=BoundingBox(x=0, y=0, w=32, h=32),
        confidence=0.99,
    )


@pytest.fixture()
def sample_preprocessed_face(sample_face: Face) -> PreprocessedFace:
    """A 112×112 normalised face tensor."""
    data = np.zeros((112, 112, 3), dtype=np.float32)
    return PreprocessedFace(data=data, original_face=sample_face)


@pytest.fixture()
def sample_embedding() -> Embedding:
    """A 512-d zero embedding."""
    return Embedding(vector=np.zeros(512, dtype=np.float32))
