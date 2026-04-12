"""Shared data models used across all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class Image:
    """Raw image loaded from a source."""

    data: NDArray[np.uint8]
    source: str  # origin path or URL


@dataclass
class Face:
    """A detected face region within an image."""

    bbox: BoundingBox
    confidence: float
    landmarks: NDArray[np.float32] | None = None


@dataclass
class PreprocessedFace:
    """A face after alignment, cropping and normalization."""

    data: NDArray[np.float32]
    original_face: Face


@dataclass
class Embedding:
    """Feature vector produced by the inference model."""

    vector: NDArray[np.float32]


@dataclass
class PipelineResult:
    """End-to-end result for a single detected face."""

    source: str
    face: Face
    embedding: Optional[Embedding] = None
    image: Optional[Image] = None
    preprocessed_face: Optional[PreprocessedFace] = None
    metadata: dict[str, object] = field(default_factory=dict)
