"""Protocol definitions for each pipeline stage.

Using :class:`typing.Protocol` provides structural sub-typing so that any
class implementing the right method signatures is accepted — no inheritance
required.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from face_descriptor.core.types import (
    Embedding,
    Face,
    Image,
    PipelineResult,
    PreprocessedFace,
)


class ImageReader(Protocol):
    """Reads an image from a given source (file path, URL, …)."""

    def read(self, source: str) -> Image: ...


class FaceDetector(Protocol):
    """Detects faces in an image and returns their locations."""

    def detect(self, image: Image) -> Sequence[Face]: ...


class Preprocessor(Protocol):
    """Prepares a detected face crop for model inference."""

    def preprocess(self, image: Image, face: Face) -> PreprocessedFace: ...


class Inferencer(Protocol):
    """Runs a model on a preprocessed face and returns an embedding."""

    def infer(self, face: PreprocessedFace) -> Embedding: ...


class FaceAnalyzer(Protocol):
    """Extracts facial attributes from a preprocessed face.

    Implementations should return a dictionary of attribute names to
    predicted values (e.g. ``{"age": 25, "gender": "male"}``).
    """

    def analyze(self, face: PreprocessedFace) -> dict[str, object]: ...


class Reporter(Protocol):
    """Outputs or persists a collection of pipeline results."""

    def report(self, results: Sequence[PipelineResult]) -> None: ...
