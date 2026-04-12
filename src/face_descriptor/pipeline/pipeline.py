"""Pipeline orchestration — composes all stages into a single workflow."""

from __future__ import annotations

import logging
from typing import Sequence

from face_descriptor.core.protocols import (
    FaceDetector,
    ImageReader,
    Inferencer,
    Preprocessor,
    Reporter,
)
from face_descriptor.core.types import PipelineResult

logger = logging.getLogger(__name__)


class FaceDescriptorPipeline:
    """Orchestrates the full face_descriptor pipeline.

    Parameters
    ----------
    reader:
        Reads images from a source identifier.
    detector:
        Detects faces in an image.
    preprocessor:
        Prepares a detected face for model inference.
    inferencer:
        Produces an embedding from a preprocessed face. Optional.
    reporter:
        Outputs or persists the final results. Optional.
    """

    def __init__(
        self,
        reader: ImageReader,
        detector: FaceDetector,
        preprocessor: Preprocessor,
        inferencer: Inferencer | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self._reader = reader
        self._detector = detector
        self._preprocessor = preprocessor
        self._inferencer = inferencer
        self._reporter = reporter

    def run(self, sources: Sequence[str]) -> list[PipelineResult]:
        """Execute the pipeline over one or more image sources.

        Parameters
        ----------
        sources:
            Iterable of source identifiers (file paths, URLs, …).

        Returns
        -------
        list[PipelineResult]
            One result per detected face across all sources.
        """
        results: list[PipelineResult] = []

        for src in sources:
            logger.info("Processing source: %s", src)
            image = self._reader.read(src)

            faces = self._detector.detect(image)
            logger.info("Detected %d face(s) in %s", len(faces), src)

            for face in faces:
                preprocessed = self._preprocessor.preprocess(image, face)
                embedding = None
                if self._inferencer is not None:
                    embedding = self._inferencer.infer(preprocessed)
                results.append(
                    PipelineResult(
                        source=src,
                        face=face,
                        embedding=embedding,
                        image=image,
                        preprocessed_face=preprocessed,
                    )
                )

        if self._reporter is not None:
            self._reporter.report(results)
        logger.info("Pipeline finished — %d result(s) reported.", len(results))
        return results
