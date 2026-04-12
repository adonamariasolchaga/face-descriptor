"""Quick script to run the pipeline and display visual results.

Usage:
    poetry run python scripts/run_pipeline.py <image_path> <model_path>
"""

from __future__ import annotations

import sys

from face_descriptor.detection.detectors import SCRFDDetector
from face_descriptor.io.readers import FileImageReader
from face_descriptor.pipeline.pipeline import FaceDescriptorPipeline
from face_descriptor.preprocessing.preprocessors import AffineAlignPreprocessor
from face_descriptor.reporting.reporters import VisualReporter


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    pipeline = FaceDescriptorPipeline(
        reader=FileImageReader(),
        detector=SCRFDDetector(),
        preprocessor=AffineAlignPreprocessor(),
        reporter=VisualReporter(),
    )
    pipeline.run([image_path])


if __name__ == "__main__":
    main()
