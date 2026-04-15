"""Quick script to run the pipeline and display visual results.

Usage:
    poetry run python scripts/visual_pipeline.py <image_path> [--device cpu|cuda]

Runs age, gender, glasses, facial-hair and skin-tone analyzers and
displays the predictions in the visual report.
"""

from __future__ import annotations

import argparse
import logging
import sys

from face_descriptor.analysis.analyzers import (
    AgeAnalyzer,
    FacialHairAnalyzer,
    GenderAnalyzer,
    SkinToneAnalyzer,
)
from face_descriptor.detection.detectors import SCRFDDetector
from face_descriptor.io.readers import FileImageReader
from face_descriptor.pipeline.pipeline import FaceDescriptorPipeline
from face_descriptor.preprocessing.preprocessors import AffineAlignPreprocessor
from face_descriptor.reporting.reporters import VisualReporter

logger = logging.getLogger(__name__)


def _build_analyzers(device: str = "cpu") -> list:
    """Instantiate all face attribute analyzers."""
    return [
        AgeAnalyzer(device=device),
        GenderAnalyzer(device=device),
        FacialHairAnalyzer(device=device),
        SkinToneAnalyzer(),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the face-descriptor pipeline and display visual results."
    )
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device for the attribute models (default: cpu).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="If set, save figures to this directory instead of displaying them.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-36s  %(levelname)-8s  %(message)s",
    )

    args = _parse_args()

    analyzers = _build_analyzers(device=args.device)
    if analyzers:
        logger.info("Loaded %d face analyzer(s).", len(analyzers))
    else:
        logger.info("Running without face analyzers (detection + visualisation only).")

    pipeline = FaceDescriptorPipeline(
        reader=FileImageReader(),
        detector=SCRFDDetector(),
        preprocessor=AffineAlignPreprocessor(),
        analyzers=analyzers or None,
        reporter=VisualReporter(save_dir=args.save_dir),
    )
    pipeline.run([args.image_path])


if __name__ == "__main__":
    main()
