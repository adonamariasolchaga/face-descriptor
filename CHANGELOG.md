# Changelog

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioned following [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-04-15
### Added
- **Core architecture**: protocol-based pipeline with structural typing (`core/protocols.py`, `core/types.py`)
- **Data models**: `Image`, `Face`, `BoundingBox`, `PreprocessedFace`, `Embedding`, `PipelineResult`
- **Image I/O**: `FileImageReader` with BGR→RGB conversion via OpenCV (`io/readers.py`)
- **Face detection**: `SCRFDDetector` running SCRFD models via ONNX Runtime, with auto-detection of model topology, proper input normalization, anchor caching and custom NMS (`detection/detectors.py`)
- **Preprocessing**: `AffineAlignPreprocessor` with 5-point landmark similarity transform and bbox-crop fallback (`preprocessing/preprocessors.py`)
- **Face analyzers** (`analysis/analyzers.py`):
  - `AgeAnalyzer` — age group + numeric estimate ([nateraw/vit-age-classifier](https://huggingface.co/nateraw/vit-age-classifier))
  - `GenderAnalyzer` — perceived gender ([rizvandwiki/gender-classification-2](https://huggingface.co/rizvandwiki/gender-classification-2))
  - `FacialHairAnalyzer` — beard detection ([dima806/beard_face_image_detection](https://huggingface.co/dima806/beard_face_image_detection))
  - `SkinToneAnalyzer` — Fitzpatrick skin type via ITA (model-free)
- **Inference**: `OnnxInferencer` skeleton for embedding extraction (`inference/models.py`)
- **Reporters** (`reporting/reporters.py`):
  - `ConsoleReporter` (skeleton)
  - `JsonReporter` (skeleton)
  - `VisualReporter` — matplotlib figure with original image, detections overlay, and a 6-column grid of preprocessed faces with prediction overlays
- **Pipeline orchestration**: `FaceDescriptorPipeline` composing reader → detector → preprocessor → analyzers → inferencer → reporter, with optional inferencer/reporter (`pipeline/pipeline.py`)
- **Demo script**: `scripts/visual_pipeline.py` for quick visual inspection
- **Test scaffold**: `conftest.py` with shared fixtures, placeholder tests per module, and basic pipeline integration tests
- **Project setup**: Poetry with `poetry-dynamic-versioning`, ruff, pytest

[Unreleased]: https://github.com/usuario/face-descriptor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/usuario/face-descriptor/releases/tag/v0.1.0
