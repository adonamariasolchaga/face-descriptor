# face_descriptor

## Requirements
- Python 3.11+
- [Poetry](https://python-poetry.org/) (`pipx install poetry`)
- Create virtual environment in project folder (`poetry config virtualenvs.in-project true`)
- Versioning plugin: `poetry self add poetry-dynamic-versioning`

## Development

```bash
# Install dependencies
poetry install --with dev

# Add a new production dependency
poetry add <package-name>

# Add a new development dependency
poetry add --group dev <package-name>

# Update all dependencies
poetry update

# Update a specific dependency
poetry update <package-name>

# Run tests
poetry run pytest

# Lint
poetry run ruff check src/ tests/

# Format code
poetry run ruff format src/ tests/
```

## Publishing

```bash
# Create tag (version is read from tag via poetry-dynamic-versioning)
git tag v1.0.0
git push origin v1.0.0

# Manual build
poetry build

# Publish to PyPI
poetry publish

# Publish to private registry (configured in pyproject.toml)
poetry publish --repository private
```

## Core models

The project uses [SCRFD](https://arxiv.org/abs/2105.04714) (Sample and Computation Redistribution for Face Detection) as its primary face detector, running via ONNX Runtime.

Download a pretrained ONNX model from the [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/detection/scrfd#pretrained-models). The `_KPS` variants include 5-point landmark prediction:

| Model | Easy | Medium | Hard | FLOPs | Link |
|-------|------|--------|------|-------|------|
| SCRFD_2.5G_KPS | 93.80 | 92.02 | 77.13 | 2.5G | [download](https://github.com/deepinsight/insightface/releases/download/v0.4/scrfd_2.5g_bnkps.onnx) |
| SCRFD_10G_KPS | 95.40 | 94.01 | 82.80 | 10G | [download](https://github.com/deepinsight/insightface/releases/download/v0.4/scrfd_10g_bnkps.onnx) |

Place the `.onnx` file under `models/` and pass the path to `SCRFDDetector(model_path=...)`.

## Face Analyzers

After detection and preprocessing, faces are passed through a set of attribute analyzers. Each one implements the `FaceAnalyzer` protocol and returns predictions merged into `PipelineResult.metadata`.

### HuggingFace-based analyzers

| Analyzer | Attribute | Model |
|----------|-----------|-------|
| `AgeAnalyzer` | Age group + numeric estimate | [nateraw/vit-age-classifier](https://huggingface.co/nateraw/vit-age-classifier) |
| `GenderAnalyzer` | Perceived gender | [rizvandwiki/gender-classification-2](https://huggingface.co/rizvandwiki/gender-classification-2) |
| `FacialHairAnalyzer` | Beard / facial hair | [dima806/beard_face_image_detection](https://huggingface.co/dima806/beard_face_image_detection) |

All HuggingFace models are loaded lazily on first use and support `device="cpu"` or `device="cuda"`.

### Algorithmic analyzers

| Analyzer | Attribute | Method |
|----------|-----------|--------|
| `SkinToneAnalyzer` | Fitzpatrick skin type (I–VI) | ITA (Individual Typology Angle) computed from CIE-Lab colour space — no model required |


## Quick Start

The `scripts/visual_pipeline.py` script runs the full pipeline (detection → preprocessing → analysis → visual report) on a single image:

```bash
# Display results interactively
poetry run python scripts/visual_pipeline.py path/to/image.jpg

# Use GPU for HuggingFace analyzers
poetry run python scripts/visual_pipeline.py path/to/image.jpg --device cuda

# Save figure to disk instead of displaying
poetry run python scripts/visual_pipeline.py path/to/image.jpg --save-dir outputs/
```

## Contributing

The project follows a protocol-based architecture — each pipeline stage is defined by a `Protocol` in `core/protocols.py`. To add a new implementation you only need to create a class that satisfies the matching method signature. No inheritance required.

### Adding a new image reader (`io/readers.py`)

Implement the `read(self, source: str) -> Image` method. The returned `Image.data` must be **RGB `uint8` `(H, W, 3)`**.

```python
class UrlImageReader:
    def read(self, source: str) -> Image:
        # fetch from URL, decode, ensure RGB uint8
        ...
```

### Adding a new face detector (`detection/detectors.py`)

Implement `detect(self, image: Image) -> Sequence[Face]`. Return a `Face` with a `BoundingBox` and optionally 5-point `landmarks` as a `(5, 2)` float32 array.

```python
class RetinaFaceDetector:
    def detect(self, image: Image) -> Sequence[Face]:
        ...
```

### Adding a new preprocessor (`preprocessing/preprocessors.py`)

Implement `preprocess(self, image: Image, face: Face) -> PreprocessedFace`. Output should be a float32 tensor normalised to `[0, 1]`.

```python
class CropResizePreprocessor:
    def preprocess(self, image: Image, face: Face) -> PreprocessedFace:
        ...
```

### Adding a new analyzer (`analysis/analyzers.py`)

Implement `analyze(self, face: PreprocessedFace) -> dict[str, object]`. The returned dict is merged into `PipelineResult.metadata`. For HuggingFace image-classification models, subclass `_HuggingFaceClassifierBase` for free lazy-loading and PIL conversion.

```python
class EmotionAnalyzer(_HuggingFaceClassifierBase):
    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__("some-org/emotion-model", device=device)

    def analyze(self, face: PreprocessedFace) -> dict[str, object]:
        preds = self._classify(face, top_k=3)
        return {"emotion": preds[0]["label"], "emotion_confidence": preds[0]["score"]}
```

### Adding a new inferencer (`inference/models.py`)

Implement `infer(self, face: PreprocessedFace) -> Embedding`. Return an `Embedding` with a float32 feature vector.

```python
class ArcFaceInferencer:
    def infer(self, face: PreprocessedFace) -> Embedding:
        ...
```

### Adding a new reporter (`reporting/reporters.py`)

Implement `report(self, results: Sequence[PipelineResult]) -> None`.

```python
class CsvReporter:
    def report(self, results: Sequence[PipelineResult]) -> None:
        ...
```

### Wiring it up

Pass your new components to `FaceDescriptorPipeline`:

```python
pipeline = FaceDescriptorPipeline(
    reader=YourReader(),
    detector=YourDetector(),
    preprocessor=YourPreprocessor(),
    analyzers=[YourAnalyzer()],
    inferencer=YourInferencer(),     # optional
    reporter=YourReporter(),         # optional
)
pipeline.run(["image1.jpg", "image2.jpg"])
```

### Tests

Place tests under `tests/test_<module>/` mirroring the source layout. Run the full suite with:

```bash
poetry run pytest
```


## Structure

```
face_descriptor/
├── src/face_descriptor/
│   ├── __init__.py                  # package version
│   │
│   ├── core/                        # abstractions & shared types
│   │   ├── __init__.py
│   │   ├── protocols.py             # Protocol classes: ImageReader, FaceDetector,
│   │   │                            #   Preprocessor, Inferencer, Reporter
│   │   └── types.py                 # shared data models (Image, Face, BoundingBox,
│   │                                #   Embedding, PipelineResult, etc.)
│   │
│   ├── io/                          # image reading stage
│   │   ├── __init__.py
│   │   └── readers.py               # FileImageReader
│   │
│   ├── detection/                   # face detection stage
│   │   ├── __init__.py
│   │   └── detectors.py             # SCRFDDetector (ONNX Runtime)
│   │
│   ├── preprocessing/               # face preprocessing stage
│   │   ├── __init__.py
│   │   └── preprocessors.py         # AffineAlignPreprocessor
│   │
│   ├── analysis/                    # face attribute analysis
│   │   ├── __init__.py
│   │   └── analyzers.py             # AgeAnalyzer, GenderAnalyzer, FacialHairAnalyzer,
│   │                                #   SkinToneAnalyzer
│   │
│   ├── inference/                   # model inference stage
│   │   ├── __init__.py
│   │   └── models.py                # OnnxInferencer
│   │
│   ├── reporting/                   # results reporting stage
│   │   ├── __init__.py
│   │   └── reporters.py             # ConsoleReporter, JsonReporter, VisualReporter
│   │
│   └── pipeline/                    # orchestration
│       ├── __init__.py
│       └── pipeline.py              # FaceDescriptorPipeline
│
├── tests/
│   ├── conftest.py
│   ├── test_io/
│   ├── test_detection/
│   ├── test_preprocessing/
│   ├── test_inference/
│   ├── test_reporting/
│   └── test_pipeline/
│
├── scripts/
│   └── visual_pipeline.py           # quick demo script
│
├── models/                          # ONNX model files (not committed)
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```
