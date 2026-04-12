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
│   │   └── readers.py               # FileImageReader, UrlImageReader, etc.
│   │
│   ├── detection/                   # face detection stage
│   │   ├── __init__.py
│   │   └── detectors.py             # e.g. MediaPipeDetector, RetinaFaceDetector
│   │
│   ├── preprocessing/               # face preprocessing stage
│   │   ├── __init__.py
│   │   └── preprocessors.py         # alignment, cropping, normalization, etc.
│   │
│   ├── inference/                   # model inference stage
│   │   ├── __init__.py
│   │   └── models.py                # ArcFaceModel, ONNXModel, etc.
│   │
│   ├── reporting/                   # results reporting stage
│   │   ├── __init__.py
│   │   └── reporters.py             # JsonReporter, CsvReporter, ConsoleReporter
│   │
│   └── pipeline/                    # orchestration
│       ├── __init__.py
│       └── pipeline.py              # FaceDescriptorPipeline (composes all stages)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # shared fixtures (sample images, mock components)
│   ├── test_io/
│   ├── test_detection/
│   ├── test_preprocessing/
│   ├── test_inference/
│   ├── test_reporting/
│   └── test_pipeline/
│
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```
