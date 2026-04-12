"""Model inference implementations."""

from __future__ import annotations

from pathlib import Path

from face_descriptor.core.types import Embedding, PreprocessedFace


class OnnxInferencer:
    """Runs inference on a preprocessed face using an ONNX model.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)

    def infer(self, face: PreprocessedFace) -> Embedding:
        """Produce an embedding vector for *face*.

        Parameters
        ----------
        face:
            A preprocessed (aligned + normalised) face tensor.

        Returns
        -------
        Embedding
            The resulting feature vector.
        """
        raise NotImplementedError
