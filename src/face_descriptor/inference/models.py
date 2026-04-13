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


class PyTorchInferencer:
    """Runs inference on a preprocessed face using a PyTorch model.

    Supports loading from:
    - A local ``.pt`` / ``.pth`` file (a ``torch.nn.Module`` saved with
      ``torch.save``).
    - A HuggingFace Hub model identifier (loaded via ``transformers``
      ``AutoModel``).

    The model is expected to accept a ``(1, 3, H, W)`` float32 tensor and
    return an embedding vector.

    Parameters
    ----------
    model_source:
        Path to a local model file **or** a HuggingFace Hub repo id.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, …).
    """

    def __init__(self, model_source: str | Path, *, device: str = "cpu") -> None:
        self._model_source = str(model_source)
        self._device = device
        self._model: object | None = None  # lazy

    def _ensure_loaded(self) -> object:
        if self._model is None:
            import torch

            source = Path(self._model_source)
            if source.exists() and source.suffix in (".pt", ".pth"):
                self._model = torch.load(
                    str(source), map_location=self._device, weights_only=False,
                )
            else:
                # Treat as HuggingFace repo id
                from transformers import AutoModel

                self._model = AutoModel.from_pretrained(self._model_source)
                self._model.to(self._device)  # type: ignore[union-attr]

            if callable(getattr(self._model, "eval", None)):
                self._model.eval()  # type: ignore[union-attr]
        return self._model

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
        import torch
        import numpy as np

        model = self._ensure_loaded()

        # (H, W, 3) float32 [0,1] → (1, 3, H, W) tensor
        tensor = torch.from_numpy(
            face.data.transpose(2, 0, 1)[np.newaxis]
        ).to(self._device)

        with torch.no_grad():
            output = model(tensor)  # type: ignore[operator]

        # Handle transformers BaseModelOutput or plain tensor
        if hasattr(output, "last_hidden_state"):
            # Use CLS token or mean pool
            vec = output.last_hidden_state[:, 0, :]
        elif hasattr(output, "pooler_output"):
            vec = output.pooler_output
        elif isinstance(output, torch.Tensor):
            vec = output
        else:
            vec = output[0]

        embedding = vec.squeeze().cpu().numpy().astype(np.float32)
        return Embedding(vector=embedding)
