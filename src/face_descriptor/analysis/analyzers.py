"""Face attribute analyzers.

Each analyzer implements the ``FaceAnalyzer`` protocol: it takes a
:class:`PreprocessedFace` and returns a ``dict[str, object]`` of
predicted attributes that get merged into ``PipelineResult.metadata``.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from face_descriptor.core.types import PreprocessedFace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reusable base: wraps any HuggingFace image-classification model
# ---------------------------------------------------------------------------

class _HuggingFaceClassifierBase:
    """Thin wrapper around a ``transformers`` image-classification pipeline.

    Lazily loads the model on first call so that import time stays fast.

    Parameters
    ----------
    repo_id:
        HuggingFace Hub model identifier
        (e.g. ``"nateraw/vit-age-classifier"``).
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
        Defaults to ``"cpu"``.
    """

    def __init__(self, repo_id: str, *, device: str = "cpu") -> None:
        self._repo_id = repo_id
        self._device = device
        self._pipe: Any | None = None  # lazy

    # -- lazy init ---------------------------------------------------------

    def _ensure_loaded(self) -> Any:
        if self._pipe is None:
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline(
                "image-classification",
                model=self._repo_id,
                device=self._device,
            )
            logger.info("Loaded HuggingFace model %s on %s", self._repo_id, self._device)
        return self._pipe

    # -- helper: run classification ----------------------------------------

    def _classify(self, face: PreprocessedFace, top_k: int = 5) -> list[dict[str, Any]]:
        """Run the HF pipeline on *face* and return raw predictions.

        Returns a list of ``{"label": str, "score": float}`` dicts
        sorted by descending score.
        """
        from PIL import Image as PILImage

        pipe = self._ensure_loaded()

        # Convert float32 [0,1] (H,W,3) → uint8 PIL RGB image
        img_uint8 = (face.data * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_uint8, mode="RGB")

        results: list[dict[str, Any]] = pipe(pil_img, top_k=top_k)
        return results


# ---------------------------------------------------------------------------
# Concrete analyzers
# ---------------------------------------------------------------------------


class AgeAnalyzer(_HuggingFaceClassifierBase):
    """Predicts the age group of a face.

    Uses the ``nateraw/vit-age-classifier`` model from HuggingFace Hub.
    The model classifies faces into age brackets and this analyzer also
    computes an approximate numeric age as the expected value across
    bracket midpoints.

    Parameters
    ----------
    device:
        PyTorch device string.  Defaults to ``"cpu"``.
    """

    # Midpoints for the 9 age buckets produced by the model.
    _BUCKET_MIDPOINTS: dict[str, float] = {
        "0-2": 1.0,
        "3-9": 6.0,
        "10-19": 14.5,
        "20-29": 24.5,
        "30-39": 34.5,
        "40-49": 44.5,
        "50-59": 54.5,
        "60-69": 64.5,
        "more than 70": 75.0,
    }

    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__("nateraw/vit-age-classifier", device=device)

    def analyze(self, face: PreprocessedFace) -> dict[str, object]:
        """Return ``{"age_group": str, "age_estimate": float, "age_confidence": float}``."""
        preds = self._classify(face, top_k=len(self._BUCKET_MIDPOINTS))

        top = preds[0]
        age_group: str = top["label"]
        confidence: float = top["score"]

        # Compute expected age across all buckets
        expected_age = 0.0
        for pred in preds:
            midpoint = self._BUCKET_MIDPOINTS.get(pred["label"], 35.0)
            expected_age += pred["score"] * midpoint

        return {
            "age_group": age_group,
            "age_estimate": round(expected_age, 1),
            "age_confidence": round(confidence, 4),
        }


class GenderAnalyzer(_HuggingFaceClassifierBase):
    """Predicts the perceived gender of a face.

    Uses the ``rizvandwiki/gender-classification-2`` model from
    HuggingFace Hub.

    Parameters
    ----------
    device:
        PyTorch device string.  Defaults to ``"cpu"``.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__("rizvandwiki/gender-classification-2", device=device)

    def analyze(self, face: PreprocessedFace) -> dict[str, object]:
        """Return ``{"gender": str, "gender_confidence": float}``."""
        preds = self._classify(face, top_k=2)
        top = preds[0]
        return {
            "gender": top["label"].lower(),
            "gender_confidence": round(top["score"], 4),
        }


class GlassesAnalyzer(_HuggingFaceClassifierBase):
    """Detects whether a face is wearing glasses.

    Uses the ``OmarSalah95/Eyeglasses-Classifier`` model from
    HuggingFace Hub (fine-tuned ViT on glasses / no-glasses).

    Parameters
    ----------
    device:
        PyTorch device string.  Defaults to ``"cpu"``.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__("OmarSalah95/Eyeglasses-Classifier", device=device)

    def analyze(self, face: PreprocessedFace) -> dict[str, object]:
        """Return ``{"glasses": bool, "glasses_confidence": float}``."""
        preds = self._classify(face, top_k=2)
        top = preds[0]
        has_glasses = "glass" in top["label"].lower() and "no" not in top["label"].lower()
        return {
            "glasses": has_glasses,
            "glasses_label": top["label"],
            "glasses_confidence": round(top["score"], 4),
        }


class FacialHairAnalyzer(_HuggingFaceClassifierBase):
    """Detects facial hair attributes.

    Uses the ``dima806/facial_hair_image_detection`` model from
    HuggingFace Hub.

    Parameters
    ----------
    device:
        PyTorch device string.  Defaults to ``"cpu"``.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__("dima806/facial_hair_image_detection", device=device)

    def analyze(self, face: PreprocessedFace) -> dict[str, object]:
        """Return ``{"facial_hair": str, "facial_hair_confidence": float}``."""
        preds = self._classify(face, top_k=5)
        top = preds[0]
        return {
            "facial_hair": top["label"],
            "facial_hair_confidence": round(top["score"], 4),
        }


class SkinToneAnalyzer:
    """Estimates skin tone using the ITA (Individual Typology Angle).

    This is a **model-free** approach: it converts the central face region
    to CIE-Lab colour space and computes the ITA, which is then mapped to
    the Fitzpatrick-like categories defined by Chardon *et al.* (1991):

    +-------+-------------------+-----------+
    | Type  | Description       | ITA range |
    +=======+===================+===========+
    | I     | Very light        | > 55°     |
    | II    | Light             | 41°– 55°  |
    | III   | Intermediate      | 28°– 41°  |
    | IV    | Tan               | 10°– 28°  |
    | V     | Brown             | −30°– 10° |
    | VI    | Dark              | < −30°    |
    +-------+-------------------+-----------+

    Parameters
    ----------
    roi_fraction:
        Fraction of the face crop (centred) used for the colour sample.
        Defaults to ``0.4`` (the inner 40 % of width and height).
    """

    _CATEGORIES: list[tuple[float, str, str]] = [
        (55.0, "I",   "very_light"),
        (41.0, "II",  "light"),
        (28.0, "III", "intermediate"),
        (10.0, "IV",  "tan"),
        (-30.0, "V",  "brown"),
    ]

    def __init__(self, roi_fraction: float = 0.4) -> None:
        self._roi_fraction = roi_fraction

    def analyze(self, face: PreprocessedFace) -> dict[str, object]:
        """Return ``{"skin_tone_ita": float, "skin_tone_type": str, "skin_tone_label": str}``."""
        img_uint8 = (face.data * 255).clip(0, 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]

        # Centre ROI
        margin_x = int(w * (1 - self._roi_fraction) / 2)
        margin_y = int(h * (1 - self._roi_fraction) / 2)
        roi = img_uint8[margin_y:h - margin_y, margin_x:w - margin_x]

        # Convert to CIE-Lab
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab).astype(np.float64)
        mean_l = lab[:, :, 0].mean()  # L channel (0-255 in OpenCV)
        mean_b = lab[:, :, 2].mean()  # b channel (0-255, centre at 128)

        # ITA = arctan((L - 50) / b) × 180/π
        # OpenCV Lab ranges: L ∈ [0,255] → real L* ∈ [0,100], b ∈ [0,255] → real b* ∈ [-128,127]
        real_l = mean_l * 100.0 / 255.0
        real_b = mean_b - 128.0

        if abs(real_b) < 1e-6:
            real_b = 1e-6  # avoid division by zero

        ita = np.degrees(np.arctan2(real_l - 50.0, real_b))

        # Classify
        skin_type = "VI"
        skin_label = "dark"
        for threshold, stype, slabel in self._CATEGORIES:
            if ita >= threshold:
                skin_type = stype
                skin_label = slabel
                break

        return {
            "skin_tone_ita": round(float(ita), 2),
            "skin_tone_type": skin_type,
            "skin_tone_label": skin_label,
        }
