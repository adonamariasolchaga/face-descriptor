"""Face preprocessing implementations."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from face_descriptor.core.types import Face, Image, PreprocessedFace

# ArcFace reference landmarks for a 112×112 crop (5-point: left eye, right eye,
# nose tip, left mouth corner, right mouth corner).
ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class AffineAlignPreprocessor:
    """Aligns, crops and normalises a detected face for model inference.

    Uses a similarity transform from detected 5-point landmarks to canonical
    ArcFace reference positions.  Falls back to a simple bbox crop when
    landmarks are not available.
    """

    def __init__(self, target_size: tuple[int, int] = (112, 112)) -> None:
        self._target_size = target_size
        self._ref_landmarks = self._scale_ref_landmarks(target_size)

    def preprocess(self, image: Image, face: Face) -> PreprocessedFace:
        """Align and normalise a face crop.

        Parameters
        ----------
        image:
            The full original image (RGB uint8).
        face:
            The detected face region (with optional landmarks).

        Returns
        -------
        PreprocessedFace
            The aligned, cropped, and normalised face tensor (RGB float32, 0-1).
        """
        if face.landmarks is not None and len(face.landmarks) == 5:
            aligned = self._align_by_landmarks(image.data, face.landmarks)
        else:
            aligned = self._crop_by_bbox(image.data, face)

        normalised = aligned.astype(np.float32) / 255.0
        return PreprocessedFace(data=normalised, original_face=face)

    def _align_by_landmarks(
        self, img: NDArray[np.uint8], landmarks: NDArray[np.float32]
    ) -> NDArray[np.uint8]:
        """Warp *img* so that *landmarks* map to the canonical reference."""
        src = landmarks.reshape(5, 2).astype(np.float32)
        dst = self._ref_landmarks
        M = cv2.estimateAffinePartial2D(src, dst)[0]
        aligned: NDArray[np.uint8] = cv2.warpAffine(
            img, M, self._target_size, borderValue=0
        )
        return aligned

    def _crop_by_bbox(self, img: NDArray[np.uint8], face: Face) -> NDArray[np.uint8]:
        """Simple bbox crop + resize fallback when landmarks are absent."""
        b = face.bbox
        h, w = img.shape[:2]
        x1, y1 = max(b.x, 0), max(b.y, 0)
        x2, y2 = min(b.x + b.w, w), min(b.y + b.h, h)
        crop = img[y1:y2, x1:x2]
        resized: NDArray[np.uint8] = cv2.resize(crop, self._target_size)
        return resized

    @staticmethod
    def _scale_ref_landmarks(target_size: tuple[int, int]) -> NDArray[np.float32]:
        """Scale the 112×112 reference landmarks to *target_size*."""
        sx = target_size[0] / 112.0
        sy = target_size[1] / 112.0
        scaled = ARCFACE_REF_LANDMARKS.copy()
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy
        return scaled
