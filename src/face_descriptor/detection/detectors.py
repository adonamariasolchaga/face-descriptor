"""Face detector implementations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from face_descriptor.core.types import BoundingBox, Face, Image
from face_descriptor.detection import models

class MediaPipeDetector:
    """Face detector backed by MediaPipe Face Detection."""

    def detect(self, image: Image) -> Sequence[Face]:
        """Detect faces in *image*.

        Parameters
        ----------
        image:
            The input image to scan for faces.

        Returns
        -------
        Sequence[Face]
            Zero or more detected faces.
        """
        raise NotImplementedError


class SCRFDDetector:
    """SCRFD face detector running via ONNX Runtime.

    Parameters
    ----------
    model_path:
        Path to the SCRFD ``.onnx`` model file.
    input_size:
        Model input resolution ``(width, height)``.
    conf_threshold:
        Minimum confidence to keep a detection.
    nms_threshold:
        IoU threshold for non-maximum suppression.
    """

    _FEAT_STRIDE_FPN = [8, 16, 32]
    _NUM_ANCHORS = 2

    def __init__(
        self,
        model_path: str | Path = os.path.join(os.path.dirname(models.__file__), "scrfd_2.5g_gnkps.onnx"),
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        import onnxruntime as ort

        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image: Image) -> Sequence[Face]:
        """Detect faces in *image* and return bounding boxes + 5-point landmarks."""
        img = image.data  # RGB uint8
        img_h, img_w = img.shape[:2]
        inp_w, inp_h = self._input_size

        # --- preprocess: letterbox resize --------------------------------
        scale = min(inp_w / img_w, inp_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        padded = np.zeros((inp_h, inp_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis]

        # --- inference ----------------------------------------------------
        outputs = self._session.run(None, {self._input_name: blob})

        # --- decode outputs per stride ------------------------------------
        all_scores: list[NDArray] = []
        all_bboxes: list[NDArray] = []
        all_landmarks: list[NDArray] = []

        for idx, stride in enumerate(self._FEAT_STRIDE_FPN):
            scores = outputs[idx]                # (1, H*W*anchors, 1)
            bbox_deltas = outputs[idx + 3]       # (1, H*W*anchors, 4)
            kps_deltas = outputs[idx + 6]        # (1, H*W*anchors, 10)

            scores = scores.reshape(-1)
            bbox_deltas = bbox_deltas.reshape(-1, 4)
            kps_deltas = kps_deltas.reshape(-1, 10)

            fh = inp_h // stride
            fw = inp_w // stride
            anchors = self._build_anchors(fw, fh, stride)

            bboxes = self._decode_bboxes(anchors, bbox_deltas, stride)
            landmarks = self._decode_landmarks(anchors, kps_deltas, stride)

            all_scores.append(scores)
            all_bboxes.append(bboxes)
            all_landmarks.append(landmarks)

        scores = np.concatenate(all_scores)
        bboxes = np.concatenate(all_bboxes)
        landmarks = np.concatenate(all_landmarks)

        # --- filter + NMS -------------------------------------------------
        mask = scores > self._conf_threshold
        scores = scores[mask]
        bboxes = bboxes[mask]
        landmarks = landmarks[mask]

        if len(scores) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), scores.tolist(), self._conf_threshold, self._nms_threshold,
        ).flatten()

        # --- map back to original image coords ----------------------------
        faces: list[Face] = []
        for i in indices:
            x1, y1, x2, y2 = bboxes[i] / scale
            lm = (landmarks[i].reshape(5, 2) / scale).astype(np.float32)
            bx, by, bw, bh = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            faces.append(Face(
                bbox=BoundingBox(x=bx, y=by, w=bw, h=bh),
                confidence=float(scores[i]),
                landmarks=lm,
            ))
        return faces

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anchors(fw: int, fh: int, stride: int) -> NDArray[np.float32]:
        """Generate anchor centre points for a feature map."""
        ys, xs = np.mgrid[:fh, :fw]
        centres = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
        centres = (centres * stride).astype(np.float32)
        # duplicate for each anchor
        centres = np.tile(centres, (1, SCRFDDetector._NUM_ANCHORS)).reshape(-1, 2)
        return centres

    @staticmethod
    def _decode_bboxes(anchors: NDArray, deltas: NDArray, stride: int) -> NDArray:
        """Decode bbox deltas (distance from anchor) to ``[x1, y1, x2, y2]``."""
        x1 = anchors[:, 0] - deltas[:, 0] * stride
        y1 = anchors[:, 1] - deltas[:, 1] * stride
        x2 = anchors[:, 0] + deltas[:, 2] * stride
        y2 = anchors[:, 1] + deltas[:, 3] * stride
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _decode_landmarks(anchors: NDArray, deltas: NDArray, stride: int) -> NDArray:
        """Decode landmark deltas to absolute pixel coordinates."""
        lm = deltas.reshape(-1, 5, 2) * stride + anchors[:, np.newaxis, :]
        return lm.reshape(-1, 10)
