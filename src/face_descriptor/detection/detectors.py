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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _distance2bbox(points: NDArray, distance: NDArray) -> NDArray:
    """Decode distance predictions to bounding boxes [x1, y1, x2, y2]."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: NDArray, distance: NDArray) -> NDArray:
    """Decode distance predictions to keypoint coordinates."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


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

    Faithfully follows the reference InsightFace implementation, including
    proper input normalization, model topology auto-detection, and anchor
    caching.

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

    def __init__(
        self,
        model_path: str | Path = os.path.join(
            os.path.dirname(models.__file__), "scrfd_2.5g_gnkps.onnx"
        ),
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
            providers=ort.get_available_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name
        self._center_cache: dict[tuple[int, int, int], NDArray] = {}

        # Auto-detect model topology from number of outputs
        num_outputs = len(self._session.get_outputs())
        self._batched = len(self._session.get_outputs()[0].shape) == 3

        if num_outputs == 6:
            self._fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self._use_kps = False
        elif num_outputs == 9:
            self._fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self._use_kps = True
        elif num_outputs == 10:
            self._fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self._use_kps = False
        elif num_outputs == 15:
            self._fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self._use_kps = True
        else:
            raise ValueError(f"Unexpected number of model outputs: {num_outputs}")

    def detect(self, image: Image) -> Sequence[Face]:
        img = image.data  # RGB uint8
        img_h, img_w = img.shape[:2]
        inp_w, inp_h = self._input_size

        # --- letterbox resize ---
        det_scale = min(inp_w / img_w, inp_h / img_h)
        new_w, new_h = int(img_w * det_scale), int(img_h * det_scale)
        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((inp_h, inp_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        # --- normalize: (pixel - 127.5) / 128, swap RGB→BGR ---
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128, (inp_w, inp_h), (127.5, 127.5, 127.5), swapRB=True
        )

        # --- inference ---
        net_outs = self._session.run(None, {self._input_name: blob})

        # --- decode per stride ---
        scores_list: list[NDArray] = []
        bboxes_list: list[NDArray] = []
        kpss_list: list[NDArray] = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            if self._batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self._fmc][0] * stride
                if self._use_kps:
                    kps_preds = net_outs[idx + self._fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self._fmc] * stride
                if self._use_kps:
                    kps_preds = net_outs[idx + self._fmc * 2] * stride

            height = inp_h // stride
            width = inp_w // stride
            anchor_centers = self._get_anchors(height, width, stride)

            # Early threshold filtering (per-stride)
            pos_inds = np.where(scores >= self._conf_threshold)[0]
            if len(pos_inds) == 0:
                continue

            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self._use_kps:
                kpss = _distance2kps(anchor_centers, kps_preds).reshape(-1, 5, 2)
                kpss_list.append(kpss[pos_inds])

        if not scores_list:
            return []

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)

        order = scores.argsort()[::-1]
        scores = scores[order]
        bboxes = bboxes[order]

        has_kps = self._use_kps and len(kpss_list) > 0
        if has_kps:
            kpss = np.vstack(kpss_list)[order]

        # --- NMS ---
        keep = self._nms(
            np.hstack((bboxes, scores[:, None])).astype(np.float32)
        )

        # --- build results, map back to original coords ---
        faces: list[Face] = []
        for i in keep:
            x1, y1, x2, y2 = bboxes[i] / det_scale
            bx, by = int(x1), int(y1)
            bw, bh = int(x2 - x1), int(y2 - y1)
            lm = None
            if has_kps:
                lm = (kpss[i] / det_scale).astype(np.float32)
            faces.append(Face(
                bbox=BoundingBox(x=bx, y=by, w=bw, h=bh),
                confidence=float(scores[i]),
                landmarks=lm,
            ))
        return faces

    def _get_anchors(self, height: int, width: int, stride: int) -> NDArray:
        key = (height, width, stride)
        if key not in self._center_cache:
            centers = np.stack(
                np.mgrid[:height, :width][::-1], axis=-1
            ).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if self._num_anchors > 1:
                centers = np.stack([centers] * self._num_anchors, axis=1).reshape(-1, 2)
            if len(self._center_cache) < 100:
                self._center_cache[key] = centers
            return centers
        return self._center_cache[key]

    def _nms(self, dets: NDArray) -> list[int]:
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
            inds = np.where(ovr <= self._nms_threshold)[0]
            order = order[inds + 1]
        return keep
