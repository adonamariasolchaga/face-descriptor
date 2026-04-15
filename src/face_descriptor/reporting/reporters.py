"""Pipeline result reporters."""

from __future__ import annotations

import logging
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib

try:
    import tkinter  # noqa: F401
except ImportError:
    # tkinter unavailable (common with Python 3.12+); fall back to the
    # non-interactive Agg backend so figure creation still works.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from face_descriptor.core.types import Image, PipelineResult

logger = logging.getLogger(__name__)


class ConsoleReporter:
    """Prints pipeline results to standard output."""

    def report(self, results: Sequence[PipelineResult]) -> None:
        """Print a human-readable summary of *results* to stdout.

        Parameters
        ----------
        results:
            The pipeline results to display.
        """
        raise NotImplementedError


class JsonReporter:
    """Writes pipeline results as a JSON file.

    Parameters
    ----------
    output_path:
        Destination file path for the JSON output.
    """

    def __init__(self, output_path: str | Path) -> None:
        self._output_path = Path(output_path)

    def report(self, results: Sequence[PipelineResult]) -> None:
        """Serialise *results* to a JSON file.

        Parameters
        ----------
        results:
            The pipeline results to serialise.
        """
        raise NotImplementedError


class VisualReporter:
    """Displays a single figure per source image summarising all detections.

    Layout
    ------
    - **Row 0**: original image (left) + all detections overlay (right),
      spanning 3 columns each across a 6-column grid.
    - **Rows 1+**: a fixed 6-column grid where each cell shows one
      detected face (preprocessed crop + predictions overlaid as text).

    Parameters
    ----------
    grid_cols:
        Number of columns in the face grid. Defaults to ``6``.
    figsize_base:
        Base ``(width, height)`` in inches for the top row.
    bbox_color:
        RGB colour tuple ``(R, G, B)`` in ``[0, 255]``.
    landmark_color:
        RGB colour tuple ``(R, G, B)`` in ``[0, 255]``.
    bbox_thickness:
        Line thickness for bounding boxes.
    landmark_radius:
        Marker size for landmark dots.
    save_dir:
        If set, figures are saved here instead of displayed.
    """

    def __init__(
        self,
        grid_cols: int = 6,
        figsize_base: tuple[float, float] = (18, 5),
        bbox_color: tuple[int, int, int] = (0, 255, 0),
        landmark_color: tuple[int, int, int] = (0, 0, 255),
        bbox_thickness: int = 2,
        landmark_radius: int = 3,
        save_dir: str | Path | None = None,
    ) -> None:
        self._grid_cols = grid_cols
        self._figsize_base = figsize_base
        self._bbox_color = tuple(c / 255.0 for c in bbox_color)
        self._landmark_color = tuple(c / 255.0 for c in landmark_color)
        self._bbox_thickness = bbox_thickness
        self._landmark_radius = landmark_radius
        self._save_dir = Path(save_dir) if save_dir is not None else None

        if self._save_dir is not None:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self, results: Sequence[PipelineResult]) -> None:
        grouped = self._group_by_source(results)
        interactive = matplotlib.get_backend().lower() != "agg"

        for idx, (source, group) in enumerate(grouped.items()):
            fig = self._build_figure(source, group)

            if self._save_dir is not None:
                dest = self._save_dir / f"visual_result_{idx:04d}.png"
                fig.savefig(str(dest), bbox_inches="tight", dpi=150)
                logger.info("Saved visual report to %s", dest)
                plt.close(fig)
            elif interactive:
                plt.show()
            else:
                tmp = Path(tempfile.mkdtemp()) / f"visual_result_{idx:04d}.png"
                fig.savefig(str(tmp), bbox_inches="tight", dpi=150)
                plt.close(fig)
                logger.info("No interactive backend; opening %s", tmp)
                self._open_with_default_viewer(tmp)

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self, source: str, results: list[PipelineResult]) -> Figure:
        import math

        cols = self._grid_cols
        n_faces = len(results)
        face_rows = max(1, math.ceil(n_faces / cols))

        base_w, base_h = self._figsize_base
        cell_h = base_h * 0.6
        fig_h = base_h + face_rows * cell_h

        # Total rows: 1 (top) + face_rows
        n_rows = 1 + face_rows
        fig, axes = plt.subplots(
            n_rows, cols,
            figsize=(base_w, fig_h),
            gridspec_kw={"height_ratios": [base_h] + [cell_h] * face_rows},
        )
        fig.suptitle(source, fontsize=12)

        # Ensure axes is always 2-D
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        # --- Top row: merge left 3 cols for original, right 3 for detections ---
        for c in range(cols):
            axes[0, c].set_axis_off()

        # Original image — drawn on the left-half merged axis
        ax_orig = fig.add_subplot(n_rows, 2, 1)
        self._panel_original(ax_orig, results[0])

        # Detections overlay — drawn on the right-half merged axis
        ax_det = fig.add_subplot(n_rows, 2, 2)
        self._panel_all_detections(ax_det, results)

        # --- Face grid rows ---
        for i, result in enumerate(results):
            row = 1 + i // cols
            col = i % cols
            ax = axes[row, col]
            self._panel_face_cell(ax, result, face_idx=i)

        # Hide unused cells
        for i in range(n_faces, face_rows * cols):
            row = 1 + i // cols
            col = i % cols
            axes[row, col].set_axis_off()

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    def _panel_original(self, ax: plt.Axes, result: PipelineResult) -> None:
        ax.set_title("Original image")
        if result.image is None:
            ax.text(0.5, 0.5, "Image not available", ha="center", va="center")
            ax.set_axis_off()
            return
        ax.imshow(result.image.data)
        ax.set_axis_off()

    def _panel_all_detections(self, ax: plt.Axes, results: list[PipelineResult]) -> None:
        ax.set_title(f"Detections ({len(results)} face(s))")
        image = results[0].image
        if image is None:
            ax.text(0.5, 0.5, "Image not available", ha="center", va="center")
            ax.set_axis_off()
            return

        ax.imshow(image.data)
        for i, r in enumerate(results):
            bbox = r.face.bbox
            rect = patches.Rectangle(
                (bbox.x, bbox.y), bbox.w, bbox.h,
                linewidth=self._bbox_thickness,
                edgecolor=self._bbox_color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                bbox.x, max(bbox.y - 6, 0),
                f"#{i} ({r.face.confidence:.2f})",
                fontsize=8, color=self._bbox_color,
            )
            if r.face.landmarks is not None:
                for lx, ly in r.face.landmarks.astype(int):
                    ax.plot(lx, ly, "o", color=self._landmark_color,
                            markersize=self._landmark_radius)
        ax.set_axis_off()

    def _panel_face_cell(self, ax: plt.Axes, result: PipelineResult,
                         face_idx: int) -> None:
        """Single grid cell: preprocessed face with predictions as overlay text."""
        ax.set_title(f"#{face_idx}", fontsize=9, pad=2)

        if result.preprocessed_face is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_axis_off()
            return

        displayable = self._normalise_for_display(result.preprocessed_face.data)
        ax.imshow(displayable)
        ax.set_axis_off()

        # Overlay predictions text
        if result.metadata:
            lines = [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in result.metadata.items()
            ]
            ax.text(
                0.02, 0.98, "\n".join(lines),
                transform=ax.transAxes,
                fontsize=6, verticalalignment="top",
                fontfamily="monospace", color="white",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.6},
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_source(results: Sequence[PipelineResult]) -> dict[str, list[PipelineResult]]:
        grouped: dict[str, list[PipelineResult]] = {}
        for r in results:
            grouped.setdefault(r.source, []).append(r)
        return grouped

    @staticmethod
    def _open_with_default_viewer(path: Path) -> None:
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", str(path)])
            elif system == "Windows":
                subprocess.Popen(["cmd", "/c", "start", "", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except FileNotFoundError:
            logger.warning("Could not open %s automatically.", path)

    @staticmethod
    def _normalise_for_display(data: NDArray[np.float32]) -> NDArray[np.uint8]:
        min_val, max_val = data.min(), data.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(data, dtype=np.uint8)
        return ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
