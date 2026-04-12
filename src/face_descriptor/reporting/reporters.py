"""Pipeline result reporters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("TkAgg")
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
    """Displays a 4-panel figure for each pipeline result.

    Panels
    ------
    1. **Original image** — the raw image as read from disk.
    2. **Detection overlay** — bounding boxes and landmarks drawn on the image.
    3. **Preprocessed face** — the aligned / normalised face crop.
    4. **Model predictions** — textual summary of metadata entries
       (age, gender, facial attributes, anti-spoofing, etc.).

    Parameters
    ----------
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.
    bbox_color:
        RGB colour tuple ``(R, G, B)`` in ``[0, 255]`` used for bounding boxes.
    landmark_color:
        RGB colour tuple ``(R, G, B)`` in ``[0, 255]`` used for landmark dots.
    bbox_thickness:
        Line thickness for bounding boxes in pixels.
    landmark_radius:
        Radius of landmark circles in pixels.
    save_dir:
        If provided, each figure is saved to this directory instead of
        being displayed interactively.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (18, 5),
        bbox_color: tuple[int, int, int] = (0, 255, 0),
        landmark_color: tuple[int, int, int] = (0, 0, 255),
        bbox_thickness: int = 2,
        landmark_radius: int = 3,
        save_dir: str | Path | None = None,
    ) -> None:
        self._figsize = figsize
        self._bbox_color = bbox_color
        self._landmark_color = landmark_color
        self._bbox_thickness = bbox_thickness
        self._landmark_radius = landmark_radius
        self._save_dir = Path(save_dir) if save_dir is not None else None

        if self._save_dir is not None:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API (satisfies Reporter protocol)
    # ------------------------------------------------------------------

    def report(self, results: Sequence[PipelineResult]) -> None:
        """Render a 4-panel visualisation for every result.

        Parameters
        ----------
        results:
            The pipeline results to visualise.  Each result **must**
            carry ``image`` and ``preprocessed_face`` (i.e. they must
            not be ``None``).
        """
        for idx, result in enumerate(results):
            fig = self._build_figure(result, idx)

            if self._save_dir is not None:
                dest = self._save_dir / f"visual_result_{idx:04d}.png"
                fig.savefig(str(dest), bbox_inches="tight", dpi=150)
                logger.info("Saved visual report to %s", dest)
                plt.close(fig)
            else:
                plt.show()

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def _build_figure(self, result: PipelineResult, idx: int) -> Figure:
        """Compose the 4-panel figure for a single *result*."""
        fig, axes = plt.subplots(1, 4, figsize=self._figsize)
        fig.suptitle(f"Result {idx} — {result.source}", fontsize=12)

        self._panel_original(axes[0], result)
        self._panel_detection(axes[1], result)
        self._panel_preprocessed(axes[2], result)
        self._panel_predictions(axes[3], result)

        fig.tight_layout()
        return fig

    def _panel_original(self, ax: plt.Axes, result: PipelineResult) -> None:
        """Panel 1: the raw input image."""
        ax.set_title("Original image")
        if result.image is None:
            ax.text(0.5, 0.5, "Image not available", ha="center", va="center")
            ax.set_axis_off()
            return

        ax.imshow(result.image.data)
        ax.set_axis_off()

    def _panel_detection(self, ax: plt.Axes, result: PipelineResult) -> None:
        """Panel 2: bounding boxes and landmarks overlaid on the image."""
        ax.set_title("Face detection")
        if result.image is None:
            ax.text(0.5, 0.5, "Image not available", ha="center", va="center")
            ax.set_axis_off()
            return

        ax.imshow(result.image.data)

        face = result.face
        bbox = face.bbox
        bbox_color_norm = tuple(c / 255.0 for c in self._bbox_color)
        landmark_color_norm = tuple(c / 255.0 for c in self._landmark_color)

        rect = patches.Rectangle(
            (bbox.x, bbox.y),
            bbox.w,
            bbox.h,
            linewidth=self._bbox_thickness,
            edgecolor=bbox_color_norm,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            bbox.x,
            max(bbox.y - 6, 0),
            f"{face.confidence:.2f}",
            fontsize=8,
            color=bbox_color_norm,
        )

        if face.landmarks is not None:
            for lx, ly in face.landmarks.astype(int):
                ax.plot(lx, ly, "o", color=landmark_color_norm, markersize=self._landmark_radius)

        ax.set_axis_off()

    def _panel_preprocessed(self, ax: plt.Axes, result: PipelineResult) -> None:
        """Panel 3: the aligned / normalised face crop."""
        ax.set_title("Preprocessed face")
        if result.preprocessed_face is None:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
            ax.set_axis_off()
            return

        displayable = self._normalise_for_display(result.preprocessed_face.data)
        ax.imshow(displayable)
        ax.set_axis_off()

    def _panel_predictions(self, ax: plt.Axes, result: PipelineResult) -> None:
        """Panel 4: textual model-prediction summary from metadata."""
        ax.set_title("Model predictions")
        ax.set_axis_off()

        if not result.metadata:
            ax.text(
                0.5, 0.5,
                "No predictions available",
                ha="center", va="center",
                fontsize=10, style="italic", color="grey",
            )
            return

        lines: list[str] = []
        for key, value in result.metadata.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        text_block = "\n".join(lines)
        ax.text(
            0.05, 0.95,
            text_block,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.5},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_for_display(data: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Scale a float tensor into displayable ``[0, 255]`` uint8 range.

        Assumes the input is already in RGB channel order.
        """
        min_val, max_val = data.min(), data.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(data, dtype=np.uint8)
        return ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
