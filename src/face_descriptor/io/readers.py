"""Image reader implementations."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from face_descriptor.core.types import Image


class FileImageReader:
    """Reads an image from a local file path."""

    def read(self, source: str) -> Image:
        """Read an image file and return an :class:`Image` instance.

        Parameters
        ----------
        source:
            Filesystem path to the image.

        Returns
        -------
        Image
            The loaded image with its pixel data in uint8 RGB format [H, W, 3].
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {source}")

        pixel_data = cv2.imread(str(path))
        if pixel_data is None:
            raise ValueError(f"Failed to read image: {source}")

        rgb_data: NDArray[np.uint8] = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2RGB)
        return Image(data=rgb_data, source=source)
