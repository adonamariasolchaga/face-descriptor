"""Tests for image reader implementations."""
import os
import pytest

import numpy as np

from face_descriptor.core import Image
from face_descriptor.io import FileImageReader
from tests import resources

@pytest.fixture
def reader():
    return FileImageReader()


@pytest.mark.parametrize("im_path, expected_shape", [
        (
            os.path.join(os.path.dirname(resources.__file__), "face_images/single_face_1.jpeg"),
            (198, 255, 3)
        ),
    ]
)

def test_read_img(reader, im_path, expected_shape):
    readed_im = reader.read(im_path)
    print(readed_im.data.shape)
    assert isinstance(readed_im, Image)
    assert readed_im.data.shape == expected_shape
