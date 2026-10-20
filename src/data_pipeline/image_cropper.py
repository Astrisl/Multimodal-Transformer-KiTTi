"""2D image patch for a 3D box: project the 8 corners and take their bounding rectangle."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from multimodal_transformer.utils.calibration import Calibration


def box_to_image_roi(
    corners_rect: np.ndarray,
    calibration: Calibration,
    image_shape: Tuple[int, int],
    padding: int = 5,
    min_size: int = 8,
) -> Optional[Tuple[int, int, int, int]]:
    """Image rectangle (x1, y1, x2, y2) covering a 3D box, clipped to the image.

    Args:
        corners_rect: Box corners [8, 3] in the rectified camera frame.
        calibration: Camera calibration.
        image_shape: (height, width) of the image.
        padding: Context pixels added on each side.
        min_size: Rectangles narrower/shorter than this are rejected.

    Returns:
        The rectangle, or None when the box is behind the camera, fully outside the
        image or too small to carry visual information.
    """
    pixels, depth = calibration.rect_to_image(corners_rect)
    # A box partly behind the camera has corners that project to garbage coordinates.
    if np.any(~np.isfinite(pixels)) or np.any(depth <= 0):
        return None
    height, width = image_shape
    x1, y1 = np.floor(pixels.min(axis=0)) - padding
    x2, y2 = np.ceil(pixels.max(axis=0)) + padding
    x1, x2 = int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))
    y1, y2 = int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return x1, y1, x2, y2


def crop_image(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Returns a copy of the ROI so later in-place edits do not touch the frame."""
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2].copy()
