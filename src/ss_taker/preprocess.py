from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    *,
    blur: bool = True,
    upscale: bool = True,
    scale: float = 1.5,
    threshold: bool = True,
    sharpen: bool = True,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    denoised = cv2.medianBlur(normalized, 3) if blur else normalized
    if upscale:
        denoised = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if threshold:
        processed = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
    else:
        processed = denoised
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(processed, -1, kernel)
    return processed
