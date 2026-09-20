import cv2
import numpy as np


def imread_unicode(path) -> np.ndarray | None:
    """Read an image from a path that may contain non-ASCII (e.g. Chinese) characters.

    cv2.imread uses OpenCV's C++ file API, which cannot open Unicode paths on
    Windows. np.fromfile uses Python's Unicode-safe file I/O, and cv2.imdecode
    decodes the bytes.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
