import cv2
import numpy as np

# ArcFace canonical 5-point layout for a 112x112 aligned face, in the order:
# left eye, right eye, nose, left mouth, right mouth.
ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041],   # right mouth
    ],
    dtype=np.float32,
)


def _align_matrix(landmarks: np.ndarray) -> np.ndarray:
    """Return a 2x3 affine transform mapping the face to ArcFace's canonical layout.

    Uses the three most robust landmarks (two eyes + nose) via a full affine
    transform; cv2.getAffineTransform maps three points exactly. `landmarks` is
    in YuNet order: right eye, left eye, nose, right mouth, left mouth.
    """
    src = np.array(
        [landmarks[0], landmarks[1], landmarks[2]], dtype=np.float32
    )  # right eye, left eye, nose
    dst = np.array(
        [ARCFACE_SRC[1], ARCFACE_SRC[0], ARCFACE_SRC[2]], dtype=np.float32
    )  # right eye, left eye, nose (canonical)
    return cv2.getAffineTransform(src, dst)


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Align a face using its 5 YuNet landmarks to ArcFace's canonical layout.

    `landmarks` is (5, 2) in YuNet order: right eye, left eye, nose,
    right mouth, left mouth. Returns an output_size x output_size BGR image.
    """
    M = _align_matrix(landmarks)
    return cv2.warpAffine(image_bgr, M, (output_size, output_size), borderValue=0.0)
