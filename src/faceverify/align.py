import cv2
import numpy as np

# ArcFace canonical 5-point layout for a 112x112 aligned face, in the order:
# left eye, right eye, nose, left mouth, right mouth.
ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a similarity transform mapping src -> dst (both Nx2). Returns 2x3 M."""
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = (dst_demean.T @ src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    T = np.eye(dim + 1, dtype=np.float64)
    if rank >= dim - 1:
        T[:dim, :dim] = U @ np.diag(d) @ Vt
    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T[:dim, :]


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Align a face using its 5 YuNet landmarks to ArcFace's canonical layout.

    `landmarks` is (5, 2) in YuNet order: right eye, left eye, nose,
    right mouth, left mouth. Returns an output_size x output_size BGR image.
    """
    dst = np.array(
        [landmarks[1], landmarks[0], landmarks[2], landmarks[4], landmarks[3]],
        dtype=np.float32,
    )
    M = _umeyama(dst, ARCFACE_SRC)
    return cv2.warpAffine(image_bgr, M, (output_size, output_size), borderValue=0.0)
