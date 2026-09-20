import numpy as np

from faceverify.align import ARCFACE_SRC, _align_matrix, align_face


def test_align_matrix_maps_landmarks_to_canonical():
    # YuNet order: right eye, left eye, nose, right mouth, left mouth.
    # Shift the canonical eye/nose points by a known offset; mouth unused.
    lm = np.array(
        [
            [ARCFACE_SRC[1][0] + 100.0, ARCFACE_SRC[1][1] + 50.0],  # right eye
            [ARCFACE_SRC[0][0] + 100.0, ARCFACE_SRC[0][1] + 50.0],  # left eye
            [ARCFACE_SRC[2][0] + 100.0, ARCFACE_SRC[2][1] + 50.0],  # nose
            [0.0, 0.0],
            [0.0, 0.0],  # unused mouth points
        ],
        dtype=np.float32,
    )
    M = _align_matrix(lm)
    h = np.hstack([lm[:3], np.ones((3, 1))])
    out = (M @ h.T).T
    expected = np.array([ARCFACE_SRC[1], ARCFACE_SRC[0], ARCFACE_SRC[2]])
    assert np.allclose(out, expected, atol=1e-3)


def test_align_face_output_shape():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarks = np.array(
        [
            [150, 100],  # right eye
            [50, 100],   # left eye
            [100, 130],  # nose
            [150, 170],  # right mouth
            [50, 170],   # left mouth
        ],
        dtype=np.float32,
    )
    out = align_face(img, landmarks)
    assert out.shape == (112, 112, 3)
