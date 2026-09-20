import numpy as np

from faceverify.align import align_face


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
