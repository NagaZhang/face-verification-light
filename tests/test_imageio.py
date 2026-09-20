import cv2
import numpy as np

from faceverify.imageio import imread_unicode


def test_imread_unicode_chinese_path(tmp_path):
    img = np.full((10, 10, 3), 0, dtype=np.uint8)
    img[:, :] = (0, 0, 255)
    path = tmp_path / "人脸测试.jpg"
    ok, buf = cv2.imencode(".jpg", img)
    assert ok is True
    buf.tofile(str(path))
    loaded = imread_unicode(str(path))
    assert loaded is not None
    assert loaded.shape == (10, 10, 3)


def test_imread_unicode_missing_file(tmp_path):
    assert imread_unicode(str(tmp_path / "不存在.jpg")) is None
