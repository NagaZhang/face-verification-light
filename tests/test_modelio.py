from pathlib import Path

import pytest

from faceverify.detector import FaceDetector
from faceverify.modelio import opencv_safe_path

YU_NET = (
    Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
)


def test_opencv_safe_path_ascii_untouched(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"x")
    assert opencv_safe_path(str(p)) == str(p)


def test_opencv_safe_path_copies_non_ascii(tmp_path):
    src = tmp_path / "中文目录" / "model.onnx"
    src.parent.mkdir()
    src.write_bytes(b"dummy")
    out = opencv_safe_path(str(src))
    assert out != str(src)
    assert out.isascii()
    assert Path(out).read_bytes() == b"dummy"


@pytest.mark.skipif(not YU_NET.exists(), reason="YuNet model not present")
def test_detector_loads_from_chinese_path(tmp_path):
    # Mimic PyInstaller onefile: the bundled model ends up under a directory
    # with Chinese characters (as happens with a Chinese Windows user name).
    # cv2.FaceDetectorYN.create would fail directly on this path; FaceDetector
    # must route it through opencv_safe_path and load successfully.
    chinese_dir = tmp_path / "中文用户"
    chinese_dir.mkdir()
    target = chinese_dir / YU_NET.name
    target.write_bytes(YU_NET.read_bytes())

    det = FaceDetector(str(target))
    assert det.detector is not None
