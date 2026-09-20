"""Helpers for loading model files that OpenCV's C++ dnn importer can read.

OpenCV's cv::dnn ONNX importer opens files with the ANSI codepage and cannot
read paths containing non-ASCII characters. PyInstaller's onefile mode extracts
bundled data into %TEMP%\\_MEIxxxx, and %TEMP% lives under the Windows user
profile -- so a Chinese user name makes that path non-ASCII and cv2.dnn fails
with "Can't read ONNX file".

opencv_safe_path copies the model bytes (with Python's Unicode-safe file I/O)
to a pure-ASCII temp directory when the given path is non-ASCII.
"""

import os
import tempfile
from pathlib import Path

_ASCII_TMP_DIR = (
    Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Temp" / "face-verify"
)


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def opencv_safe_path(model_path: str) -> str:
    """Return a path cv2.dnn can open, copying to an ASCII dir when needed."""
    if _is_ascii(model_path):
        return model_path

    src = Path(model_path)
    data = src.read_bytes()
    if not data:
        return model_path

    dst = _ASCII_TMP_DIR / src.name
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return str(dst)
    except OSError:
        # C:\Windows\Temp is normally world-writable; fall back to the
        # (possibly non-ASCII) system temp as a last resort.
        fallback = Path(tempfile.gettempdir()) / ("faceverify-" + src.name)
        fallback.write_bytes(data)
        return str(fallback)
