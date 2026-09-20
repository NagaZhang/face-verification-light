import sys
from pathlib import Path

from faceverify.detector import FaceDetector
from faceverify.embedder import FaceEmbedder
from faceverify.gui import FaceVerifyApp
from faceverify.store import RegistrationStore


def _models_dir() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller onefile extracts bundled data to sys._MEIPASS.
        return Path(sys._MEIPASS) / "models"
    return Path(__file__).resolve().parent.parent.parent / "models"


MODELS_DIR = _models_dir()
DATA_DIR = Path.home() / ".face-verify"


def main():
    detector = FaceDetector(str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"))
    embedder = FaceEmbedder(str(MODELS_DIR / "w600k_r50.onnx"))
    store = RegistrationStore(DATA_DIR)
    FaceVerifyApp(detector, embedder, store).run()


if __name__ == "__main__":
    main()
