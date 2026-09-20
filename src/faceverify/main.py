from pathlib import Path

from faceverify.detector import FaceDetector
from faceverify.embedder import FaceEmbedder
from faceverify.gui import FaceVerifyApp
from faceverify.store import RegistrationStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = Path.home() / ".face-verify"


def main():
    detector = FaceDetector(str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"))
    embedder = FaceEmbedder(str(MODELS_DIR / "w600k_r50.onnx"))
    store = RegistrationStore(DATA_DIR)
    FaceVerifyApp(detector, embedder, store).run()


if __name__ == "__main__":
    main()
