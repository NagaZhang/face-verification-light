"""Manual end-to-end smoke test. Register photos first, test photo last:
python scripts/smoke_test.py reg1.jpg reg2.jpg test.jpg
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from faceverify.align import align_face
from faceverify.detector import FaceDetector
from faceverify.embedder import FaceEmbedder
from faceverify.matcher import mean_embedding, verify

MODELS = Path(__file__).resolve().parent.parent / "models"


def embed_from_path(detector, embedder, path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    face = detector.detect_largest(img)
    if face is None:
        raise RuntimeError(f"no face in {path}")
    aligned = align_face(img, face.landmarks)
    return embedder.embed(aligned)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    detector = FaceDetector(str(MODELS / "face_detection_yunet_2023mar.onnx"))
    embedder = FaceEmbedder(str(MODELS / "w600k_r50.onnx"))
    register_paths = sys.argv[1:-1]
    test_path = sys.argv[-1]
    ref = mean_embedding([embed_from_path(detector, embedder, p) for p in register_paths])
    test = embed_from_path(detector, embedder, test_path)
    is_match, sim = verify(test, ref, threshold=0.35)
    print(f"similarity={sim:.4f} match={is_match}")


if __name__ == "__main__":
    main()
