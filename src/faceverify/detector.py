from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Face:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray  # shape (5, 2), float32
    score: float


class FaceDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.6):
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold
        )

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_bgr)
        result: list[Face] = []
        if faces is None:
            return result
        for f in faces:
            x, y, fw, fh = f[:4]
            landmarks = f[4:14].reshape(5, 2).astype(np.float32)
            result.append(
                Face(
                    bbox=(int(x), int(y), int(fw), int(fh)),
                    landmarks=landmarks,
                    score=float(f[14]),
                )
            )
        return result

    def detect_largest(self, image_bgr: np.ndarray) -> Face | None:
        faces = self.detect(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
