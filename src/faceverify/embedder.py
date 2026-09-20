from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class FaceEmbedder:
    def __init__(self, model_path: str):
        # Read via Unicode-safe Python I/O and pass bytes so a non-ASCII path
        # (e.g. a Chinese Windows user name) can't break model loading.
        self.session = ort.InferenceSession(
            Path(model_path).read_bytes(), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """aligned_face_bgr: 112x112 BGR -> 512-dim L2-normalized embedding."""
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (112, 112))
        blob = ((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None, ...]
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        emb = np.asarray(out, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
