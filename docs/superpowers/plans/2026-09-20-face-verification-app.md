# Face Verification App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight 1:1 face-verification desktop app that registers one named person from several photos and answers "is this test photo the same person?" with a similarity score.

**Architecture:** Seven focused modules under `src/faceverify/` — `detector` (YuNet face+landmark detection), `align` (5-point similarity transform to ArcFace's canonical 112×112), `embedder` (ArcFace ONNX → 512-dim normalized vector), `matcher` (cosine similarity + threshold, pure logic), `store` (JSON + .npy persistence, pure logic), `gui` (tkinter), `main` (assembly). Only `matcher` and `store` are unit-tested with pytest (no model needed); the detection→align→embed pipeline is covered by a manual smoke script.

**Tech Stack:** Python 3.13, OpenCV 5.0 (`cv2.FaceDetectorYN`), onnxruntime, numpy, pillow, tkinter. Reuses mature open-source models: OpenCV YuNet (detection) + InsightFace ArcFace `w600k_r50` (recognition).

## Global Constraints

- Runtime deps ONLY: `opencv-python`, `numpy`, `pillow`, `onnxruntime`. NO torch/tensorflow/dlib (Python 3.13 incompatibility + weight).
- 1:1 verification only; no liveness detection; no database; no network service.
- Model files are gitignored and downloaded separately into `models/` (never committed).
- Package name `faceverify`, source under `src/faceverify/`.
- Core logic (`matcher`, `store`) must be testable without models or a display.
- Similarity threshold default 0.35, user-adjustable in the GUI.
- Use `ensure_ascii=False` and UTF-8 for any Chinese text written to disk.

---

### Task 1: Project scaffolding, dependencies, and model download

**Files:**
- Create: `requirements.txt`
- Create: `conftest.py`
- Create: `run.py`
- Create: `src/faceverify/__init__.py`
- Create: `models/.gitkeep`
- Create: `scripts/smoke_test.py` (skeleton — filled in Task 7)

**Interfaces:**
- Produces: an importable `faceverify` package (via `conftest.py` path injection for pytest and `run.py` for the app), a `models/` dir containing `face_detection_yunet_2023mar.onnx` and `w600k_r50.onnx`.

- [ ] **Step 1: Create `requirements.txt`**

```
opencv-python>=4.8
numpy>=1.24
pillow>=9
onnxruntime>=1.16
pytest>=7
```

- [ ] **Step 2: Create `conftest.py`** (lets pytest import the `src/` package without installing)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
```

- [ ] **Step 3: Create `run.py`** (entry point; adds `src/` to path then launches the app)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from faceverify.main import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `src/faceverify/__init__.py`** (empty file) and `models/.gitkeep` (empty file)

- [ ] **Step 5: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: onnxruntime installs; opencv/numpy/pillow already satisfied. Verify: `python -c "import onnxruntime; print(onnxruntime.__version__)"`.

- [ ] **Step 6: Download YuNet detection model**

Run:
```bash
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```
Expected: `models/face_detection_yunet_2023mar.onnx` exists (~230 KB).

- [ ] **Step 7: Download ArcFace recognition model**

Run:
```bash
curl -L -o models/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o models/buffalo_l.zip -d models/buffalo_l
cp models/buffalo_l/w600k_r50.onnx models/w600k_r50.onnx
```
Expected: `models/w600k_r50.onnx` exists (~166 MB). If GitHub is unreachable (network), report and ask the user to download manually or provide a mirror — do not proceed without the model files.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt conftest.py run.py src/faceverify/__init__.py models/.gitkeep
git commit -m "chore: scaffold project and add dependencies"
```

---

### Task 2: `matcher.py` — cosine similarity and threshold logic (TDD)

**Files:**
- Create: `src/faceverify/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Produces:
  - `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`
  - `normalize(vec: np.ndarray) -> np.ndarray`
  - `mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray`
  - `verify(test_embedding: np.ndarray, reference_embedding: np.ndarray, threshold: float) -> tuple[bool, float]`

- [ ] **Step 1: Write the failing test** (`tests/test_matcher.py`)

```python
import numpy as np
import pytest

from faceverify.matcher import (
    cosine_similarity,
    mean_embedding,
    normalize,
    verify,
)


def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_normalize_unit_norm():
    n = normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(n) == pytest.approx(1.0)


def test_normalize_zero_vector_unchanged():
    v = np.zeros(3)
    assert np.allclose(normalize(v), v)


def test_mean_embedding_normalized():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    m = mean_embedding([a, b])
    assert m.shape == (3,)
    assert np.linalg.norm(m) == pytest.approx(1.0)
    assert m[0] == pytest.approx(m[1])
    assert m[2] == pytest.approx(0.0)


def test_mean_embedding_empty_raises():
    with pytest.raises(ValueError):
        mean_embedding([])


def test_verify_match():
    ref = np.array([1.0, 0.0, 0.0])
    test = normalize(np.array([0.99, 0.14, 0.0]))
    is_match, sim = verify(test, ref, threshold=0.9)
    assert is_match is True
    assert sim == pytest.approx(cosine_similarity(test, ref))


def test_verify_no_match():
    ref = np.array([1.0, 0.0, 0.0])
    test = np.array([0.0, 1.0, 0.0])
    is_match, sim = verify(test, ref, threshold=0.9)
    assert is_match is False
    assert sim == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matcher.py -v`
Expected: FAIL (ModuleNotFoundError: `faceverify.matcher`).

- [ ] **Step 3: Write minimal implementation** (`src/faceverify/matcher.py`)

```python
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors (dot product)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b))


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector; a zero vector is returned unchanged."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average a list of embeddings and L2-normalize the result."""
    if not embeddings:
        raise ValueError("embeddings must not be empty")
    mean = np.mean(np.stack(embeddings), axis=0)
    return normalize(mean)


def verify(
    test_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    threshold: float,
) -> tuple[bool, float]:
    """Return (is_match, similarity) comparing two normalized embeddings."""
    sim = cosine_similarity(test_embedding, reference_embedding)
    return bool(sim >= threshold), sim
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_matcher.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/faceverify/matcher.py tests/test_matcher.py
git commit -m "feat: add matcher (cosine similarity + threshold)"
```

---

### Task 3: `store.py` — registration persistence (TDD)

**Files:**
- Create: `src/faceverify/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `class RegistrationStore` with `__init__(self, data_dir: Path)`, `save(name, embedding, n_photos) -> None`, `load() -> dict | None` (keys `name`, `n_photos`, `embedding`), `clear() -> None`.

- [ ] **Step 1: Write the failing test** (`tests/test_store.py`)

```python
import numpy as np

from faceverify.store import RegistrationStore


def test_load_returns_none_when_empty(tmp_path):
    assert RegistrationStore(tmp_path).load() is None


def test_save_and_load_roundtrip(tmp_path):
    store = RegistrationStore(tmp_path)
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    store.save("张三", emb, 3)
    loaded = store.load()
    assert loaded["name"] == "张三"
    assert loaded["n_photos"] == 3
    assert np.allclose(loaded["embedding"], emb)


def test_save_overwrites(tmp_path):
    store = RegistrationStore(tmp_path)
    store.save("A", np.array([1.0, 0.0], dtype=np.float32), 1)
    store.save("B", np.array([0.0, 1.0], dtype=np.float32), 2)
    loaded = store.load()
    assert loaded["name"] == "B"
    assert loaded["n_photos"] == 2


def test_clear(tmp_path):
    store = RegistrationStore(tmp_path)
    store.save("A", np.array([1.0, 0.0], dtype=np.float32), 1)
    store.clear()
    assert store.load() is None


def test_load_corrupted_json_returns_none(tmp_path):
    store = RegistrationStore(tmp_path)
    store.save("A", np.array([1.0, 0.0], dtype=np.float32), 1)
    store.json_path.write_text("{not valid json", encoding="utf-8")
    assert store.load() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL (ModuleNotFoundError: `faceverify.store`).

- [ ] **Step 3: Write minimal implementation** (`src/faceverify/store.py`)

```python
import json
from pathlib import Path

import numpy as np


class RegistrationStore:
    """Persist a single registered person (name + reference embedding + photo count)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.json_path = self.data_dir / "registration.json"
        self.embedding_path = self.data_dir / "reference.npy"

    def save(self, name: str, embedding: np.ndarray, n_photos: int) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.embedding_path, np.asarray(embedding, dtype=np.float32))
        self.json_path.write_text(
            json.dumps({"name": name, "n_photos": int(n_photos)}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> dict | None:
        if not (self.json_path.exists() and self.embedding_path.exists()):
            return None
        try:
            meta = json.loads(self.json_path.read_text(encoding="utf-8"))
            embedding = np.load(self.embedding_path)
            return {
                "name": meta["name"],
                "n_photos": int(meta["n_photos"]),
                "embedding": embedding,
            }
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def clear(self) -> None:
        self.json_path.unlink(missing_ok=True)
        self.embedding_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/faceverify/store.py tests/test_store.py
git commit -m "feat: add store (registration persistence)"
```

---

### Task 4: `align.py` — 5-point alignment to ArcFace canonical (TDD)

**Files:**
- Create: `src/faceverify/align.py`
- Test: `tests/test_align.py`

**Interfaces:**
- Consumes: a 5-point landmark array in YuNet order (right eye, left eye, nose, right mouth, left mouth).
- Produces: `align_face(image_bgr: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray` returning an `output_size × output_size` BGR image.

- [ ] **Step 1: Write the failing test** (`tests/test_align.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_align.py -v`
Expected: FAIL (ModuleNotFoundError: `faceverify.align`).

- [ ] **Step 3: Write implementation** (`src/faceverify/align.py`)

```python
import cv2
import numpy as np

# ArcFace canonical 5-point layout for a 112x112 aligned face, in the order:
# left eye, right eye, nose, left mouth, right mouth.
ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a similarity transform mapping src -> dst (both Nx2). Returns 2x3 M."""
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = (dst_demean.T @ src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    T = np.eye(dim + 1, dtype=np.float64)
    if rank >= dim - 1:
        T[:dim, :dim] = U @ np.diag(d) @ Vt
    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T[:dim, :]


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Align a face using its 5 YuNet landmarks to ArcFace's canonical layout.

    `landmarks` is (5, 2) in YuNet order: right eye, left eye, nose,
    right mouth, left mouth. Returns an output_size x output_size BGR image.
    """
    dst = np.array(
        [landmarks[1], landmarks[0], landmarks[2], landmarks[4], landmarks[3]],
        dtype=np.float32,
    )
    M = _umeyama(dst, ARCFACE_SRC)
    return cv2.warpAffine(image_bgr, M, (output_size, output_size), borderValue=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_align.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/faceverify/align.py tests/test_align.py
git commit -m "feat: add align (ArcFace 5-point alignment)"
```

---

### Task 5: `detector.py` — YuNet face detection

**Files:**
- Create: `src/faceverify/detector.py`

**Interfaces:**
- Produces:
  - `@dataclass Face` with `bbox: tuple[int, int, int, int]` (x, y, w, h), `landmarks: np.ndarray` (5, 2), `score: float`
  - `class FaceDetector` with `__init__(self, model_path: str, score_threshold: float = 0.6)`, `detect(self, image_bgr) -> list[Face]`, `detect_largest(self, image_bgr) -> Face | None`

- [ ] **Step 1: Write implementation** (`src/faceverify/detector.py`)

```python
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
```

- [ ] **Step 2: Smoke-check the detector loads the model**

Run (interactive):
```bash
python -c "import sys; sys.path.insert(0, 'src'); from faceverify.detector import FaceDetector; d = FaceDetector('models/face_detection_yunet_2023mar.onnx'); print('detector ok')"
```
Expected: prints `detector ok` with no exception.

- [ ] **Step 3: Commit**

```bash
git add src/faceverify/detector.py
git commit -m "feat: add detector (YuNet)"
```

---

### Task 6: `embedder.py` — ArcFace embedding

**Files:**
- Create: `src/faceverify/embedder.py`

**Interfaces:**
- Produces: `class FaceEmbedder` with `__init__(self, model_path: str)`, `embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray` returning a 512-dim L2-normalized float32 vector.

- [ ] **Step 1: Write implementation** (`src/faceverify/embedder.py`)

```python
import cv2
import numpy as np
import onnxruntime as ort


class FaceEmbedder:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
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
```

- [ ] **Step 2: Smoke-check the embedder loads the model and returns 512-dim**

Run (interactive):
```bash
python -c "import sys, numpy as np; sys.path.insert(0, 'src'); from faceverify.embedder import FaceEmbedder; e = FaceEmbedder('models/w600k_r50.onnx'); v = e.embed(np.zeros((112,112,3), dtype=np.uint8)); print('embedding dim', v.shape, 'norm', round(float(np.linalg.norm(v)), 4))"
```
Expected: prints `embedding dim (512,) norm 1.0`.

- [ ] **Step 3: Commit**

```bash
git add src/faceverify/embedder.py
git commit -m "feat: add embedder (ArcFace)"
```

---

### Task 7: End-to-end smoke script

**Files:**
- Create: `scripts/smoke_test.py` (full content now)

**Interfaces:**
- Consumes: `detector`, `align`, `embedder`, `matcher` from prior tasks.

- [ ] **Step 1: Write the script** (`scripts/smoke_test.py`)

```python
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
```

- [ ] **Step 2: Run smoke test with real photos**

Ask the user for 2 photos of the same person + 1 photo of a different person. Then run:
```bash
python scripts/smoke_test.py personA_1.jpg personA_2.jpg personA_3.jpg   # expect match ~True
python scripts/smoke_test.py personA_1.jpg personA_2.jpg personB_1.jpg   # expect match ~False
```
Expected: first prints a high similarity with `match=True`; second a low similarity with `match=False`. If the match/no-match boundary is off, note the threshold may need adjusting (exposed in the GUI).

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "test: add end-to-end smoke script"
```

---

### Task 8: `gui.py` + `main.py` — the desktop app

**Files:**
- Create: `src/faceverify/gui.py`
- Create: `src/faceverify/main.py`

**Interfaces:**
- Consumes: `FaceDetector.detect_largest`, `align_face`, `FaceEmbedder.embed`, `mean_embedding`, `verify`, `RegistrationStore`.
- Produces: `class FaceVerifyApp` with `__init__(self, detector, embedder, store)` and `run(self)`; `main()` in `main.py`.

- [ ] **Step 1: Write `src/faceverify/gui.py`**

```python
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2
from PIL import Image, ImageTk

from faceverify.align import align_face
from faceverify.matcher import mean_embedding, verify


class FaceVerifyApp:
    def __init__(self, detector, embedder, store):
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.registration = store.load()
        self.preview_photo = None
        self.root = tk.Tk()
        self.root.title("人脸验证 (1:1)")
        self.threshold = tk.DoubleVar(value=0.35)
        self._build_ui()
        self._refresh_registration_label()

    def _build_ui(self):
        self.status_label = tk.Label(self.root, text="", font=("", 12))
        self.status_label.pack(pady=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="📁 注册照片", command=self.register).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="📁 测试照片", command=self.test).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🗑 删除注册", command=self.delete).pack(side=tk.LEFT, padx=5)

        self.preview_label = tk.Label(self.root, text="（预览）", width=56, height=20, bg="#eee")
        self.preview_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("", 14), fg="#333")
        self.result_label.pack(pady=5)

        tk.Label(self.root, text="阈值").pack()
        self.scale = tk.Scale(
            self.root, from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self.threshold, length=300,
        )
        self.scale.pack(pady=5)

    def _refresh_registration_label(self):
        if self.registration:
            self.status_label.config(
                text=f"当前注册: {self.registration['name']}（{self.registration['n_photos']} 张）"
            )
        else:
            self.status_label.config(text="当前注册: 无")

    def register(self):
        paths = filedialog.askopenfilenames(
            title="选择注册照片（可多选）",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not paths:
            return
        name = simpledialog.askstring("注册", "输入姓名（标签）：", parent=self.root)
        if not name:
            return
        if self.registration and not messagebox.askyesno(
            "覆盖确认", f"已注册 {self.registration['name']}，是否覆盖？"
        ):
            return
        embeddings = []
        used = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            face = self.detector.detect_largest(img)
            if face is None:
                messagebox.showwarning("提示", f"未检测到人脸，已跳过: {p}")
                continue
            embeddings.append(self.embedder.embed(align_face(img, face.landmarks)))
            used += 1
        if not embeddings:
            messagebox.showerror("注册失败", "所选照片中都没有检测到人脸")
            return
        self.store.save(name, mean_embedding(embeddings), used)
        self.registration = self.store.load()
        self._refresh_registration_label()
        messagebox.showinfo("完成", f"已注册 {name}（{used} 张）")

    def test(self):
        if not self.registration:
            messagebox.showwarning("提示", "请先注册照片")
            return
        path = filedialog.askopenfilename(
            title="选择测试照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误", "无法读取图片")
            return
        face = self.detector.detect_largest(img)
        if face is None:
            messagebox.showwarning("提示", "未检测到人脸")
            return
        self._show_preview(img, face)
        emb = self.embedder.embed(align_face(img, face.landmarks))
        is_match, sim = verify(emb, self.registration["embedding"], self.threshold.get())
        verdict = "✅ 是这个人" if is_match else "❌ 不是"
        self.result_label.config(text=f"{verdict}   相似度 {sim * 100:.1f}%")

    def _show_preview(self, img_bgr, face):
        h, w = img_bgr.shape[:2]
        scale = 280 / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        x, y, fw, fh = face.bbox
        cv2.rectangle(
            small,
            (int(x * scale), int(y * scale)),
            (int((x + fw) * scale), int((y + fh) * scale)),
            (0, 255, 0),
            2,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_label.config(image=self.preview_photo, width=rgb.shape[1], height=rgb.shape[0])

    def delete(self):
        if not self.registration:
            messagebox.showinfo("提示", "当前没有注册")
            return
        if messagebox.askyesno("删除确认", f"确定删除注册 {self.registration['name']} 吗？"):
            self.store.clear()
            self.registration = None
            self._refresh_registration_label()
            self.result_label.config(text="")
            self.preview_label.config(image="", text="（预览）")
            self.preview_photo = None

    def run(self):
        self.root.mainloop()
```

- [ ] **Step 2: Write `src/faceverify/main.py`**

```python
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
```

- [ ] **Step 3: Launch the app**

Run: `python run.py`
Expected: a window opens titled "人脸验证 (1:1)" with 注册/测试/删除 buttons. Verify the GUI opens without exceptions (models must already be in `models/`).

- [ ] **Step 4: Commit**

```bash
git add src/faceverify/gui.py src/faceverify/main.py
git commit -m "feat: add GUI and app entry point"
```

---

### Task 9: README, full test run, and push

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# face-verification-light

轻量级 1:1 人脸验证桌面软件。注册一个人若干张照片，然后对一张测试照片判断"是不是这个人"，并给出相似度分数。

## 技术栈

- 检测：OpenCV YuNet（`cv2.FaceDetectorYN`）
- 识别：InsightFace ArcFace（`w600k_r50` ONNX，onnxruntime 推理）
- 界面：tkinter

## 安装

```bash
pip install -r requirements.txt
```

## 下载模型

模型文件较大，需单独下载到 `models/`（已 gitignore）：

```bash
# YuNet 检测模型（约 230 KB）
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# ArcFace 识别模型（约 166 MB）
curl -L -o models/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o models/buffalo_l.zip -d models/buffalo_l
cp models/buffalo_l/w600k_r50.onnx models/w600k_r50.onnx
```

## 运行

```bash
python run.py
```

## 使用

1. 点「注册照片」，选一张或多张含人脸的图片，输入姓名。
2. 点「测试照片」，选一张图片，软件给出「是/不是 + 相似度」。
3. 阈值默认 0.35，可在界面拖动调整。

## 测试

```bash
python -m pytest
```

## 说明

- 识别率受照片角度、光照、遮挡、年龄变化影响；注册时多选不同角度/光照的照片可提升准确率。
- 若阈值 0.35 在你照片上不准，用界面滑块微调。
- 检测到多张脸时默认取最大的一张。
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -v`
Expected: all matcher/store/align tests pass (14 tests).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

- [ ] **Step 4: Push to GitHub**

Run:
```bash
git push -u origin main
```
Expected: pushed to https://github.com/NagaZhang/face-verification-light.git. Report the outcome (including any auth prompt). If push fails for auth/network reasons, report exactly what happened and ask the user how to proceed — do not retry blindly.

---

## Self-Review Notes

- Spec coverage: face-registration (register/replace/delete/persist) → Task 3 (store), Task 8 (gui register/delete/overwrite-confirm). face-verification (verify/no-face/multi-face/not-registered/threshold) → Task 2 (verify), Task 5 (detect_largest for multi-face + no-face), Task 8 (gui test/threshold). All requirements map to tasks.
- No placeholders: every code step contains full source.
- Type consistency: `detect_largest -> Face | None`, `align_face(img, landmarks)`, `embed -> ndarray`, `verify -> (bool, float)`, `RegistrationStore.load() -> dict | None` — consistent across all tasks.
