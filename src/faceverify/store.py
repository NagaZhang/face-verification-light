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
