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
