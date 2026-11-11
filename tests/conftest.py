import importlib.util
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_example(relative_path):
    """リポジトリ内のスクリプトをモジュールとして読み込む。"""
    path = REPO_ROOT / relative_path
    name = "example_" + relative_path.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_mnist(monkeypatch):
    """MNIST のダウンロードの代わりに、同じ形式の極小データを返す。"""
    rng = np.random.default_rng(0)

    def load_data():
        x_train = rng.integers(0, 256, size=(32, 28, 28), dtype=np.uint8)
        y_train = rng.integers(0, 10, size=(32,), dtype=np.uint8)
        x_test = rng.integers(0, 256, size=(16, 28, 28), dtype=np.uint8)
        y_test = rng.integers(0, 10, size=(16,), dtype=np.uint8)
        return (x_train, y_train), (x_test, y_test)

    monkeypatch.setattr(keras.datasets.mnist, "load_data", load_data)


@pytest.fixture
def fake_cifar10(monkeypatch):
    """CIFAR-10 のダウンロードの代わりに、同じ形式の極小データを返す。"""
    rng = np.random.default_rng(0)

    def load_data():
        x_train = rng.integers(0, 256, size=(32, 32, 32, 3), dtype=np.uint8)
        y_train = rng.integers(0, 10, size=(32, 1), dtype=np.uint8)
        x_test = rng.integers(0, 256, size=(16, 32, 32, 3), dtype=np.uint8)
        y_test = rng.integers(0, 10, size=(16, 1), dtype=np.uint8)
        return (x_train, y_train), (x_test, y_test)

    monkeypatch.setattr(keras.datasets.cifar10, "load_data", load_data)
