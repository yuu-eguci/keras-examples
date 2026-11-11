import runpy

import keras
import numpy as np
import pytest

from conftest import REPO_ROOT, load_example

HISTORY_HEADER = "epoch\tloss\tacc\tval_loss\tval_acc"

MNIST_SCRIPTS = ["mlp/mnist.py", "cnn/mnist/mnist.py"]
CIFAR10_SCRIPTS = ["mlp/cifar10.py", "cnn/cifar10/cifar10.py"]


@pytest.mark.parametrize("script, input_shape", [
    ("mlp/mnist.py", (None, 784)),
    ("cnn/mnist/mnist.py", (None, 28, 28, 1)),
    ("mlp/cifar10.py", (None, 3072)),
    ("cnn/cifar10/cifar10.py", (None, 32, 32, 3)),
])
def test_model_outputs_probabilities_for_ten_classes(script, input_shape):
    model = load_example(script).build_model()

    assert model.input_shape == input_shape
    assert model.output_shape == (None, 10)
    probabilities = model.predict(np.zeros((2,) + input_shape[1:]), verbose=0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("script, epochs", [
    ("mlp/mnist.py", 100),
    ("cnn/mnist/mnist.py", 100),
    ("mlp/cifar10.py", 200),
    ("cnn/cifar10/cifar10.py", 50),
])
def test_default_arguments(script, epochs):
    args = load_example(script).parse_args([])

    assert args.epochs == epochs
    assert args.result_dir == "result"


def test_cifar10_cnn_augments_only_when_requested():
    module = load_example("cnn/cifar10/cifar10.py")

    def has_translation(model):
        return any(isinstance(layer, keras.layers.RandomTranslation)
                   for layer in model.layers)

    assert not module.parse_args([]).augment
    assert not has_translation(module.build_model())
    assert has_translation(module.build_model(augment=True))


def test_cifar10_cnn_augmentation_fills_with_nearest_pixels():
    model = load_example("cnn/cifar10/cifar10.py").build_model(augment=True)

    translation = model.layers[0]

    assert isinstance(translation, keras.layers.RandomTranslation)
    assert translation.fill_mode == "nearest"


@pytest.fixture
def fit_calls(monkeypatch):
    """Model.fit に渡された引数を記録する（学習自体はそのまま行う）。"""
    calls = []
    original_fit = keras.Model.fit

    def recording_fit(self, *args, **kwargs):
        calls.append(kwargs)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "fit", recording_fit)
    return calls


@pytest.mark.parametrize("script", MNIST_SCRIPTS)
def test_mnist_validates_on_train_split_and_stops_early(
        script, fake_mnist, fit_calls, tmp_path):
    load_example(script).main(["--epochs", "1", "--result-dir", str(tmp_path)])

    (kwargs,) = fit_calls
    assert kwargs["validation_split"] == 0.1
    assert any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in kwargs["callbacks"])


@pytest.mark.parametrize("script", CIFAR10_SCRIPTS)
def test_cifar10_validates_on_test_data(script, fake_cifar10, fit_calls, tmp_path):
    load_example(script).main(["--epochs", "1", "--result-dir", str(tmp_path)])

    (kwargs,) = fit_calls
    x_val, y_val = kwargs["validation_data"]
    assert len(x_val) == len(y_val) == 16


def assert_history_has_one_epoch(result_dir):
    lines = (result_dir / "history.txt").read_text().splitlines()
    assert lines[0] == HISTORY_HEADER
    assert len(lines) == 2
    assert len(lines[1].split("\t")) == 5


def assert_png(path):
    assert path.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("script, extra_files", [
    ("mlp/mnist.py", []),
    ("cnn/mnist/mnist.py", ["weight_before.png", "weight_after.png"]),
])
def test_mnist_main_saves_results(script, extra_files, fake_mnist, tmp_path):
    result_dir = tmp_path / "result"

    load_example(script).main(["--epochs", "1", "--result-dir", str(result_dir)])

    assert_history_has_one_epoch(result_dir)
    for name in ["acc.png", "loss.png"] + extra_files:
        assert_png(result_dir / name)


def test_mlp_cifar10_main_saves_results(fake_cifar10, tmp_path):
    result_dir = tmp_path / "result"

    load_example("mlp/cifar10.py").main(
        ["--epochs", "1", "--result-dir", str(result_dir)])

    assert_history_has_one_epoch(result_dir)
    assert_png(result_dir / "acc.png")
    assert_png(result_dir / "loss.png")


@pytest.mark.parametrize("augment", [False, True])
def test_cnn_cifar10_results_feed_filter_visualization(augment, fake_cifar10, tmp_path):
    result_dir = tmp_path / "result"
    args = ["--epochs", "1", "--result-dir", str(result_dir)]

    load_example("cnn/cifar10/cifar10.py").main(args + (["--augment"] if augment else []))
    load_example("cnn/cifar10/visualize_filters.py").main(["--result-dir", str(result_dir)])

    assert_history_has_one_epoch(result_dir)
    for name in ["acc.png", "loss.png", "plot.png", "filters.png"]:
        assert_png(result_dir / name)
    assert isinstance(keras.models.load_model(result_dir / "model.keras"), keras.Model)


def test_plot_results_compares_histories(monkeypatch, tmp_path):
    for name in ["result_without_datagen", "result_with_datagen"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "history.txt").write_text(
            HISTORY_HEADER + "\n0\t1.0\t0.5\t1.1\t0.4\n1\t0.9\t0.6\t1.0\t0.5\n")
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(REPO_ROOT / "cnn/cifar10/plot_results.py"))

    assert_png(tmp_path / "val_loss.png")
    assert_png(tmp_path / "val_acc.png")
