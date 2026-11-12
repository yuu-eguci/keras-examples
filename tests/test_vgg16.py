import runpy

import keras
import numpy as np
import pytest
from PIL import Image

from conftest import REPO_ROOT, load_example

CLASSES = ["daisy", "iris", "tulip"]
METHODS = ["smallcnn", "extractor", "finetuning", "scratch"]


@pytest.fixture(autouse=True)
def vgg16_dir_on_path(monkeypatch):
    # predict.py は同じディレクトリの transfer_learning.py を import する
    monkeypatch.syspath_prepend(str(REPO_ROOT / "vgg16"))


@pytest.fixture
def no_pretrained_weights(monkeypatch):
    """ImageNet の学習済み重みをダウンロードせず、ランダムな重みの VGG16 を使う。

    スクリプトが要求した weights の値を記録したリストを返す。
    """
    original = keras.applications.VGG16
    requested = []

    def vgg16_without_download(*args, **kwargs):
        requested.append(kwargs.get("weights", "imagenet"))
        kwargs["weights"] = None
        return original(*args, **kwargs)

    monkeypatch.setattr(keras.applications, "VGG16", vgg16_without_download)
    return requested


def write_images(directory, per_class, rng):
    for name in CLASSES:
        (directory / name).mkdir(parents=True)
        for i in range(per_class):
            pixels = rng.integers(0, 256, (40, 36, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / name / f"{i}.jpg")


@pytest.fixture
def image_dirs(tmp_path):
    rng = np.random.default_rng(0)
    train_dir, validation_dir = tmp_path / "train", tmp_path / "validation"
    write_images(train_dir, 4, rng)
    write_images(validation_dir, 2, rng)
    return train_dir, validation_dir


def train(train_dir, validation_dir, result_dir, *extra):
    load_example("vgg16/transfer_learning.py").main([
        "--train-dir", str(train_dir), "--validation-dir", str(validation_dir),
        "--epochs", "1", "--image-size", "32", "--result-dir", str(result_dir), *extra])


def test_default_arguments():
    args = load_example("vgg16/transfer_learning.py").parse_args(
        ["--method", "smallcnn", "--train-dir", "t", "--validation-dir", "v"])

    assert args.preprocess == "rescale"
    assert args.epochs == 50
    assert args.image_size == 150
    assert args.result_dir == "results"
    assert args.name is None
    assert args.top_weights is None


def test_load_images_uses_sorted_subdirectories_as_classes(image_dirs):
    module = load_example("vgg16/transfer_learning.py")

    x, y, classes = module.load_images(image_dirs[0], 32)

    assert classes == CLASSES
    assert x.shape == (12, 32, 32, 3)
    assert sorted(y.tolist()) == [0] * 4 + [1] * 4 + [2] * 4


def test_load_images_skips_hidden_directories(image_dirs):
    module = load_example("vgg16/transfer_learning.py")
    (image_dirs[0] / ".ipynb_checkpoints").mkdir()

    _, _, classes = module.load_images(image_dirs[0], 32)

    assert classes == CLASSES


def test_vgg16_preprocess_layer_matches_keras_applications():
    module = load_example("vgg16/transfer_learning.py")
    x = np.random.default_rng(0).uniform(0, 255, (2, 4, 4, 3)).astype("float32")

    actual = keras.ops.convert_to_numpy(module.VGG16Preprocess()(x))

    expected = keras.applications.vgg16.preprocess_input(x.copy())
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_finetuning_trains_only_block5_and_top(no_pretrained_weights):
    module = load_example("vgg16/transfer_learning.py")

    model = module.build_model("finetuning", 3, 32, "rescale")

    vgg16 = model.get_layer("vgg16")
    weighted = [layer for layer in vgg16.layers if layer.weights]
    assert all(layer.trainable == layer.name.startswith("block5") for layer in weighted)
    assert model.get_layer("top").trainable


def test_finetuning_starts_from_given_top_weights(no_pretrained_weights, tmp_path):
    module = load_example("vgg16/transfer_learning.py")
    top = module.build_top((1, 1, 512), 3)
    top.save_weights(tmp_path / "top.weights.h5")

    model = module.build_model("finetuning", 3, 32, "rescale",
                               top_weights=str(tmp_path / "top.weights.h5"))

    for expected, actual in zip(top.get_weights(), model.get_layer("top").get_weights()):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("method", METHODS)
def test_main_saves_history_model_and_classes(method, image_dirs, no_pretrained_weights, tmp_path):
    train(*image_dirs, tmp_path, "--method", method)

    lines = (tmp_path / f"history_{method}.txt").read_text().splitlines()
    assert lines[0] == "epoch\tloss\tacc\tval_loss\tval_acc"
    assert len(lines) == 2
    assert (tmp_path / "classes.txt").read_text().split() == CLASSES
    model = keras.models.load_model(tmp_path / f"model_{method}.keras")
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 3)
    assert (tmp_path / "top_extractor.weights.h5").exists() == (method == "extractor")


def test_vgg16_preprocess_is_saved_in_model_and_used_by_predict(
        image_dirs, no_pretrained_weights, tmp_path, capsys):
    train(*image_dirs, tmp_path, "--method", "smallcnn", "--preprocess", "vgg16")
    model_path = tmp_path / "model_smallcnn_with_preprocess.keras"
    image = image_dirs[1] / "tulip" / "0.jpg"
    capsys.readouterr()

    load_example("vgg16/predict.py").main(["--model", str(model_path), str(image)])

    model = keras.models.load_model(model_path)
    assert "VGG16Preprocess" in [type(layer).__name__ for layer in model.layers]
    output = capsys.readouterr().out
    assert str(image) in output
    assert all(name in output for name in CLASSES)


def test_name_overrides_output_file_names(image_dirs, no_pretrained_weights, tmp_path):
    train(*image_dirs, tmp_path, "--method", "smallcnn", "--name", "trial1")

    assert (tmp_path / "history_trial1.txt").exists()
    assert (tmp_path / "model_trial1.keras").exists()


def test_imagenet_prediction_prints_top5(no_pretrained_weights, monkeypatch, capsys):
    def fake_decode_predictions(preds, top):
        assert preds.shape == (1, 1000)
        return [[("n0000", "class%d" % i, 0.1) for i in range(top)]]

    monkeypatch.setattr(keras.applications.vgg16, "decode_predictions", fake_decode_predictions)

    load_example("vgg16/test_vgg16/test_vgg16.py").main(
        [str(REPO_ROOT / "vgg16/test_vgg16/elephant.jpg")])

    assert capsys.readouterr().out.count("class") == 5


@pytest.mark.parametrize("method", METHODS)
def test_augmentation_is_used_only_while_training(method, image_dirs, no_pretrained_weights, tmp_path):
    module = load_example("vgg16/transfer_learning.py")
    wrapped = []
    original = module.with_augmentation

    def recording_with_augmentation(model):
        trainer = original(model)
        wrapped.append({type(layer).__name__ for layer in trainer.layers})
        return trainer

    module.with_augmentation = recording_with_augmentation
    module.main(["--train-dir", str(image_dirs[0]), "--validation-dir", str(image_dirs[1]),
                 "--epochs", "1", "--image-size", "32", "--result-dir", str(tmp_path),
                 "--method", method])

    if method == "extractor":
        assert wrapped == []
    else:
        (layer_types,) = wrapped
        assert {"RandomFlip", "RandomZoom", "RandomShear"} <= layer_types
    saved = keras.models.load_model(tmp_path / f"model_{method}.keras")
    assert not any(type(layer).__name__.startswith("Random") for layer in saved.layers)


def test_17flowers_setup_splits_images_by_class(tmp_path, monkeypatch):
    labels = (REPO_ROOT / "vgg16/17flowers/labels.txt").read_text()
    (tmp_path / "labels.txt").write_text(labels)
    (tmp_path / "jpg").mkdir()
    for i in range(1, 1361):
        (tmp_path / "jpg" / f"image_{i:04d}.jpg").touch()
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(REPO_ROOT / "vgg16/17flowers/setup.py"))

    classes = [line.split()[2] for line in labels.splitlines()]
    assert sorted(p.name for p in (tmp_path / "train_images").iterdir()) == sorted(classes)
    for name in classes:
        assert len(list((tmp_path / "train_images" / name).iterdir())) == 70
        assert len(list((tmp_path / "test_images" / name).iterdir())) == 10


def test_dogs_vs_cats_setup_splits_images(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    for animal in ["dog", "cat"]:
        for i in range(1, 1401):
            (tmp_path / "train" / f"{animal}.{i}.jpg").touch()
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(REPO_ROOT / "vgg16/dogs_vs_cats/setup.py"))

    for split, count in [("train", 1000), ("validation", 400)]:
        for name in ["dogs", "cats"]:
            assert len(list((tmp_path / "data" / split / name).iterdir())) == count


def test_top_weights_is_only_for_finetuning():
    module = load_example("vgg16/transfer_learning.py")

    with pytest.raises(SystemExit):
        module.parse_args(["--method", "extractor", "--train-dir", "t",
                           "--validation-dir", "v", "--top-weights", "top.weights.h5"])


@pytest.mark.parametrize("method, weights, optimizer", [
    ("smallcnn", [], "adam"),
    ("extractor", ["imagenet"], "sgd"),
    ("finetuning", ["imagenet"], "sgd"),
    ("scratch", [None], "adam"),
])
def test_method_uses_documented_weights_and_optimizer(
        method, weights, optimizer, image_dirs, no_pretrained_weights, monkeypatch, tmp_path):
    compiled = []
    original_compile = keras.Model.compile

    def recording_compile(self, *args, **kwargs):
        compiled.append(kwargs["optimizer"])
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "compile", recording_compile)

    train(*image_dirs, tmp_path, "--method", method)

    assert no_pretrained_weights == weights
    (used,) = compiled
    if optimizer == "adam":
        assert used == "adam"
    else:
        assert isinstance(used, keras.optimizers.SGD)
        assert float(used.learning_rate) == pytest.approx(1e-4)
        assert used.momentum == pytest.approx(0.9)


@pytest.mark.parametrize("dataset, names", [
    ("17flowers", ["smallcnn", "finetuning"]),
    ("dogs_vs_cats", ["smallcnn", "extractor", "finetuning", "finetuning_from_random"]),
])
def test_plot_results_compares_documented_histories(dataset, names, tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    for name in names:
        (tmp_path / "results" / f"history_{name}.txt").write_text(
            "epoch\tloss\tacc\tval_loss\tval_acc\n0\t1.0\t0.5\t1.1\t0.4\n")
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(REPO_ROOT / "vgg16" / dataset / "plot_results.py"))

    for image in ["val_loss.png", "val_acc.png"]:
        assert (tmp_path / "results" / image).read_bytes().startswith(b"\x89PNG")
