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
    """ImageNet の学習済み重みをダウンロードせず、ランダムな重みの VGG16 を使う。"""
    original = keras.applications.VGG16

    def vgg16_without_download(*args, **kwargs):
        kwargs["weights"] = None
        return original(*args, **kwargs)

    monkeypatch.setattr(keras.applications, "VGG16", vgg16_without_download)


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
