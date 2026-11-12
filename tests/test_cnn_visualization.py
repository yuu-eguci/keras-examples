import json

import keras
import numpy as np
import pytest
from PIL import Image

from conftest import REPO_ROOT, load_example


@pytest.fixture
def visualize(monkeypatch, no_pretrained_weights):
    module = load_example("dream/visualize_vgg16.py")
    # テストでは小さな画像とランダムな重みの VGG16 を使う
    monkeypatch.setattr(module, "IMAGE_SIZE", 32)
    return module


def test_predictions_layer_outputs_scores_before_softmax(visualize):
    model = visualize.build_model()

    assert model.input_shape == (None, 32, 32, 3)
    assert model.get_layer("predictions").activation is keras.activations.linear


def test_visualize_filter_increases_objective(visualize):
    model = visualize.build_model()

    image, objectives = visualize.visualize_filter(
        model, "block1_conv1", 3, iterations=20, rng=np.random.default_rng(0))

    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
    assert len(objectives) == 20
    assert objectives[-1] > objectives[0]


def test_class_names_come_from_bundled_imagenet_index(visualize):
    names = visualize.class_names()

    index = json.loads((REPO_ROOT / "imagenet/imagenet_class_index.json").read_text())
    assert len(names) == 1000
    assert names[0] == index["0"][1]


@pytest.mark.parametrize("layer", ["block1_conv1", "predictions"])
def test_visualize_main_saves_grid(visualize, layer, tmp_path):
    output = tmp_path / "result.png"

    visualize.main(["--layer", layer, "--iterations", "1", "--output", str(output)])

    assert output.read_bytes().startswith(b"\x89PNG")


def test_visualize_rejects_unknown_layer(visualize):
    with pytest.raises(SystemExit):
        visualize.main(["--layer", "no_such_layer", "--iterations", "1"])


@pytest.fixture
def dream(no_pretrained_weights):
    return load_example("dream/deep_dream.py")


def test_dream_defaults():
    args = load_example("dream/deep_dream.py").parse_args(["in.jpg", "out"])

    assert args.setting == "dreamy"
    assert args.iterations == 10
    assert args.image_size == 600


def test_deprocess_restores_preprocessed_image(dream, tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "in.png")

    x = dream.preprocess_image(str(tmp_path / "in.png"), 16)

    assert x.shape == (1, 16, 16, 3)
    np.testing.assert_array_equal(dream.deprocess_image(x), pixels)


@pytest.mark.parametrize("setting", ["dreamy", "bad_trip"])
def test_dream_gradient_points_downhill(dream, setting):
    model = dream.build_model(96)
    loss_and_grads = dream.make_loss_and_grads(model, dream.SETTINGS[setting], 96)
    x = np.random.default_rng(0).normal(0, 20, 96 * 96 * 3)

    loss, grads = loss_and_grads(x)
    smaller, _ = loss_and_grads(x - 1e-3 * grads / np.abs(grads).max())

    assert grads.shape == x.shape
    assert grads.dtype == np.float64
    assert smaller < loss


def test_dream_main_saves_each_iteration(dream, tmp_path):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)).save(tmp_path / "in.jpg")
    prefix = tmp_path / "out"

    dream.main([str(tmp_path / "in.jpg"), str(prefix), "--iterations", "2", "--image-size", "96"])

    for i in range(2):
        image = Image.open(f"{prefix}_at_iteration_{i}.png")
        assert image.size == (96, 96)


def test_objective_is_activation_minus_lp_and_tv_penalties(visualize):
    model = visualize.build_model()
    x = np.random.default_rng(0).uniform(0, 255, (1, 32, 32, 3)).astype("float32")

    value, _ = visualize.objective_and_grads(model, "block1_conv1")(x, 5)

    output = np.asarray(keras.Model(model.input, model.get_layer("block1_conv1").output)(x))
    x64, size = x.astype("float64"), 32 * 32 * 3
    lp = np.sum(np.abs(x64) ** 6) ** (1 / 6)
    a = np.square(x64[:, 1:, :-1, :] - x64[:, :-1, :-1, :])
    b = np.square(x64[:, :-1, 1:, :] - x64[:, :-1, :-1, :])
    expected = output[..., 5].mean() - 10 * lp / size - 10 * np.sum(a + b) / size
    np.testing.assert_allclose(float(value), expected, rtol=1e-4)


def test_visualize_rejects_input_layer(visualize, monkeypatch):
    model = visualize.build_model()
    monkeypatch.setattr(visualize, "build_model", lambda: model)

    with pytest.raises(SystemExit):
        visualize.main(["--layer", model.layers[0].name, "--iterations", "1"])


def test_dream_loss_rewards_features_and_penalizes_roughness(dream):
    settings = dream.SETTINGS["dreamy"]
    model = dream.build_model(96)
    x = np.random.default_rng(0).normal(0, 20, (1, 96, 96, 3))

    loss, _ = dream.make_loss_and_grads(model, settings, 96)(x.ravel())

    size = 96 * 96 * 3
    expected = 0.0
    for name, coeff in settings["features"].items():
        output = np.asarray(keras.Model(model.input, model.get_layer(name).output)(x.astype("float32")))
        _, h, w, _ = output.shape
        expected -= coeff * np.sum(np.square(output[:, 2:h - 2, 2:w - 2, :])) / output[0].size
    a = np.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = np.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    expected += settings["continuity"] * np.sum((a + b) ** 1.25) / size
    expected += settings["dream_l2"] * np.sum(np.square(x)) / size
    np.testing.assert_allclose(loss, expected, rtol=1e-4)
