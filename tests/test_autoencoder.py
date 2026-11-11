import pytest

from conftest import load_example

DENSE_SCRIPTS = [
    "autoencoder/mnist_autoencoder.py",
    "autoencoder/mnist_autoencoder_l1.py",
    "autoencoder/mnist_deep_autoencoder.py",
]
ALL_SCRIPTS = DENSE_SCRIPTS + ["autoencoder/mnist_conv_autoencoder.py"]


@pytest.mark.parametrize("script", DENSE_SCRIPTS)
def test_dense_autoencoder_reconstructs_flat_images(script):
    model = load_example(script).build_model()

    assert model.input_shape == (None, 784)
    assert model.output_shape == (None, 784)


def test_conv_autoencoder_reconstructs_image_tensors():
    model = load_example("autoencoder/mnist_conv_autoencoder.py").build_model()

    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 28, 28, 1)


@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_main_trains_and_saves_reconstruction(script, fake_mnist, tmp_path):
    output = tmp_path / "reconstruction.png"

    load_example(script).main(["--epochs", "1", "--output", str(output)])

    assert output.read_bytes().startswith(b"\x89PNG")
