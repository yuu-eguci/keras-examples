import pytest

from conftest import load_example

DENSE_SCRIPTS = [
    "autoencoder/mnist_autoencoder.py",
    "autoencoder/mnist_autoencoder_l1.py",
    "autoencoder/mnist_deep_autoencoder.py",
]
ALL_SCRIPTS = DENSE_SCRIPTS + ["autoencoder/mnist_conv_autoencoder.py"]


@pytest.mark.parametrize("script", DENSE_SCRIPTS)
def test_dense_autoencoder_maps_flat_images_to_same_shape(script):
    model = load_example(script).build_model()

    assert model.input_shape == (None, 784)
    assert model.output_shape == (None, 784)


@pytest.mark.parametrize("script", [
    "autoencoder/mnist_autoencoder.py",
    "autoencoder/mnist_autoencoder_l1.py",
])
def test_encoded_layer_outputs_encoding_dim(script):
    module = load_example(script)

    encoded = module.build_model().get_layer("encoded")

    assert encoded.output.shape == (None, module.ENCODING_DIM)


def test_conv_autoencoder_maps_image_tensors_to_same_shape():
    model = load_example("autoencoder/mnist_conv_autoencoder.py").build_model()

    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 28, 28, 1)


def test_l1_autoencoder_regularizes_encoded_activity():
    model = load_example("autoencoder/mnist_autoencoder_l1.py").build_model()

    assert model.get_layer("encoded").activity_regularizer is not None


@pytest.mark.parametrize("script, epochs", [
    ("autoencoder/mnist_autoencoder.py", 100),
    ("autoencoder/mnist_autoencoder_l1.py", 100),
    ("autoencoder/mnist_deep_autoencoder.py", 100),
    ("autoencoder/mnist_conv_autoencoder.py", 50),
])
def test_default_arguments(script, epochs):
    args = load_example(script).parse_args([])

    assert args.epochs == epochs
    assert args.output == "reconstruction.png"


@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_main_trains_and_saves_reconstruction(script, fake_mnist, tmp_path):
    output = tmp_path / "reconstruction.png"

    load_example(script).main(["--epochs", "1", "--output", str(output)])

    assert output.read_bytes().startswith(b"\x89PNG")
