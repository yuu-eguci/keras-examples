import numpy as np
import pytest

from conftest import load_example


def snapshot(model):
    return [w.numpy().copy() for w in model.weights]


def changed(before, model):
    return any(not np.array_equal(a, b) for a, b in zip(before, snapshot(model)))


@pytest.mark.parametrize("script, epochs", [
    ("vae/mnist_vae.py", 50),
    ("gan/dcgan_mnist.py", 20),
])
def test_default_arguments(script, epochs):
    args = load_example(script).parse_args([])

    assert args.epochs == epochs
    assert args.result_dir == "result"


def test_vae_models_share_latent_space():
    vae, encoder, generator = load_example("vae/mnist_vae.py").build_models()

    assert vae.input_shape == (None, 784)
    assert vae.output_shape == (None, 784)
    assert encoder.output_shape == (None, 2)
    assert generator.input_shape == (None, 2)
    assert generator.output_shape == (None, 784)


def test_vae_adds_kl_divergence_to_loss():
    vae, _, _ = load_example("vae/mnist_vae.py").build_models()

    vae(np.random.default_rng(0).random((4, 784), dtype=np.float32))

    (kl,) = vae.losses
    assert float(kl) >= 0.0


def test_vae_main_saves_latent_space_plots(fake_mnist, tmp_path):
    load_example("vae/mnist_vae.py").main(["--epochs", "1", "--result-dir", str(tmp_path)])

    for name in ["latent.png", "manifold.png"]:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_dcgan_models_connect_noise_to_judgement():
    module = load_example("gan/dcgan_mnist.py")

    generator, discriminator, dcgan = module.build_models()

    assert generator.input_shape == (None, 100)
    assert generator.output_shape == (None, 28, 28, 1)
    assert discriminator.output_shape == (None, 1)
    assert dcgan.output_shape == (None, 1)


def test_dcgan_updates_only_the_model_being_trained():
    module = load_example("gan/dcgan_mnist.py")
    generator, discriminator, dcgan = module.build_models()
    rng = np.random.default_rng(0)
    real = rng.uniform(-1, 1, (4, 28, 28, 1)).astype("float32")

    # Generator の更新を挟んだあとも Discriminator が学習できることを確認するため2回繰り返す
    for _ in range(2):
        d_before, g_before = snapshot(discriminator), snapshot(generator)
        module.train_discriminator(discriminator, real, generator.predict(module.make_noise(rng, 4), verbose=0))
        assert changed(d_before, discriminator)
        assert not changed(g_before, generator)

        d_before, g_before = snapshot(discriminator), snapshot(generator)
        module.train_generator(dcgan, discriminator, module.make_noise(rng, 4))
        assert changed(g_before, generator)
        assert not changed(d_before, discriminator)


def test_dcgan_main_saves_generated_images_and_weights(fake_mnist, tmp_path):
    load_example("gan/dcgan_mnist.py").main(["--epochs", "1", "--result-dir", str(tmp_path)])

    assert (tmp_path / "0000_0000.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "generator.weights.h5").stat().st_size > 0
    assert (tmp_path / "discriminator.weights.h5").stat().st_size > 0
