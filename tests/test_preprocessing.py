import numpy as np
import pytest
from PIL import Image

from conftest import load_example

AUGMENTATION_FILES = [
    "result_rotation.jpg",
    "result_width_shift.jpg",
    "result_height_shift.jpg",
    "result_shear.jpg",
    "result_zoom.jpg",
    "result_horizontal_flip.jpg",
    "result_vertical_flip.jpg",
]


@pytest.fixture
def image_file(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "input.jpg"
    Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)).save(path)
    return path


def test_zca_whitening_decorrelates_pixels():
    module = load_example("zca_whitening/zca_whitening.py")
    rng = np.random.default_rng(0)
    # 画素どうしに相関のある 2x2x3 の画像 2,000 枚
    mixing = rng.normal(size=(12, 12))
    x = (rng.normal(size=(2000, 12)) @ mixing).reshape(2000, 2, 2, 3)

    z = module.zca_whiten(x)

    assert z.shape == x.shape
    flat = z.reshape(len(z), -1)
    np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(flat.T @ flat / len(flat), np.eye(12), atol=1e-3)


def test_zca_whitening_main_saves_before_and_after(fake_cifar10, tmp_path):
    load_example("zca_whitening/zca_whitening.py").main(["--result-dir", str(tmp_path)])

    for name in ["zca_whitening_before.png", "zca_whitening_after.png"]:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_datagen_defaults_to_bundled_elephant_image():
    args = load_example("datagen/test_datagen3.py").parse_args([])

    assert args.image.endswith("vgg16/test_vgg16/elephant.jpg")
    assert args.result_dir == "result"
    assert Image.open(args.image).mode == "RGB"


def test_horizontal_flip_returns_original_or_mirrored_images(image_file):
    module = load_example("datagen/test_datagen3.py")
    image = np.asarray(Image.open(image_file), dtype="float32")

    augmented = module.augment(image, module.AUGMENTATIONS["horizontal_flip"]())

    assert augmented.shape == (9,) + image.shape
    for result in augmented:
        assert (np.allclose(result, image)
                or np.allclose(result, image[:, ::-1]))


def test_datagen_main_saves_one_grid_per_augmentation(image_file, tmp_path):
    load_example("datagen/test_datagen3.py").main(
        ["--image", str(image_file), "--result-dir", str(tmp_path)])

    assert sorted(p.name for p in tmp_path.glob("result_*.jpg")) == sorted(AUGMENTATION_FILES)
    for name in AUGMENTATION_FILES:
        assert Image.open(tmp_path / name).format == "JPEG"
