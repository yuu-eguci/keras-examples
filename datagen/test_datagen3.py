"""1枚の画像に Keras のデータ拡張レイヤを繰り返しかけ、拡張の種類ごとに結果を並べて保存する。"""
import argparse
import os

import keras
import numpy as np
from matplotlib.figure import Figure

DEFAULT_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "vgg16", "test_vgg16", "elephant.jpg")

layers = keras.layers
# 拡張の名前と、そのレイヤを作る関数。画像の外側になった部分は端の画素で埋める
AUGMENTATIONS = {
    "rotation": lambda: layers.RandomRotation(0.25, fill_mode="nearest"),  # ±90度
    "width_shift": lambda: layers.RandomTranslation(0, 0.2, fill_mode="nearest"),
    "height_shift": lambda: layers.RandomTranslation(0.2, 0, fill_mode="nearest"),
    "shear": lambda: layers.RandomShear(x_factor=1.0, fill_mode="nearest"),  # 最大約45度
    "zoom": lambda: layers.RandomZoom(0.5, fill_mode="nearest"),
    "horizontal_flip": lambda: layers.RandomFlip("horizontal"),
    "vertical_flip": lambda: layers.RandomFlip("vertical"),
}


def augment(image, layer, n=9):
    """同じ画像を n 枚並べ、1枚ずつ独立にデータ拡張をかけた結果を返す。"""
    batch = np.repeat(image[np.newaxis], n, axis=0)
    return keras.ops.convert_to_numpy(layer(batch, training=True))


def save_grid(images, path):
    fig = Figure()
    for i, image in enumerate(images):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(np.clip(image, 0, 255).astype("uint8"), aspect="auto")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=os.path.normpath(DEFAULT_IMAGE))
    parser.add_argument("--result-dir", default="result")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.result_dir, exist_ok=True)

    image = keras.utils.img_to_array(keras.utils.load_img(args.image))
    for name, make_layer in AUGMENTATIONS.items():
        save_grid(augment(image, make_layer()),
                  os.path.join(args.result_dir, "result_%s.jpg" % name))


if __name__ == "__main__":
    main()
