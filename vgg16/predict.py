"""transfer_learning.py で学習したモデルで画像を分類し、確率の高いクラスを表示する。"""
import argparse
import os

import keras
import numpy as np

import transfer_learning  # noqa: F401  モデルに含まれる前処理レイヤを読み込めるようにする

TOP = 5


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="transfer_learning.py が保存した .keras ファイル")
    parser.add_argument("images", nargs="+")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = keras.models.load_model(args.model)
    with open(os.path.join(os.path.dirname(args.model), "classes.txt")) as fp:
        classes = fp.read().splitlines()
    image_size = model.input_shape[1:3]

    for path in args.images:
        x = keras.utils.img_to_array(keras.utils.load_img(path, target_size=image_size))
        probabilities = model.predict(x[np.newaxis], verbose=0)[0]
        print(path)
        for i in np.argsort(probabilities)[::-1][:TOP]:
            print("  %s: %.4f" % (classes[i], probabilities[i]))


if __name__ == "__main__":
    main()
