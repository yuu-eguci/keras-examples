"""ImageNet で学習済みの VGG16 で画像を分類し、確率の高い5クラスを表示する。"""
import argparse

import keras
import numpy as np


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 全結合層を含む VGG16 と学習済みの重みを読み込む
    model = keras.applications.VGG16(weights="imagenet")

    # VGG16 の入力サイズ 224x224 にそろえ、(1, 縦, 横, チャネル) の4次元テンソルにする
    image = keras.utils.load_img(args.image, target_size=(224, 224))
    x = keras.utils.img_to_array(image)[np.newaxis]

    preds = model.predict(keras.applications.vgg16.preprocess_input(x))
    # 1000クラスの番号を ImageNet のクラス名に変換する
    for result in keras.applications.vgg16.decode_predictions(preds, top=5)[0]:
        print(result)


if __name__ == "__main__":
    main()
