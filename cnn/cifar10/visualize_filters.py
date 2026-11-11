"""cifar10.py で学習したモデルの1層目の畳み込みフィルタを画像として保存する。"""
import argparse
import os

import keras
from matplotlib.figure import Figure


def save_filters(model, path):
    """1層目の畳み込みフィルタを、フィルタごとに 0〜1 に正規化した RGB 画像で並べて保存する。"""
    conv = next(layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D))
    # 重みの形状は (縦, 横, 入力チャネル, フィルタ数)
    weights = conv.get_weights()[0]
    fig = Figure()
    for i in range(weights.shape[3]):
        w = weights[:, :, :, i]
        ax = fig.add_subplot(4, 8, i + 1)
        ax.imshow((w - w.min()) / (w.max() - w.min()))
        ax.axis("off")
    fig.savefig(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="result",
                        help="model.keras があり、filters.png を保存するディレクトリ")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = keras.models.load_model(os.path.join(args.result_dir, "model.keras"))
    model.summary()
    save_filters(model, os.path.join(args.result_dir, "filters.png"))


if __name__ == "__main__":
    main()
