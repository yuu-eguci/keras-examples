"""CIFAR-10 の訓練画像に ZCA 白色化をかけ、白色化前後の画像を保存する。"""
import argparse
import os

import keras
import numpy as np
from matplotlib.figure import Figure

EPS = 1e-6


def zca_whiten(x, eps=EPS):
    """画像集合 x（先頭の軸が画像）に ZCA 白色化をかけ、同じ形状で返す。"""
    # 1枚の画像を1本のベクトルとみなし、全画像の平均を引く
    flat = x.reshape(len(x), -1)
    flat = flat - flat.mean(axis=0)
    # 共分散行列を固有値分解し、白色化行列 U diag(1/sqrt(S + eps)) U^T を作る
    cov = flat.T @ flat / len(flat)
    u, s, _ = np.linalg.svd(cov)
    zca = (u / np.sqrt(s + eps)) @ u.T
    return (flat @ zca.T).reshape(x.shape)


def save_images(x, path):
    """先頭16枚を、画像ごとに 0〜1 に正規化して4×4に並べて保存する。"""
    fig = Figure()
    for i, image in enumerate(x[:16]):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow((image - image.min()) / (image.max() - image.min()))
        ax.axis("off")
    fig.savefig(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="result")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.result_dir, exist_ok=True)

    (x, _), _ = keras.datasets.cifar10.load_data()
    x = x.astype("float32") / 255.0

    save_images(x, os.path.join(args.result_dir, "zca_whitening_before.png"))
    save_images(zca_whiten(x), os.path.join(args.result_dir, "zca_whitening_after.png"))


if __name__ == "__main__":
    main()
