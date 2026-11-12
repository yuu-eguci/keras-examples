"""2つの試験の点数から合否を予測する2クラスロジスティック回帰。"""
import argparse
import os

import keras
import numpy as np
from matplotlib.figure import Figure

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "ex2data1.txt")


def load_data():
    """各次元を平均0・分散1に標準化した点数 x と、合否のラベル t を返す。"""
    data = np.genfromtxt(DATA_FILE, delimiter=",")
    x = data[:, :2]
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x, data[:, 2]


def build_model():
    return keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


def plot_data(ax, x, t):
    # ラベル1を正例、ラベル0を負例とする
    ax.scatter(x[t == 1, 0], x[t == 1, 1], c="red", marker="o", label="positive")
    ax.scatter(x[t == 0, 0], x[t == 0, 1], c="blue", marker="o", label="negative")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--result-dir", default="result")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.result_dir, exist_ok=True)
    # 結果を再現できるように乱数の種を固定
    keras.utils.set_random_seed(1)

    x, t = load_data()
    fig = Figure()
    plot_data(fig.add_subplot(), x, t)
    fig.savefig(os.path.join(args.result_dir, "data.png"))

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, t, epochs=args.epochs, batch_size=5)

    # 決定境界 w1 * x1 + w2 * x2 + b = 0 を x2 について解いて描く
    weights, bias = model.layers[0].get_weights()
    w1, w2, b = weights[0, 0], weights[1, 0], bias[0]
    fig = Figure()
    ax = fig.add_subplot()
    plot_data(ax, x, t)
    xs = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    ax.plot(xs, -(w1 / w2) * xs - (b / w2), "b-", label="decision boundary")
    ax.set_xlim(x[:, 0].min(), x[:, 0].max())
    ax.set_ylim(x[:, 1].min(), x[:, 1].max())
    ax.legend()
    fig.savefig(os.path.join(args.result_dir, "result.png"))


if __name__ == "__main__":
    main()
