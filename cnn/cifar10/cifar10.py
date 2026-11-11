"""CIFAR-10 の画像を畳み込みニューラルネットワークで分類する。"""
import argparse
import os

import keras
import numpy as np
from matplotlib.figure import Figure


def build_model(augment=False):
    layers = keras.layers
    model = keras.Sequential([keras.Input(shape=(32, 32, 3))])
    if augment:
        # 学習時だけ画像を縦横ランダムに最大10%平行移動し、空いた部分は端の画素で埋める
        # （推論時は何もしない）
        model.add(layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"))
    for layer in [
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]:
        model.add(layer)
    return model


def plot_cifar10(x, y, path):
    """クラスごとに訓練画像を最大10枚ずつランダムに選び、1クラス1行で並べて保存する。"""
    rng = np.random.default_rng()
    fig = Figure(figsize=(10, 10))
    for target_class in range(10):
        indices = np.flatnonzero(y.ravel() == target_class)
        for i, idx in enumerate(rng.permutation(indices)[:10]):
            ax = fig.add_subplot(10, 10, target_class * 10 + i + 1)
            ax.imshow(x[idx])
            ax.axis("off")
    fig.savefig(path)


def save_history(history, result_dir):
    h = history.history
    with open(os.path.join(result_dir, "history.txt"), "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(len(h["loss"])):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (
                i, h["loss"][i], h["accuracy"][i], h["val_loss"][i], h["val_accuracy"][i]))


def plot_history(history, result_dir):
    """精度と損失の学習曲線を acc.png / loss.png に保存する。"""
    for metric, filename in (("accuracy", "acc.png"), ("loss", "loss.png")):
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot(history.history[metric], marker=".", label="train")
        ax.plot(history.history["val_" + metric], marker=".", label="validation")
        ax.set_title("model " + metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid()
        ax.legend()
        fig.savefig(os.path.join(result_dir, filename))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--augment", action="store_true",
                        help="学習時に画像をランダムに平行移動してデータを拡張する")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.result_dir, exist_ok=True)
    print("epochs:", args.epochs)
    print("augment:", args.augment)
    print("result_dir:", args.result_dir)

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    plot_cifar10(x_train, y_train, os.path.join(args.result_dir, "plot.png"))

    # 画素値を 0.0〜1.0 に変換
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_model(augment=args.augment)
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    # 本来は好ましくないが、テストデータを検証データとして使う
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=args.epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)

    model.save(os.path.join(args.result_dir, "model.keras"))
    save_history(history, args.result_dir)
    plot_history(history, args.result_dir)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", loss)
    print("Test acc:", acc)


if __name__ == "__main__":
    main()
