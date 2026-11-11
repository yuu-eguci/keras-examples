"""MNIST の数字を畳み込みニューラルネットワークで分類する。

参考: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
"""
import argparse
import os

import keras
from matplotlib.figure import Figure


def build_model():
    layers = keras.layers
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 5, activation="relu"),
        layers.Conv2D(32, 5, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # (サンプル数, 縦, 横, チャネル) の4次元テンソルにし、画素値を 0.0〜1.0 に変換
    x_train = x_train.reshape(len(x_train), 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), 28, 28, 1).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


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


def save_filters(model, path):
    """1層目の畳み込みフィルタ32枚をグレースケールで並べて保存する。"""
    # 重みの形状は (縦, 横, 入力チャネル, フィルタ数)
    weights = model.layers[0].get_weights()[0]
    fig = Figure()
    for i in range(weights.shape[3]):
        ax = fig.add_subplot(4, 8, i + 1)
        ax.imshow(weights[:, :, 0, i], cmap="gray")
        ax.axis("off")
    fig.savefig(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--result-dir", default="result")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.result_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    save_filters(model, os.path.join(args.result_dir, "weight_before.png"))

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=args.epochs,
                        validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping()])

    save_filters(model, os.path.join(args.result_dir, "weight_after.png"))
    save_history(history, args.result_dir)
    plot_history(history, args.result_dir)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", loss)
    print("Test acc:", acc)


if __name__ == "__main__":
    main()
