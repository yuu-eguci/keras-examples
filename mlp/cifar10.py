"""CIFAR-10 の画像を多層パーセプトロンで分類する。"""
import argparse
import os

import keras
from matplotlib.figure import Figure


def build_model():
    layers = keras.layers
    return keras.Sequential([
        keras.Input(shape=(32 * 32 * 3,)),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # 画像を1次元配列にし、画素値を 0.0〜1.0 に変換
    x_train = x_train.reshape(len(x_train), -1).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype("float32") / 255.0
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


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
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

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=args.epochs,
                        validation_data=(x_test, y_test))

    save_history(history, args.result_dir)
    plot_history(history, args.result_dir)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", loss)
    print("Test acc:", acc)


if __name__ == "__main__":
    main()
