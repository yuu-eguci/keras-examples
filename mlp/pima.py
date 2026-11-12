"""Pima Indians Diabetes のデータから糖尿病の有無を多層パーセプトロンで分類する。

参考: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""
import argparse
import os

import keras
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data",
                         "pima-indians-diabetes.data")


def load_data():
    dataset = np.loadtxt(DATA_FILE, delimiter=",")
    return dataset[:, :8], dataset[:, 8]


def build_model():
    layers = keras.layers
    return keras.Sequential([
        keras.Input(shape=(8,)),
        layers.Dense(12, kernel_initializer="random_uniform", activation="relu"),
        layers.Dense(8, kernel_initializer="random_uniform", activation="relu"),
        layers.Dense(1, kernel_initializer="random_uniform", activation="sigmoid"),
    ])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--checkpoint-dir",
                        help="指定すると、データの3分の1で検証し、検証精度が最高を更新するたびに重みを保存する")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # 結果を再現できるように乱数の種を固定
    keras.utils.set_random_seed(7)

    x, y = load_data()
    model = build_model()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    fit_kwargs = {}
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(args.checkpoint_dir,
                                "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.weights.h5")
        fit_kwargs = {
            "validation_split": 0.33,
            "callbacks": [keras.callbacks.ModelCheckpoint(
                filepath, monitor="val_accuracy", mode="max", save_best_only=True,
                save_weights_only=True, verbose=1)],
        }
    model.fit(x, y, epochs=args.epochs, batch_size=10, **fit_kwargs)

    # 学習に使ったデータ全体での精度
    _, accuracy = model.evaluate(x, y, verbose=0)
    print("accuracy: %.2f%%" % (accuracy * 100))


if __name__ == "__main__":
    main()
