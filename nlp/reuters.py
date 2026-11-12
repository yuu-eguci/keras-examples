"""Reuters のニュースを、よく出る単語の有無を表すベクトルにして、多層パーセプトロンでトピックに分類する。

参考: https://github.com/keras-team/keras/blob/2.3.1/examples/reuters_mlp.py
"""
import argparse

import keras
import numpy as np

MAX_WORDS = 1000


def to_binary_matrix(sequences, num_words):
    """単語番号の列を、各単語が出てくれば1、出てこなければ0の行列にする。"""
    matrix = np.zeros((len(sequences), num_words), dtype="float32")
    for i, sequence in enumerate(sequences):
        matrix[i, [word for word in sequence if word < num_words]] = 1.0
    return matrix


def build_model(num_classes):
    return keras.Sequential([
        keras.Input(shape=(MAX_WORDS,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(1337)

    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(
        num_words=MAX_WORDS, test_split=0.2)
    print(len(x_train), "train sequences")
    print(len(x_test), "test sequences")
    num_classes = int(np.max(y_train)) + 1
    print(num_classes, "classes")

    x_train = to_binary_matrix(x_train, MAX_WORDS)
    x_test = to_binary_matrix(x_test, MAX_WORDS)

    model = build_model(num_classes)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=32,
              validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping()])

    _, accuracy = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
    print("Test acc:", accuracy)


if __name__ == "__main__":
    main()
