"""アルファベット全体を1つの系列として LSTM に入力し、各位置で次の文字を予測する。

LSTM の状態が系列の先頭からの文脈を覚えるので、同じ文字でも位置に応じて次の文字を予測できる。
Keras 3 の JAX バックエンドでは stateful な RNN の状態が fit / predict でバッチをまたいで
引き継がれないため、1文字ずつのバッチに分けずに系列全体を入力する。

参考: http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
"""
import argparse

import keras
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode(indices):
    """文字の番号の並びを、0〜1 に正規化した (1, 系列の長さ, 1) の入力にする。"""
    return np.reshape(indices, (1, len(indices), 1)) / float(len(ALPHABET))


def make_dataset():
    """A〜Y の系列を入力、各位置の次の文字の番号を正解とするデータを返す。"""
    indices = np.arange(len(ALPHABET))
    return encode(indices[:-1]), indices[np.newaxis, 1:]


def build_model():
    return keras.Sequential([
        keras.Input(shape=(None, 1)),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.Dense(len(ALPHABET), activation="softmax"),
    ])


def predict_sequence(model, start, length):
    """start の文字から始め、予測した文字を系列の末尾に足しながら length 文字分予測する。"""
    sequence = [start]
    for _ in range(length):
        probabilities = model.predict(encode(sequence), verbose=0)[0, -1]
        sequence.append(int(np.argmax(probabilities)))
    return sequence


def print_pairs(sequence):
    for current, following in zip(sequence, sequence[1:]):
        print(ALPHABET[current], "->", ALPHABET[following])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(7)

    x, y = make_dataset()
    model = build_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x, y, epochs=args.epochs, verbose=0)

    _, accuracy = model.evaluate(x, y, verbose=0)
    print("Model accuracy (train data): %.2f%%" % (accuracy * 100))

    # A から順に、予測した文字を次の入力に足していく
    print_pairs(predict_sequence(model, 0, len(ALPHABET) - 1))

    # 途中の K から始めると、それまでの文脈がないので正しく予測できるとは限らない
    print("New start: K")
    print_pairs(predict_sequence(model, ALPHABET.index("K"), 5))


if __name__ == "__main__":
    main()
