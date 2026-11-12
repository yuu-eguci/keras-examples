"""状態を持つ（stateful）LSTM で、アルファベットの1文字から次の文字を予測する。

参考: http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
"""
import argparse

import keras
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode(indices):
    """文字の番号の並びを、0〜1 に正規化した (サンプル数, 1, 1) の入力にする。"""
    return np.reshape(indices, (len(indices), 1, 1)) / float(len(ALPHABET))


def make_dataset():
    """各文字を入力、その次の文字の番号を正解とするデータを返す。"""
    indices = np.arange(len(ALPHABET))
    return encode(indices[:-1]), indices[1:]


def build_model():
    # バッチサイズ1で、バッチをまたいで LSTM の状態を引き継ぐ
    return keras.Sequential([
        keras.Input(batch_shape=(1, 1, 1)),
        keras.layers.LSTM(32, stateful=True, name="lstm"),
        keras.layers.Dense(len(ALPHABET), activation="softmax"),
    ])


def predict_next(model, index):
    return int(np.argmax(model.predict(encode([index]), verbose=0)))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(7)

    x, y = make_dataset()
    model = build_model()
    lstm = model.get_layer("lstm")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # アルファベットを1周するごとに状態をリセットする
    for _ in range(args.epochs):
        model.fit(x, y, epochs=1, batch_size=1, verbose=2, shuffle=False)
        lstm.reset_states()

    _, accuracy = model.evaluate(x, y, batch_size=1, verbose=0)
    lstm.reset_states()
    print("Model accuracy (train data): %.2f%%" % (accuracy * 100))

    # A から順に、予測した文字を次の入力にして1周分予測する
    index = 0
    for _ in range(len(ALPHABET) - 1):
        predicted = predict_next(model, index)
        print(ALPHABET[index], "->", ALPHABET[predicted])
        index = predicted
    lstm.reset_states()

    # 途中の文字 K から始めて5文字分予測する
    print("New start: K")
    index = ALPHABET.index("K")
    for _ in range(5):
        predicted = predict_next(model, index)
        print(ALPHABET[index], "->", ALPHABET[predicted])
        index = predicted
    lstm.reset_states()


if __name__ == "__main__":
    main()
