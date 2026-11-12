"""短い文から色が green かどうかを学習し、単語ごとの埋め込みベクトルを表示する。

参考: http://benjaminbolte.com/blog/2016/keras-language-modeling.html
"""
import argparse

import keras
import numpy as np

SENTENCES = [
    "sam is red",
    "hannah not red",
    "hannah is green",
    "bob is green",
    "bob not red",
    "sam not green",
    "sarah is red",
    "sarah not green",
]
# 各文が表す色が green なら1
IS_GREEN = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype="float32")
EMBED_DIMS = 3


def build_model(num_words):
    return keras.Sequential([
        keras.Input(shape=(3,)),
        keras.layers.Embedding(num_words, EMBED_DIMS),
        keras.layers.SimpleRNN(1, activation="sigmoid"),
    ])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    words = sorted({word for sentence in SENTENCES for word in sentence.split()})
    word_index = {word: i for i, word in enumerate(words)}
    x = np.array([[word_index[word] for word in sentence.split()] for sentence in SENTENCES])

    model = build_model(len(words))
    model.summary()
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    model.fit(x, IS_GREEN, epochs=args.epochs, verbose=0)

    embeddings = model.layers[0].get_weights()[0]
    for word, vector in zip(words, embeddings):
        print("%s: [%s]" % (word, " ".join("%.3f" % value for value in vector)))


if __name__ == "__main__":
    main()
