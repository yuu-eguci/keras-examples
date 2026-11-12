"""Iris のがく片と花弁の大きさから、アヤメの品種（3クラス）を多層パーセプトロンで分類する。"""
import argparse

import keras
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split


def build_model():
    return keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    iris = datasets.load_iris()
    # 各特徴量を平均0・分散1に標準化し、8割を訓練データ、2割をテストデータにする
    x = preprocessing.scale(iris.data)
    train_x, test_x, train_y, test_y = train_test_split(x, iris.target, train_size=0.8)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=args.epochs, batch_size=1)

    _, accuracy = model.evaluate(test_x, test_y, verbose=0)
    print("Accuracy = {:.2f}".format(accuracy))


if __name__ == "__main__":
    main()
