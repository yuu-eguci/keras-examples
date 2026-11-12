"""Boston Housing の住宅価格を回帰し、交差検証で平均二乗誤差（MSE）を求める。"""
import argparse

import keras
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# モデルの名前と隠れ層のユニット数
HIDDEN_UNITS = {
    "baseline": [13],
    "deeper": [13, 6],
    "wider": [20],
}


def build_model(name):
    model = keras.Sequential([keras.Input(shape=(13,))])
    for units in HIDDEN_UNITS[name]:
        model.add(keras.layers.Dense(units, kernel_initializer="random_normal", activation="relu"))
    model.add(keras.layers.Dense(1, kernel_initializer="random_normal"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def cross_validate(x, y, name, standardize, epochs, folds):
    """分割ごとに新しいモデルを学習し、検証側の MSE を並べた配列を返す。

    標準化するときは、その分割の訓練側だけから平均と標準偏差を求める。
    """
    mses = []
    for train, test in KFold(n_splits=folds).split(x):
        x_train, x_test = x[train], x[test]
        if standardize:
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        model = build_model(name)
        model.fit(x_train, y[train], epochs=epochs, batch_size=5, verbose=0)
        mses.append(model.evaluate(x_test, y[test], verbose=0))
    return np.array(mses)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=HIDDEN_UNITS, default="baseline")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--folds", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 全506件を交差検証に使う
    (x, y), _ = keras.datasets.boston_housing.load_data(test_split=0)

    mses = cross_validate(x, y, args.model, args.standardize, args.epochs, args.folds)
    print("MSE: %.2f (%.2f)" % (mses.mean(), mses.std()))


if __name__ == "__main__":
    main()
