"""Boston Housing の住宅価格を回帰し、交差検証で平均二乗誤差（MSE）を求める。"""
import argparse

import keras
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
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


def build_pipeline(name, standardize, epochs):
    """（必要なら）標準化してから回帰する scikit-learn のパイプラインを作る。"""
    regressor = keras.wrappers.SKLearnRegressor(
        model=lambda X, y: build_model(name),
        fit_kwargs={"epochs": epochs, "batch_size": 5, "verbose": 0})
    steps = [("standardize", StandardScaler())] if standardize else []
    return Pipeline(steps + [("regression", regressor)])


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

    pipeline = build_pipeline(args.model, args.standardize, args.epochs)
    scores = cross_val_score(pipeline, x, y, cv=KFold(n_splits=args.folds),
                             scoring="neg_mean_squared_error")
    print("MSE: %.2f (%.2f)" % (-scores.mean(), scores.std()))


if __name__ == "__main__":
    main()
