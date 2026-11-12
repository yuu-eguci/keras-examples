"""ImageNet で学習済みの VGG16 について、指定した層のフィルタ（またはクラス）の出力を最大化する画像を勾配法で求める。"""
import argparse
import functools
import json
import os

import jax
import jax.numpy as jnp
import keras
import numpy as np
from matplotlib.figure import Figure

IMAGE_SIZE = 224
CLASS_INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                "imagenet", "imagenet_class_index.json")
# Lp ノルムと Total Variation による正則化の設定
LP_P, LP_WEIGHT = 6.0, 10.0
TV_BETA, TV_WEIGHT = 2.0, 10.0
EPSILON = 1e-7


def build_model():
    # 出力層をソフトマックスをかける前の値にして、クラスごとの出力を最大化できるようにする
    return keras.applications.VGG16(weights="imagenet", classifier_activation=None,
                                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))


def class_names():
    with open(CLASS_INDEX_FILE) as fp:
        index = json.load(fp)
    return [index[str(i)][1] for i in range(len(index))]


@functools.cache
def objective_and_grads(model, layer_name):
    """(画像, フィルタ番号) から目的関数の値と画像に対する勾配を返す関数を作る。"""
    feature = keras.Model(model.input, model.get_layer(layer_name).output)
    trainable = [v.value for v in feature.trainable_variables]
    non_trainable = [v.value for v in feature.non_trainable_variables]

    def objective(x, filter_index):
        output, _ = feature.stateless_call(trainable, non_trainable, x)
        # 指定したフィルタ（predictions 層ならクラス）の出力の平均を大きくしたい
        value = jnp.mean(jnp.take(output, filter_index, axis=-1))
        size = np.prod(x.shape[1:])
        # Lp ノルムが大きい画像と、隣り合う画素の差（Total Variation）が大きい画像にペナルティを課す
        lp = jnp.sum(jnp.abs(x) ** LP_P) ** (1.0 / LP_P)
        a = jnp.square(x[:, 1:, :-1, :] - x[:, :-1, :-1, :])
        b = jnp.square(x[:, :-1, 1:, :] - x[:, :-1, :-1, :])
        tv = jnp.sum((a + b) ** (TV_BETA / 2.0))
        return value - LP_WEIGHT * lp / size - TV_WEIGHT * tv / size

    return jax.jit(jax.value_and_grad(objective))


def deprocess_image(x):
    """勾配法で求めた画像を、平均0.5・標準偏差0.1に正規化して表示用の uint8 画像にする。"""
    x = (x - x.mean()) / (x.std() + 1e-5) * 0.1 + 0.5
    return (np.clip(x, 0, 1) * 255).astype("uint8")


def visualize_filter(model, layer_name, filter_index, iterations, rng):
    """フィルタの出力を最大化する画像と、反復ごとの目的関数の値を返す。"""
    step_function = objective_and_grads(model, layer_name)
    # ノイズを含んだ灰色の画像から始める
    x = (rng.random((1, IMAGE_SIZE, IMAGE_SIZE, 3)) - 0.5) * 20 + 128
    cache = np.zeros_like(x)
    objectives = []
    for _ in range(iterations):
        value, grads = step_function(jnp.asarray(x, dtype="float32"), filter_index)
        grads = np.asarray(grads, dtype="float64")
        # 勾配の大きさをそろえてから RMSprop で画像を更新する（目的関数を大きくする向き）
        grads /= np.sqrt(np.mean(np.square(grads))) + EPSILON
        cache = 0.95 * cache + 0.05 * grads ** 2
        x += grads / np.sqrt(cache + EPSILON)
        objectives.append(float(value))
    return deprocess_image(x[0]), objectives


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", required=True, help="可視化する層の名前")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", default="result.png")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if keras.backend.backend() != "jax":
        raise SystemExit("このスクリプトは KERAS_BACKEND=jax で実行してください")

    model = build_model()
    layer_names = [layer.name for layer in model.layers]
    if args.layer not in layer_names:
        raise SystemExit("層 %s はありません。選べる層: %s" % (args.layer, ", ".join(layer_names[1:])))
    num_filters = model.get_layer(args.layer).output.shape[-1]
    names = class_names() if args.layer == "predictions" else None

    rng = np.random.default_rng()
    nrows, ncols = 4, 4
    fig = Figure(figsize=(8, 8))
    for i, filter_index in enumerate(rng.integers(0, num_filters, nrows * ncols)):
        image, objectives = visualize_filter(model, args.layer, int(filter_index), args.iterations, rng)
        print("filter %d: %f" % (filter_index, objectives[-1]))
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(image)
        ax.axis("off")
        label = names[filter_index] if names else "filter: %d" % filter_index
        ax.text(5, 20, label)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
