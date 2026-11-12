"""ImageNet で学習済みの VGG16 の中間層の出力が大きくなるように画像を書き換える（DeepDream）。

参考: https://github.com/keras-team/keras/blob/2.0.0/examples/deep_dream.py
"""
import argparse

import jax
import jax.numpy as jnp
import keras
import numpy as np
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b

# 面白い結果が得られる設定
SETTINGS = {
    "bad_trip": {"features": {"block4_conv1": 0.05,
                              "block4_conv2": 0.01,
                              "block4_conv3": 0.01},
                 "continuity": 0.1,
                 "dream_l2": 0.8,
                 "jitter": 5},
    "dreamy": {"features": {"block5_conv1": 0.05,
                            "block5_conv2": 0.02},
               "continuity": 0.1,
               "dream_l2": 0.02,
               "jitter": 0},
}
# VGG16 の重みはもともと Caffe のもので、BGR の画像から平均画素を引いて学習されている
MEAN_PIXEL_BGR = np.array([103.939, 116.779, 123.68])


def preprocess_image(path, image_size):
    image = keras.utils.load_img(path, target_size=(image_size, image_size))
    x = keras.utils.img_to_array(image)[np.newaxis]
    return keras.applications.vgg16.preprocess_input(x)


def deprocess_image(x):
    """preprocess_image の逆変換で uint8 の RGB 画像に戻す。"""
    x = x.reshape(x.shape[-3:]) + MEAN_PIXEL_BGR
    return np.clip(np.round(x[:, :, ::-1]), 0, 255).astype("uint8")


def build_model(image_size):
    return keras.applications.VGG16(include_top=False, weights="imagenet",
                                    input_shape=(image_size, image_size, 3))


def make_loss_and_grads(model, settings, image_size):
    """平らにした画像から (損失, 勾配) を返す、L-BFGS に渡す関数を作る。"""
    names = list(settings["features"])
    feature = keras.Model(model.input, [model.get_layer(name).output for name in names])
    trainable = [v.value for v in feature.trainable_variables]
    non_trainable = [v.value for v in feature.non_trainable_variables]
    shape = (1, image_size, image_size, 3)
    size = np.prod(shape[1:])

    # 重みは jit した関数の引数として渡す（クロージャで参照すると定数として埋め込まれる）
    def loss_function(x, trainable, non_trainable):
        outputs, _ = feature.stateless_call(trainable, non_trainable, x)
        loss = 0.0
        # 層の全フィルタの出力（端の2画素を除く）の二乗和を大きくしたいので、損失では引く
        for name, output in zip(names, outputs):
            _, height, width, _ = output.shape
            loss -= (settings["features"][name]
                     * jnp.sum(jnp.square(output[:, 2:height - 2, 2:width - 2, :]))
                     / np.prod(output.shape[1:]))
        # 隣り合う画素の差（continuity）と画素値の大きさ（L2）を罰して、画像が崩れすぎないようにする
        a = jnp.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
        b = jnp.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
        loss += settings["continuity"] * jnp.sum((a + b) ** 1.25) / size
        loss += settings["dream_l2"] * jnp.sum(jnp.square(x)) / size
        return loss

    value_and_grad = jax.jit(jax.value_and_grad(loss_function))

    def loss_and_grads(x):
        loss, grads = value_and_grad(jnp.asarray(x.reshape(shape), dtype="float32"),
                                     trainable, non_trainable)
        return float(loss), np.asarray(grads, dtype="float64").ravel()

    return loss_and_grads


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_image_path", help="書き換える画像")
    parser.add_argument("result_prefix", help="保存する画像のファイル名の接頭辞")
    parser.add_argument("--setting", choices=SETTINGS, default="dreamy")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=600)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if keras.backend.backend() != "jax":
        raise SystemExit("このスクリプトは KERAS_BACKEND=jax で実行してください")
    settings = SETTINGS[args.setting]
    print(settings)

    model = build_model(args.image_size)
    loss_and_grads = make_loss_and_grads(model, settings, args.image_size)

    rng = np.random.default_rng()
    x = preprocess_image(args.base_image_path, args.image_size).astype("float64")
    for i in range(args.iterations):
        print("Start of iteration", i)
        # 画像を少しずらしてから最適化し、あとで戻す
        jitter = settings["jitter"] * 2 * (rng.random(x.shape) - 0.5)
        x += jitter
        x, min_val, _ = fmin_l_bfgs_b(loss_and_grads, x.ravel(), maxfun=7)
        print("Current loss value:", min_val)
        x = x.reshape((1, args.image_size, args.image_size, 3)) - jitter

        filename = "%s_at_iteration_%d.png" % (args.result_prefix, i)
        Image.fromarray(deprocess_image(x)).save(filename)
        print("Image saved as", filename)


if __name__ == "__main__":
    main()
