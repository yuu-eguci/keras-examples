"""潜在空間が2次元の変分オートエンコーダ（VAE）で MNIST を学習する。

参考:
- https://blog.keras.io/building-autoencoders-in-keras.html
- https://github.com/keras-team/keras/blob/2.3.1/examples/variational_autoencoder.py
"""
import argparse
import os

import keras
import numpy as np
from keras import ops
from matplotlib.figure import Figure

BATCH_SIZE = 100
ORIGINAL_DIM = 784
LATENT_DIM = 2
INTERMEDIATE_DIM = 256
EPSILON_STD = 0.01


class Sampling(keras.layers.Layer):
    """平均と対数分散から潜在変数をサンプリングし、KL ダイバージェンスを損失に加える。"""

    def __init__(self, epsilon_std, **kwargs):
        super().__init__(**kwargs)
        self.epsilon_std = epsilon_std
        self.seed_generator = keras.random.SeedGenerator()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * ops.sum(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
        self.add_loss(ops.mean(kl_loss))
        epsilon = keras.random.normal(
            ops.shape(z_mean), stddev=self.epsilon_std, seed=self.seed_generator)
        return z_mean + ops.exp(z_log_var / 2) * epsilon


def reconstruction_loss(x, x_decoded_mean):
    return ORIGINAL_DIM * keras.losses.binary_crossentropy(x, x_decoded_mean)


def build_models():
    """VAE 本体、入力を潜在空間に写すエンコーダ、潜在変数から画像を作るジェネレータを返す。"""
    x = keras.Input(shape=(ORIGINAL_DIM,))
    h = keras.layers.Dense(INTERMEDIATE_DIM, activation="relu")(x)
    z_mean = keras.layers.Dense(LATENT_DIM)(h)
    z_log_var = keras.layers.Dense(LATENT_DIM)(h)
    z = Sampling(EPSILON_STD)([z_mean, z_log_var])

    # ジェネレータでも使うため、デコーダの層は別に作っておく
    decoder_h = keras.layers.Dense(INTERMEDIATE_DIM, activation="relu")
    decoder_mean = keras.layers.Dense(ORIGINAL_DIM, activation="sigmoid")
    vae = keras.Model(x, decoder_mean(decoder_h(z)))

    encoder = keras.Model(x, z_mean)

    decoder_input = keras.Input(shape=(LATENT_DIM,))
    generator = keras.Model(decoder_input, decoder_mean(decoder_h(decoder_input)))
    return vae, encoder, generator


def load_data():
    (x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(len(x_train), ORIGINAL_DIM).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), ORIGINAL_DIM).astype("float32") / 255.0
    return x_train, (x_test, y_test)


def save_latent_space(encoder, x_test, y_test, path):
    """テスト画像を潜在空間に写し、ラベルで色分けした散布図を保存する。"""
    x_test_encoded = encoder.predict(x_test, batch_size=BATCH_SIZE)
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot()
    points = ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    fig.colorbar(points)
    fig.savefig(path)


def save_manifold(generator, path, n=15, digit_size=28):
    """潜在空間の n×n の格子点から生成した数字を並べて保存する。"""
    # 標準偏差の -15〜15 倍の範囲から格子点を取る
    grid = np.linspace(-15, 15, n) * EPSILON_STD
    z_samples = np.array([[xi, yi] for yi in grid for xi in grid])
    digits = generator.predict(z_samples, batch_size=BATCH_SIZE)
    figure = (digits.reshape(n, n, digit_size, digit_size)
              .transpose(0, 2, 1, 3)
              .reshape(n * digit_size, n * digit_size))
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.imshow(figure)
    fig.savefig(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--result-dir", default="result")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.result_dir, exist_ok=True)

    vae, encoder, generator = build_models()
    vae.compile(optimizer="rmsprop", loss=reconstruction_loss)

    x_train, (x_test, y_test) = load_data()
    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=args.epochs,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, x_test))

    save_latent_space(encoder, x_test, y_test, os.path.join(args.result_dir, "latent.png"))
    save_manifold(generator, os.path.join(args.result_dir, "manifold.png"))


if __name__ == "__main__":
    main()
