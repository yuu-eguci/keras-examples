"""DCGAN で手書き数字の画像を生成する。

参考: https://elix-tech.github.io/ja/2017/02/06/gan.html
"""
import argparse
import math
import os

import keras
import numpy as np
from PIL import Image

BATCH_SIZE = 32
NOISE_DIM = 100
# このバッチ数ごとに生成画像を保存する
SAVE_INTERVAL = 500


def generator_model():
    layers = keras.layers
    return keras.Sequential([
        keras.Input(shape=(NOISE_DIM,)),
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(7 * 7 * 128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Reshape((7, 7, 128)),
        layers.UpSampling2D(2),  # 14x14
        layers.Conv2D(64, 5, padding="same"),
        layers.Activation("tanh"),
        layers.UpSampling2D(2),  # 28x28
        layers.Conv2D(1, 5, padding="same"),  # 28x28x1 が出力
        layers.Activation("tanh"),
    ])


def discriminator_model():
    layers = keras.layers
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        # プーリングの代わりにストライド2で出力のサイズを半分にする
        layers.Conv2D(64, 5, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 5, strides=2),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.5),
        layers.Dense(1),
        layers.Activation("sigmoid"),
    ])


def build_models():
    """Generator、Discriminator、Generator のあとに Discriminator をつないだ DCGAN を返す。"""
    discriminator = discriminator_model()
    discriminator.compile(loss="binary_crossentropy",
                          optimizer=keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.1))

    generator = generator_model()
    # 入力 (100,) => generator => (28, 28, 1) => discriminator => (1,)
    dcgan = keras.Sequential([generator, discriminator])
    dcgan.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5))
    return generator, discriminator, dcgan


def make_noise(rng, n):
    return rng.uniform(-1, 1, (n, NOISE_DIM)).astype("float32")


def train_discriminator(discriminator, real_images, generated_images):
    """訓練画像を1、生成画像を0と判定するように Discriminator を更新する。"""
    # Keras 3 では trainable を更新のたびに設定し直す必要がある
    discriminator.trainable = True
    x = np.concatenate((real_images, generated_images))
    y = np.concatenate((np.ones((len(real_images), 1)), np.zeros((len(generated_images), 1))))
    return discriminator.train_on_batch(x, y)


def train_generator(dcgan, discriminator, noise):
    """生成画像を Discriminator が1（訓練画像）と判定するように Generator だけを更新する。"""
    discriminator.trainable = False
    return dcgan.train_on_batch(noise, np.ones((len(noise), 1)))


def combine_images(generated_images):
    """生成画像をほぼ正方形のタイル状に並べた1枚の画像にする。"""
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(total / cols)
    height, width = generated_images.shape[1:3]
    combined_image = np.zeros((height * rows, width * cols), dtype=generated_images.dtype)
    for index, image in enumerate(generated_images):
        i, j = divmod(index, cols)
        combined_image[height * i:height * (i + 1), width * j:width * (j + 1)] = image[:, :, 0]
    return combined_image


def load_data():
    (x_train, _), _ = keras.datasets.mnist.load_data()
    # 画素値を -1〜1 にし、(サンプル数, 28, 28, 1) の4次元テンソルにする
    x_train = (x_train.astype("float32") - 127.5) / 127.5
    return x_train.reshape(len(x_train), 28, 28, 1)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--result-dir", default="result")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.result_dir, exist_ok=True)
    rng = np.random.default_rng()

    x_train = load_data()
    generator, discriminator, dcgan = build_models()

    num_batches = len(x_train) // BATCH_SIZE
    print("Number of batches:", num_batches)
    for epoch in range(args.epochs):
        for index in range(num_batches):
            image_batch = x_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = generator.predict(make_noise(rng, BATCH_SIZE), verbose=0)

            if index % SAVE_INTERVAL == 0:
                image = combine_images(generated_images) * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    os.path.join(args.result_dir, "%04d_%04d.png" % (epoch, index)))

            d_loss = train_discriminator(discriminator, image_batch, generated_images)
            g_loss = train_generator(dcgan, discriminator, make_noise(rng, BATCH_SIZE))
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

    generator.save_weights(os.path.join(args.result_dir, "generator.weights.h5"))
    discriminator.save_weights(os.path.join(args.result_dir, "discriminator.weights.h5"))


if __name__ == "__main__":
    main()
