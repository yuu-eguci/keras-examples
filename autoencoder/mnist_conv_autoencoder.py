"""畳み込みオートエンコーダで MNIST を圧縮・再構成する。

参考: https://blog.keras.io/building-autoencoders-in-keras.html
"""
import argparse

import keras
from matplotlib.figure import Figure


def build_model():
    layers = keras.layers
    input_img = keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same")(x)

    # ここで表現は (4, 4, 8) になる

    x = layers.Conv2D(8, 3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    # padding なしで 16x16 を 14x14 に縮め、次のアップサンプリングで 28x28 に戻す
    x = layers.Conv2D(16, 3, activation="relu")(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(input_img, decoded)


def load_data():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(len(x_train), 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), 28, 28, 1).astype("float32") / 255.0
    return x_train, x_test


def save_reconstructions(originals, reconstructions, path, n=10):
    """元画像を上段、再構成画像を下段に並べて保存する。"""
    n = min(n, len(originals))
    fig = Figure(figsize=(2 * n, 4))
    for i in range(n):
        for row, images in enumerate((originals, reconstructions)):
            ax = fig.add_subplot(2, n, row * n + i + 1)
            ax.imshow(images[i].reshape(28, 28), cmap="gray")
            ax.axis("off")
    fig.savefig(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", default="reconstruction.png")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    autoencoder = build_model()
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    x_train, x_test = load_data()
    autoencoder.fit(x_train, x_train,
                    epochs=args.epochs,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    decoded_imgs = autoencoder.predict(x_test)
    save_reconstructions(x_test, decoded_imgs, args.output)


if __name__ == "__main__":
    main()
