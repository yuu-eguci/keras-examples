"""MNIST を全結合1層のオートエンコーダで圧縮・再構成する。

参考: https://blog.keras.io/building-autoencoders-in-keras.html
"""
import argparse

import keras
from matplotlib.figure import Figure

ENCODING_DIM = 32


def build_model():
    input_img = keras.Input(shape=(784,))
    encoded = keras.layers.Dense(ENCODING_DIM, activation="relu", name="encoded")(input_img)
    decoded = keras.layers.Dense(784, activation="sigmoid")(encoded)
    return keras.Model(input_img, decoded)


def load_data():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(len(x_train), 784).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), 784).astype("float32") / 255.0
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="reconstruction.png")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    autoencoder = build_model()
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    x_train, x_test = load_data()
    autoencoder.fit(x_train, x_train,
                    epochs=args.epochs,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # 中間層の出力を取り出すエンコーダ
    encoder = keras.Model(autoencoder.input, autoencoder.get_layer("encoded").output)
    encoded_imgs = encoder.predict(x_test)
    print("mean of hidden outputs: %f" % encoded_imgs.mean())

    decoded_imgs = autoencoder.predict(x_test)
    save_reconstructions(x_test, decoded_imgs, args.output)


if __name__ == "__main__":
    main()
