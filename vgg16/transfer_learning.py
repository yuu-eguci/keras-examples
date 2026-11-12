"""クラスごとのディレクトリに置いた画像から、小さな CNN や VGG16 を使った画像分類モデルを学習する。

参考: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import argparse
import os

import keras
import numpy as np

BATCH_SIZE = 32
METHODS = ("smallcnn", "extractor", "finetuning", "scratch")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


@keras.saving.register_keras_serializable(package="keras_examples")
class VGG16Preprocess(keras.layers.Layer):
    """ImageNet で VGG16 を学習したときと同じ前処理（RGB から BGR への並べ替えと平均画素の減算）をかける。"""

    def call(self, inputs):
        return keras.applications.vgg16.preprocess_input(inputs)


def load_images(directory, image_size):
    """サブディレクトリ名（名前順）をクラスとして画像を読み込み、(画像, ラベル, クラス名) を返す。

    "." で始まる隠しディレクトリはクラスとみなさない。
    """
    classes = sorted(name for name in os.listdir(directory)
                     if not name.startswith(".") and os.path.isdir(os.path.join(directory, name)))
    images, labels = [], []
    for label, name in enumerate(classes):
        class_dir = os.path.join(directory, name)
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image = keras.utils.load_img(os.path.join(class_dir, filename),
                                             target_size=(image_size, image_size))
                images.append(keras.utils.img_to_array(image))
                labels.append(label)
    return np.array(images, dtype="float32"), np.array(labels), classes


def build_top(input_shape, num_classes):
    """VGG16 の畳み込み部分の上に載せる全結合層。"""
    layers = keras.layers
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="top")


def build_model(method, num_classes, image_size, preprocess, top_weights=None):
    """前処理から分類までの（データ拡張を含まない）モデルを作る。"""
    layers = keras.layers
    inputs = keras.Input(shape=(image_size, image_size, 3))
    if preprocess == "vgg16":
        x = VGG16Preprocess(name="preprocess")(inputs)
    else:
        x = layers.Rescaling(1 / 255.0, name="preprocess")(inputs)

    if method == "smallcnn":
        for layer in [
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]:
            x = layer(x)
        return keras.Model(inputs, x)

    vgg16 = keras.applications.VGG16(
        include_top=False,
        weights=None if method == "scratch" else "imagenet",
        input_shape=(image_size, image_size, 3))
    if method == "extractor":
        vgg16.trainable = False
    elif method == "finetuning":
        # 最後のブロック（block5）だけを学習し、それより前の層は固定する
        for layer in vgg16.layers:
            layer.trainable = layer.name.startswith("block5")
    top = build_top(vgg16.output_shape[1:], num_classes)
    if top_weights:
        top.load_weights(top_weights)
    return keras.Model(inputs, top(vgg16(x)))


def with_augmentation(model):
    """学習時だけランダムな左右反転・拡大縮小・せん断をかけてから model に入力するモデルを作る。

    重みは model と共有するので、学習後は model をそのまま保存・推論に使える。
    """
    layers = keras.layers
    inputs = keras.Input(shape=model.input_shape[1:])
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomZoom(0.2, fill_mode="nearest")(x)
    x = layers.RandomShear(x_factor=0.2, fill_mode="nearest")(x)
    return keras.Model(inputs, model(x))


def train_extractor(model, x_train, y_train, x_val, y_val, epochs):
    """VGG16 の出力（特徴量）を一度だけ計算し、それを入力に全結合層だけを学習する。"""
    features = keras.Sequential([model.get_layer("preprocess"), model.get_layer("vgg16")])
    features_train = features.predict(x_train, batch_size=BATCH_SIZE)
    features_val = features.predict(x_val, batch_size=BATCH_SIZE)

    top = model.get_layer("top")
    top.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                metrics=["accuracy"])
    return top.fit(features_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                   validation_data=(features_val, y_val))


def save_history(history, path):
    h = history.history
    with open(path, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(len(h["loss"])):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (
                i, h["loss"][i], h["accuracy"][i], h["val_loss"][i], h["val_accuracy"][i]))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--preprocess", choices=("rescale", "vgg16"), default="rescale")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--top-weights",
                        help="finetuning のみ。extractor で学習した全結合層の重み")
    parser.add_argument("--name", help="出力ファイル名に使う名前")
    parser.add_argument("--result-dir", default="results")
    args = parser.parse_args(argv)
    if args.top_weights and args.method != "finetuning":
        parser.error("--top-weights は --method finetuning のときだけ指定できます")
    return args


def main(argv=None):
    args = parse_args(argv)
    name = args.name or args.method + ("_with_preprocess" if args.preprocess == "vgg16" else "")
    os.makedirs(args.result_dir, exist_ok=True)

    x_train, y_train, classes = load_images(args.train_dir, args.image_size)
    x_val, y_val, val_classes = load_images(args.validation_dir, args.image_size)
    if val_classes != classes:
        raise ValueError("訓練用と検証用のディレクトリでクラスが違います: %s / %s"
                         % (classes, val_classes))
    print("classes:", classes)
    print(len(x_train), "train samples,", len(x_val), "validation samples")

    model = build_model(args.method, len(classes), args.image_size, args.preprocess,
                        top_weights=args.top_weights)
    model.summary()

    if args.method == "extractor":
        history = train_extractor(model, x_train, y_train, x_val, y_val, args.epochs)
        model.get_layer("top").save_weights(
            os.path.join(args.result_dir, "top_%s.weights.h5" % name))
    else:
        if args.method == "finetuning":
            optimizer = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
        else:
            optimizer = "adam"
        trainer = with_augmentation(model)
        trainer.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                        metrics=["accuracy"])
        history = trainer.fit(x_train, y_train, epochs=args.epochs, batch_size=BATCH_SIZE,
                            shuffle=True, validation_data=(x_val, y_val))

    model.save(os.path.join(args.result_dir, "model_%s.keras" % name))
    save_history(history, os.path.join(args.result_dir, "history_%s.txt" % name))
    with open(os.path.join(args.result_dir, "classes.txt"), "w") as fp:
        fp.write("\n".join(classes) + "\n")


if __name__ == "__main__":
    main()
