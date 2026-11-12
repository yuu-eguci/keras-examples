# Keras Examples

Keras のサンプルプログラムをまとめたリポジトリです。

## 動作環境

- Python 3.12 以上
- Keras 3（バックエンドは CPU 版 JAX）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export KERAS_BACKEND=jax
```

## Keras 3 で動作確認済みの例

データセット（MNIST / CIFAR-10）は初回実行時に `~/.keras/datasets/` へダウンロードされます。

### 画像分類

| スクリプト | 内容 | 既定のエポック数 |
| --- | --- | --- |
| `mlp/mnist.py` | MNIST を多層パーセプトロンで分類 | 100（検証損失が改善しなくなった時点で終了） |
| `cnn/mnist/mnist.py` | MNIST を畳み込みニューラルネットワークで分類 | 100（検証損失が改善しなくなった時点で終了） |
| `mlp/cifar10.py` | CIFAR-10 を多層パーセプトロンで分類 | 200 |
| `cnn/cifar10/cifar10.py` | CIFAR-10 を畳み込みニューラルネットワークで分類 | 50 |

```bash
.venv/bin/python cnn/cifar10/cifar10.py --epochs 5 --augment --result-dir result_with_datagen
```

| オプション | 意味 |
| --- | --- |
| `--epochs` | 学習エポック数 |
| `--result-dir` | 結果の保存先ディレクトリ（既定は実行したディレクトリの `result`。なければ作成） |
| `--augment` | `cnn/cifar10/cifar10.py` のみ。学習時に画像を縦横ランダムに最大10%平行移動し（空いた部分は端の画素で埋める）、データを拡張する |

どのスクリプトも保存先ディレクトリに次のファイルを書き出します。

| ファイル | 内容 |
| --- | --- |
| `history.txt` | エポックごとの `loss` / `acc` / `val_loss` / `val_acc`（タブ区切り） |
| `acc.png` / `loss.png` | 精度と損失の学習曲線 |

検証データには、MNIST の例では訓練データの10%、CIFAR-10 の例ではテストデータを使います。学習後にテストデータでの損失と精度を表示します。

スクリプトによっては次のファイルも書き出します。

| スクリプト | ファイル | 内容 |
| --- | --- | --- |
| `cnn/mnist/mnist.py` | `weight_before.png` / `weight_after.png` | 学習前後の1層目の畳み込みフィルタ |
| `cnn/cifar10/cifar10.py` | `plot.png` | クラスごとに訓練画像を最大10枚ずつ並べた画像 |
| `cnn/cifar10/cifar10.py` | `model.keras` | 学習済みモデル |

`cnn/cifar10/visualize_filters.py --result-dir <ディレクトリ>` は、そのディレクトリの `model.keras` を読み込み、1層目の畳み込みフィルタを `filters.png` として同じディレクトリに保存します。`cnn/cifar10/plot_results.py` は、実行したディレクトリにある `result_without_datagen/history.txt` と `result_with_datagen/history.txt` を読み込み、検証損失と検証精度を比較するグラフを `val_loss.png` / `val_acc.png` として同じディレクトリに保存します。

### オートエンコーダ

| スクリプト | 内容 | 既定のエポック数 |
| --- | --- | --- |
| `autoencoder/mnist_autoencoder.py` | 隠れ層1層の全結合オートエンコーダ | 100 |
| `autoencoder/mnist_autoencoder_l1.py` | 中間層の出力に L1 正則化をかけたオートエンコーダ | 100 |
| `autoencoder/mnist_deep_autoencoder.py` | エンコーダ・デコーダが各3層の全結合オートエンコーダ | 100 |
| `autoencoder/mnist_conv_autoencoder.py` | 畳み込みオートエンコーダ | 50 |

どのスクリプトも MNIST を使い、テスト画像10枚の元画像（上段）と再構成画像（下段）を PNG に保存します。

```bash
.venv/bin/python autoencoder/mnist_autoencoder.py --epochs 5 --output reconstruction.png
```

| オプション | 意味 |
| --- | --- |
| `--epochs` | 学習エポック数 |
| `--output` | 再構成画像の保存先（既定は `reconstruction.png`） |

### 生成モデル

| スクリプト | 内容 | 既定のエポック数 |
| --- | --- | --- |
| `vae/mnist_vae.py` | 潜在空間が2次元の変分オートエンコーダ（VAE） | 50 |
| `gan/dcgan_mnist.py` | 手書き数字を生成する DCGAN | 20 |

```bash
.venv/bin/python gan/dcgan_mnist.py --epochs 1 --result-dir result_gan
```

どちらも MNIST を使い、`--epochs`（学習エポック数）と `--result-dir`（結果の保存先ディレクトリ。既定は実行したディレクトリの `result`。なければ作成）を指定できます。

| スクリプト | ファイル | 内容 |
| --- | --- | --- |
| `vae/mnist_vae.py` | `latent.png` | テスト画像を潜在空間に写した散布図（色は数字のラベル） |
| `vae/mnist_vae.py` | `manifold.png` | 潜在空間の格子点15×15から生成した数字の一覧 |
| `gan/dcgan_mnist.py` | `<エポック>_<バッチ>.png` | 各エポックの0, 500, 1000, …バッチ目で Generator が生成した画像の一覧（例: `0000_0500.png`） |
| `gan/dcgan_mnist.py` | `generator.weights.h5` / `discriminator.weights.h5` | 学習後の Generator と Discriminator の重み |

DCGAN は1バッチごとに Discriminator と Generator を交互に更新します。Discriminator を学習するときは Discriminator の重みだけを、Generator を学習するときは Generator の重みだけを更新します。

### 画像の前処理

| スクリプト | 内容 |
| --- | --- |
| `datagen/test_datagen3.py` | 1枚の画像に Keras のデータ拡張レイヤを9回ずつかけた結果を、拡張の種類ごとに3×3で並べて保存する |
| `zca_whitening/zca_whitening.py` | CIFAR-10 の訓練画像全体で ZCA 白色化を計算し、先頭16枚の白色化前後の画像を保存する |

```bash
.venv/bin/python datagen/test_datagen3.py --result-dir result_datagen
```

どちらも `--result-dir`（結果の保存先ディレクトリ。既定は実行したディレクトリの `result`。なければ作成）を指定できます。`datagen/test_datagen3.py` は `--image` で入力画像を指定でき、既定は `vgg16/test_vgg16/elephant.jpg` です。

`datagen/test_datagen3.py` は次の画像を保存します。平行移動・回転・拡大縮小・せん断で画像の外側になった部分は、端の画素で埋めます。

| ファイル | データ拡張 |
| --- | --- |
| `result_rotation.jpg` | ±90度の範囲で回転 |
| `result_width_shift.jpg` | 横幅の±20%の範囲で左右に平行移動 |
| `result_height_shift.jpg` | 高さの±20%の範囲で上下に平行移動 |
| `result_shear.jpg` | 横方向に最大約45度のせん断 |
| `result_zoom.jpg` | ±50%の範囲で拡大縮小 |
| `result_horizontal_flip.jpg` | 50%の確率で左右反転 |
| `result_vertical_flip.jpg` | 50%の確率で上下反転 |

`datagen/result/` には、既定の画像で実行した結果の例があります。

`zca_whitening/zca_whitening.py` は、白色化前の画像を `zca_whitening_before.png`、白色化後の画像を `zca_whitening_after.png` に保存します。どちらの画像も、表示用に画像ごとに最小値〜最大値を 0〜1 に正規化しています。50,000枚×3,072次元の行列を扱うため、3 GB 程度のメモリを使います。

### そのほか

上記以外のスクリプトは Keras 3 での動作を確認していません（多くは Keras 1 の API で書かれています）。

`blog/` は各例を解説したブログ記事の原稿で、Keras 1 の API とそのころのスクリプトに基づいています。

## テスト

```bash
.venv/bin/python -m pytest
```

テストは合成した極小データを使い、データセットのダウンロードなしに CPU だけで実行できます。GitHub Actions でも pull request と master への push ごとに同じテストを実行します。
