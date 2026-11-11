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
| `--augment` | `cnn/cifar10/cifar10.py` のみ。学習時に画像を縦横ランダムに最大10%平行移動してデータを拡張する |

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

### そのほか

上記以外のスクリプトは Keras 3 での動作を確認していません（多くは Keras 1 の API で書かれています）。

## テスト

```bash
.venv/bin/python -m pytest
```

テストは合成した極小データを使い、データセットのダウンロードなしに CPU だけで実行できます。GitHub Actions でも pull request と master への push ごとに同じテストを実行します。
