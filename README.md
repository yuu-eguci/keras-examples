# Keras Examples

Kerasのサンプルプログラムをまとめたリポジトリです。

## 動作環境

- Python 3.12 以上
- Keras 3（バックエンドは CPU 版 JAX）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export KERAS_BACKEND=jax
```

## Keras 3 で動作確認済みの例

| スクリプト | 内容 | 既定のエポック数 |
| --- | --- | --- |
| `autoencoder/mnist_autoencoder.py` | 全結合1層のオートエンコーダ | 100 |
| `autoencoder/mnist_autoencoder_l1.py` | 中間層にL1正則化をかけたオートエンコーダ | 100 |
| `autoencoder/mnist_deep_autoencoder.py` | 全結合3層のオートエンコーダ | 100 |
| `autoencoder/mnist_conv_autoencoder.py` | 畳み込みオートエンコーダ | 50 |

どのスクリプトも MNIST を使い、テスト画像10枚の元画像（上段）と再構成画像（下段）を PNG に保存します。

```bash
.venv/bin/python autoencoder/mnist_autoencoder.py --epochs 5 --output reconstruction.png
```

| オプション | 意味 |
| --- | --- |
| `--epochs` | 学習エポック数 |
| `--output` | 再構成画像の保存先（既定は `reconstruction.png`） |

MNIST は初回実行時に `~/.keras/datasets/` へダウンロードされます。

上記以外のディレクトリのスクリプトは Keras 1 の API で書かれており、Keras 3 ではそのままでは動きません。

## テスト

```bash
.venv/bin/python -m pytest
```

テストは合成した極小データを使い、データセットのダウンロードなしに CPU だけで実行できます。GitHub Actions でも push と pull request ごとに同じテストを実行します。
