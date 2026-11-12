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

データセット（MNIST / CIFAR-10 / Boston Housing）は初回実行時に `~/.keras/datasets/` へダウンロードされます。

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

### VGG16 による画像分類と転移学習

`vgg16/test_vgg16/test_vgg16.py` は、ImageNet で学習済みの VGG16 で画像を分類し、確率の高い5クラスを表示します。

```bash
.venv/bin/python vgg16/test_vgg16/test_vgg16.py vgg16/test_vgg16/elephant.jpg
```

`vgg16/transfer_learning.py` は、クラスごとのサブディレクトリに画像を置いた訓練用と検証用のディレクトリから画像分類モデルを学習します。クラスはサブディレクトリ名（名前順。`.` で始まる隠しディレクトリは除く）です。

```bash
cd vgg16/17flowers
../../.venv/bin/python setup.py  # jpg/ の画像を train_images/ と test_images/ にクラスごとに振り分ける
../../.venv/bin/python ../transfer_learning.py --method finetuning \
    --train-dir train_images --validation-dir test_images
../../.venv/bin/python ../predict.py --model results/model_finetuning.keras jpg/image_0001.jpg
```

`vgg16/dogs_vs_cats/setup.py` は Kaggle の Dogs vs. Cats の `train/` から、`data/train/` と `data/validation/` に犬と猫の画像を振り分けます。

| `--method` | 学習方法 | 最適化手法 |
| --- | --- | --- |
| `smallcnn` | 畳み込み3層の小さな CNN をゼロから学習 | Adam |
| `extractor` | ImageNet で学習済みの VGG16 の畳み込み部分で特徴量を一度だけ計算し、その上の全結合層だけを学習 | SGD（学習率 1e-4、モーメンタム 0.9） |
| `finetuning` | 学習済みの VGG16 のうち最後のブロック（block5）と全結合層を学習 | SGD（学習率 1e-4、モーメンタム 0.9） |
| `scratch` | VGG16 と同じ構成のネットワークを、学習済みの重みを使わずにゼロから学習 | Adam |

| オプション | 意味 |
| --- | --- |
| `--train-dir` / `--validation-dir` | 訓練用・検証用の画像のディレクトリ（必須） |
| `--method` | 学習方法（必須） |
| `--preprocess` | `rescale`（既定。画素値を 0〜1 にする）または `vgg16`（ImageNet の学習時と同じ VGG16 用の前処理） |
| `--epochs` | 学習エポック数（既定は50） |
| `--image-size` | 画像を縦横このサイズにそろえる（既定は150） |
| `--top-weights` | `finetuning` のみ。`extractor` で学習した全結合層の重みから学習を始める |
| `--name` | 出力ファイル名に使う名前（既定は学習方法の名前。`--preprocess vgg16` のときは末尾に `_with_preprocess` が付く） |
| `--result-dir` | 結果の保存先ディレクトリ（既定は実行したディレクトリの `results`。なければ作成） |

`smallcnn` / `finetuning` / `scratch` は、学習時に画像をランダムに左右反転・拡大縮小（±20%）・せん断（最大約11度）してデータを拡張します。VGG16 を使う3つの方法では、VGG16 の畳み込み部分の上に、256ユニットの隠れ層（Dropout 0.5）とクラス数ぶんの出力（softmax）を載せます。

訓練用と検証用の画像はすべてメモリに読み込みます（150×150 の画像 2,800 枚で約 760 MB）。

保存先ディレクトリに次のファイルを書き出します（`<名前>` は `--name` で指定した名前、または既定の名前）。

| ファイル | 内容 |
| --- | --- |
| `history_<名前>.txt` | エポックごとの `loss` / `acc` / `val_loss` / `val_acc`（タブ区切り） |
| `model_<名前>.keras` | 前処理を含む学習済みモデル |
| `classes.txt` | クラス名（1行に1クラス、モデルの出力の順） |
| `top_<名前>.weights.h5` | `extractor` のみ。学習した全結合層の重み（`--top-weights` に渡せる） |

`vgg16/17flowers/plot_results.py` と `vgg16/dogs_vs_cats/plot_results.py` は、実行したディレクトリの `results/` にある次の学習履歴を読み込み、学習方法ごとの検証損失と検証精度を比較するグラフを `results/val_loss.png` / `results/val_acc.png` に保存します。

| スクリプト | 読み込む学習履歴（作り方） |
| --- | --- |
| `17flowers/plot_results.py` | `history_smallcnn.txt`（`--method smallcnn`）、`history_finetuning.txt`（`--method finetuning`） |
| `dogs_vs_cats/plot_results.py` | `history_smallcnn.txt`（`--method smallcnn`）、`history_extractor.txt`（`--method extractor`）、`history_finetuning.txt`（`--method finetuning --top-weights results/top_extractor.weights.h5`）、`history_finetuning_from_random.txt`（`--method finetuning --name finetuning_from_random`） |

`vgg16/predict.py --model <モデル> <画像>...` は、モデルと同じディレクトリの `classes.txt` を読み、画像ごとに確率の高い最大5クラスを表示します。前処理はモデルに含まれているため、学習時と同じ前処理が自動でかかります。

ImageNet で学習済みの VGG16 の重みは、初回実行時に `~/.keras/models/` へダウンロードされます（`include_top` なしで約 60 MB、`test_vgg16.py` が使う全結合層付きで約 550 MB）。

### 表形式データの分類と回帰

| スクリプト | 内容 | 既定のエポック数 |
| --- | --- | --- |
| `logreg/ex2data1.py` | `data/ex2data1.txt`（2つの試験の点数と合否）をロジスティック回帰で分類し、決定境界を描く | 1000 |
| `mlp/pima.py` | `data/pima-indians-diabetes.data`（Pima Indians Diabetes）の糖尿病の有無を多層パーセプトロンで分類 | 150 |
| `mlp/iris.py` | scikit-learn 付属の Iris を多層パーセプトロンで3クラスに分類（8割で学習、2割で評価） | 50 |
| `regression/boston.py` | Boston Housing の住宅価格を回帰し、10分割交差検証の平均二乗誤差（MSE）を表示 | 50 |

どれも `--epochs` で学習エポック数を指定できます。データファイルはスクリプトの場所から探すので、どのディレクトリから実行してもかまいません。

```bash
.venv/bin/python regression/boston.py --model deeper
```

| スクリプト | オプション | 意味 |
| --- | --- | --- |
| `logreg/ex2data1.py` | `--result-dir` | 結果の保存先ディレクトリ（既定は実行したディレクトリの `result`。なければ作成）。標準化した訓練データの散布図を `data.png`、決定境界を重ねた図を `result.png` に保存する |
| `mlp/pima.py` | `--checkpoint-dir` | 指定すると、データの3分の1を検証データにし、検証精度が最高を更新するたびに重みを `weights-improvement-<エポック>-<検証精度>.weights.h5` としてこのディレクトリに保存する |
| `regression/boston.py` | `--model` | `baseline`（既定。隠れ層13ユニット）、`deeper`（隠れ層13・6ユニット）、`wider`（隠れ層20ユニット） |
| `regression/boston.py` | `--no-standardize` | 入力を標準化しない（既定では交差検証の各分割で訓練データから平均と標準偏差を求めて標準化する） |
| `regression/boston.py` | `--folds` | 交差検証の分割数（既定は10） |

`mlp/pima.py` は学習後に、全768件での精度を表示します（`--checkpoint-dir` を指定したときも同じ）。

### そのほか

上記以外のスクリプトは Keras 3 での動作を確認していません（多くは Keras 1 の API で書かれています）。

`blog/` は各例を解説したブログ記事の原稿で、Keras 1 の API とそのころのスクリプトに基づいています。

## テスト

```bash
.venv/bin/python -m pytest
```

テストは合成した極小データを使い、データセットのダウンロードなしに CPU だけで実行できます。GitHub Actions でも pull request と master への push ごとに同じテストを実行します。
