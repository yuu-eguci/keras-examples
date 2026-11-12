import keras
import numpy as np
import pytest

from conftest import load_example


def dense_units(model):
    return [layer.units for layer in model.layers if isinstance(layer, keras.layers.Dense)]


@pytest.mark.parametrize("script, epochs", [
    ("logreg/ex2data1.py", 1000),
    ("mlp/pima.py", 150),
    ("mlp/iris.py", 50),
    ("regression/boston.py", 50),
])
def test_default_epochs(script, epochs):
    assert load_example(script).parse_args([]).epochs == epochs


def test_logreg_loads_standardized_bundled_data():
    x, t = load_example("logreg/ex2data1.py").load_data()

    assert x.shape == (100, 2)
    assert set(np.unique(t)) == {0.0, 1.0}
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(x.std(axis=0), 1.0, atol=1e-6)


def test_logreg_main_saves_data_and_decision_boundary(tmp_path):
    load_example("logreg/ex2data1.py").main(["--epochs", "1", "--result-dir", str(tmp_path)])

    for name in ["data.png", "result.png"]:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_pima_model_and_bundled_data():
    module = load_example("mlp/pima.py")

    x, y = module.load_data()
    model = module.build_model()

    assert x.shape == (768, 8)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert model.input_shape == (None, 8)
    assert dense_units(model) == [12, 8, 1]


def test_pima_main_prints_accuracy(capsys):
    load_example("mlp/pima.py").main(["--epochs", "1"])

    assert "accuracy:" in capsys.readouterr().out


def test_pima_checkpoint_saves_best_weights(tmp_path):
    load_example("mlp/pima.py").main(["--epochs", "1", "--checkpoint-dir", str(tmp_path)])

    (saved,) = tmp_path.glob("weights-improvement-01-*.weights.h5")
    assert saved.stat().st_size > 0


def test_iris_main_trains_three_class_classifier(capsys):
    module = load_example("mlp/iris.py")

    model = module.build_model()
    module.main(["--epochs", "1"])

    assert model.input_shape == (None, 4)
    assert dense_units(model) == [16, 3]
    assert "Accuracy = " in capsys.readouterr().out


@pytest.mark.parametrize("name, units", [
    ("baseline", [13, 1]),
    ("deeper", [13, 6, 1]),
    ("wider", [20, 1]),
])
def test_boston_models(name, units):
    model = load_example("regression/boston.py").build_model(name)

    assert model.input_shape == (None, 13)
    assert dense_units(model) == units


@pytest.fixture
def fake_boston_housing(monkeypatch):
    calls = []
    rng = np.random.default_rng(0)

    def load_data(**kwargs):
        calls.append(kwargs)
        x = rng.normal(size=(20, 13))
        y = rng.normal(20, 5, size=20)
        return (x, y), (x[:0], y[:0])

    monkeypatch.setattr(keras.datasets.boston_housing, "load_data", load_data)
    return calls


def test_boston_main_cross_validates_on_all_samples(fake_boston_housing, capsys):
    load_example("regression/boston.py").main(["--epochs", "1", "--folds", "2"])

    assert fake_boston_housing == [{"test_split": 0}]
    # 交差検証中は進捗バーを出さず、正の MSE を1行だけ表示する
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("MSE: ")
    assert float(line.split()[1]) > 0


@pytest.mark.parametrize("standardize", [True, False])
def test_boston_trains_each_fold_with_given_settings(standardize, monkeypatch):
    module = load_example("regression/boston.py")
    rng = np.random.default_rng(0)
    x = rng.normal(50, 10, size=(12, 13))
    y = rng.normal(20, 5, size=12)
    fits = []
    original_fit = keras.Model.fit

    def recording_fit(self, x_fold, y_fold, **kwargs):
        fits.append((x_fold, kwargs))
        return original_fit(self, x_fold, y_fold, **kwargs)

    monkeypatch.setattr(keras.Model, "fit", recording_fit)

    mses = module.cross_validate(x, y, "baseline", standardize, epochs=2, folds=3)

    assert mses.shape == (3,)
    assert (mses > 0).all()
    assert len(fits) == 3
    for x_fold, kwargs in fits:
        assert kwargs["epochs"] == 2
        assert kwargs["batch_size"] == 5
        assert len(x_fold) == 8
        # 標準化するときは、その分割の訓練データだけから求めた平均・標準偏差で標準化する
        if standardize:
            np.testing.assert_allclose(x_fold.mean(axis=0), 0.0, atol=1e-6)
            np.testing.assert_allclose(x_fold.std(axis=0), 1.0, atol=1e-6)
        else:
            assert x_fold.mean() > 40
