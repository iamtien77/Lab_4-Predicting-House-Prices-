"""Exercise model contracts and persistence using small deterministic fixtures."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.models.baseline_models import BaselineRegressionSuite
from src.models.mlp_model import build_mlp_model, predict_mlp, train_mlp_model, evaluate_mlp_model
from src.models.persistence import (load_baseline_models, load_mlp_model,
    load_preprocessor, save_baseline_models, save_mlp_model, save_preprocessor)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    return X, X[:, 0] * 3 - X[:, 1] + 10


def test_baseline_training_and_round_trip(regression_data, tmp_path):
    X, y = regression_data
    suite = BaselineRegressionSuite(rf_n_estimators=3)
    with pytest.raises(ValueError):
        suite.predict_all(X)
    models = suite.fit_all(X, y)
    predictions = suite.predict_all(X[:1])
    assert list(predictions) == ["Linear Regression", "Decision Tree", "Random Forest"]
    path = tmp_path / "baseline.joblib"
    save_baseline_models(models, path)
    loaded = load_baseline_models(path)
    for name, values in predictions.items():
        assert values.shape == (1,)
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values, loaded[name].predict(X[:1]))
    with pytest.raises(ValueError):
        suite.fit_all(np.full((4, 3), np.nan), np.ones(4))


def test_mlp_architecture_training_and_round_trip(regression_data, tmp_path):
    torch.manual_seed(42)
    X, y = regression_data
    model = build_mlp_model(X.shape[1], learning_rate=.002)
    assert [layer.out_features for layer in model.network if isinstance(layer, torch.nn.Linear)] == [32, 16, 1]
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in model.network) == 2
    before = [weight.detach().clone() for weight in model.parameters()]
    model, history = train_mlp_model(model, X[:30], y[:30], X[30:], y[30:], epochs=1, batch_size=8, verbose=0)
    assert {"loss", "mae", "val_loss", "val_mae"} == set(history)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    predictions = predict_mlp(model, X[:1])
    assert predictions.shape == (1,)
    path = tmp_path / "model.pt"
    save_mlp_model(model, path)
    np.testing.assert_allclose(predictions, predict_mlp(load_mlp_model(path), X[:1]), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        predict_mlp(model, np.ones((2, 4)))
    with pytest.raises(ValueError):
        train_mlp_model(model, X, y, X, y, epochs=0)


def test_train_only_scaling_usd_metrics_and_inference_is_read_only(regression_data):
    X, y = regression_data
    y = y * 20000
    shifted_validation = y[30:] + 300000
    model, history = train_mlp_model(build_mlp_model(3), X[:30], y[:30], X[30:],
                                     shifted_validation, epochs=1, batch_size=8, verbose=0)
    assert model.target_mean.item() == pytest.approx(y[:30].mean())
    assert model.target_scale.item() == pytest.approx(y[:30].std())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    error = predict_mlp(model, X[30:]) - shifted_validation
    assert history["val_loss"][0] == pytest.approx(np.mean(error ** 2), rel=1e-6)
    assert history["val_mae"][0] == pytest.approx(np.mean(np.abs(error)), rel=1e-6)
    metrics = evaluate_mlp_model(model, X[30:], shifted_validation)
    assert metrics["test_mse"] == pytest.approx(history["val_loss"][0], rel=1e-6)
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)
    assert not model.training


def test_best_validation_checkpoint_and_seed_reproducibility(regression_data):
    X, y = regression_data
    runs = []
    for _ in range(2):
        torch.manual_seed(17)
        model, history = train_mlp_model(build_mlp_model(3, learning_rate=0.1),
            X[:30], y[:30], X[30:], y[30:], epochs=12, batch_size=7, verbose=0)
        assert model.best_epoch == int(np.argmin(history["val_loss"])) + 1
        assert evaluate_mlp_model(model, X[30:], y[30:])["test_mse"] == pytest.approx(
            min(history["val_loss"]), rel=1e-6)
        runs.append((history, predict_mlp(model, X)))
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_constant_target_and_invalid_checkpoint(regression_data, tmp_path):
    X, _ = regression_data
    model = build_mlp_model(3)
    with pytest.raises(ValueError, match="trained"):
        predict_mlp(model, X)
    with pytest.raises(ValueError, match="trained"):
        save_mlp_model(model, tmp_path / "untrained.pt")
    model, _ = train_mlp_model(model, X[:30], np.full(30, 100000.), X[30:],
                               np.full(10, 100000.), epochs=1, verbose=0)
    assert model.target_scale.item() == 1
    assert np.isfinite(predict_mlp(model, X)).all()
    path = tmp_path / "model.pt"
    save_mlp_model(model, path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["state_dict"]["target_scale"] = torch.tensor(0., dtype=torch.float64)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="positive scale"):
        load_mlp_model(path)


def test_preprocessor_persistence(tmp_path):
    X = pd.DataFrame({"feature": [1., 2., 5.]})
    preprocessor = ColumnTransformer([("numeric", StandardScaler(), ["feature"])])
    path = tmp_path / "preprocessor.joblib"
    with pytest.raises(ValueError):
        save_preprocessor(preprocessor, path)
    preprocessor.fit(X)
    save_preprocessor(preprocessor, path)
    np.testing.assert_allclose(preprocessor.transform(X), load_preprocessor(path).transform(X))
    with pytest.raises(FileNotFoundError):
        load_preprocessor(tmp_path / "missing.joblib")
    with pytest.raises(TypeError):
        save_preprocessor({}, path)
