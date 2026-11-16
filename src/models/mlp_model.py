"""CPU PyTorch regression with train-only target scaling and USD predictions."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.validation import check_array, check_X_y


class HousePriceMLP(nn.Module):
    """Forward predicts standardized targets; public prediction returns USD."""

    def __init__(self, input_dim, learning_rate=0.001):
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, (int, np.integer)) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        self.input_dim = int(input_dim)
        self.learning_rate = float(learning_rate)
        self.network = nn.Sequential(nn.Linear(self.input_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.register_buffer("target_mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("target_scale", torch.tensor(1.0, dtype=torch.float64))
        self.is_fitted = False
        self.best_epoch = None
        self.best_val_loss = None

    def forward(self, features):
        return self.network(features).squeeze(-1)


def build_mlp_model(input_dim: int, learning_rate: float = 0.001):
    """Build the fixed 32/16/1 network on CPU."""
    return HousePriceMLP(input_dim, learning_rate)


def _features(model, X):
    X = check_array(X, dtype=np.float32)
    if X.shape[1] != model.input_dim:
        raise ValueError("Input feature dimension does not match the model.")
    return torch.from_numpy(np.ascontiguousarray(X))


def _epoch_metrics(model, features, targets, batch_size):
    """Evaluate a fixed epoch snapshot, weighting every sample equally."""
    model.eval()
    squared_error = absolute_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            prediction = model(features[start:start + batch_size]).double()
            prediction = prediction * model.target_scale + model.target_mean
            error = prediction - targets[start:start + batch_size]
            squared_error += error.square().sum().item()
            absolute_error += error.abs().sum().item()
    metrics = squared_error / len(features), absolute_error / len(features)
    if not np.isfinite(metrics).all():
        raise ValueError("Training produced non-finite loss or metrics.")
    return metrics


def train_mlp_model(model, X_train, y_train, X_val, y_val,
                    epochs: int = 100, batch_size: int = 32, verbose: int = 1):
    """Fit scaling on train, run all epochs, restore lowest validation MSE.

    History contains end-of-epoch MSE (USD²) and MAE (USD), not the
    standardized objective optimized by Adam. Calling again refits target
    statistics and resets Adam, but continues from existing network weights.
    """
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    X_train, y_train = check_X_y(X_train, y_train, y_numeric=True)
    X_val, y_val = check_X_y(X_val, y_val, y_numeric=True)
    train_features, val_features = _features(model, X_train), _features(model, X_val)
    train_targets = torch.as_tensor(np.array(y_train, dtype=np.float64, copy=True))
    val_targets = torch.as_tensor(np.array(y_val, dtype=np.float64, copy=True))
    model.cpu()
    model.is_fitted = False
    model.best_epoch = model.best_val_loss = None
    mean, scale = train_targets.mean(), train_targets.std(unbiased=False)
    if not torch.isfinite(mean) or not torch.isfinite(scale):
        raise ValueError("Target statistics must be finite.")
    model.target_mean.copy_(mean)
    model.target_scale.copy_(scale if scale > 0 else torch.tensor(1.0))
    normalized_targets = ((train_targets - model.target_mean) / model.target_scale).float()
    generator = torch.Generator().manual_seed(torch.initial_seed())
    loader = DataLoader(TensorDataset(train_features, normalized_targets),
                        batch_size=int(batch_size), shuffle=True, generator=generator,
                        num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    criterion = nn.MSELoss()
    history = {key: [] for key in ("loss", "mae", "val_loss", "val_mae")}
    best_state = None
    for epoch in range(int(epochs)):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite objective.")
            loss.backward()
            optimizer.step()
        train_mse, train_mae = _epoch_metrics(model, train_features, train_targets, batch_size)
        val_mse, val_mae = _epoch_metrics(model, val_features, val_targets, batch_size)
        for key, value in zip(history, (train_mse, train_mae, val_mse, val_mae)):
            history[key].append(value)
        if model.best_val_loss is None or val_mse < model.best_val_loss:
            model.best_val_loss, model.best_epoch = val_mse, epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if verbose:
            print(f"Kỷ nguyên (Epoch) {epoch + 1}/{epochs}: MSE tập huấn luyện (train)={train_mse:.2f} USD²; "
                  f"MSE tập kiểm định (val)={val_mse:.2f} USD²; MAE tập kiểm định (val)={val_mae:.2f} USD")
    model.load_state_dict(best_state)
    model.eval()
    model.is_fitted = True
    return model, history


def evaluate_mlp_model(model, X_test, y_test):
    """Return test MSE in USD² and MAE in USD without changing weights."""
    X_test, y_test = check_X_y(X_test, y_test, y_numeric=True)
    error = predict_mlp(model, X_test) - np.asarray(y_test, dtype=np.float64)
    metrics = {"test_mse": float(np.mean(error ** 2)), "test_mae": float(np.mean(np.abs(error)))}
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("Evaluation produced non-finite metrics.")
    return metrics


def predict_mlp(model, X):
    """Return finite USD predictions with shape (N,), including one sample."""
    if not model.is_fitted:
        raise ValueError("The MLP must be trained before prediction.")
    features = _features(model, X)
    model.eval()
    with torch.inference_mode():
        predictions = (model(features).double() * model.target_scale + model.target_mean).cpu().numpy()
    if predictions.shape != (len(features),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid predictions.")
    return predictions
