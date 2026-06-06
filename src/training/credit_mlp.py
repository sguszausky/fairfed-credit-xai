"""Shared MLP model and training helpers for credit-default prediction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SCHEDULER_FACTOR = 0.5
DEFAULT_SCHEDULER_PATIENCE = 3
DEFAULT_HIDDEN_DIMS = (256, 128, 64)
DEFAULT_DROPOUT = (0.25, 0.20, 0.15)
DEFAULT_USE_BATCH_NORM = True

DEFAULT_MLP_CONFIG = {
    "hidden_dims": DEFAULT_HIDDEN_DIMS,
    "dropout": DEFAULT_DROPOUT,
    "batch_norm": DEFAULT_USE_BATCH_NORM,
    "batch_size": DEFAULT_BATCH_SIZE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "use_pos_weight": True,
}


def set_seed(seed=DEFAULT_SEED):
    """Seed Python, NumPy, and PyTorch for repeatable notebook runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_mlp_config(config=None):
    """Return a complete MLP config using shared defaults for missing keys."""
    merged = dict(DEFAULT_MLP_CONFIG)
    if config:
        merged.update(config)

    if config and "lr" in config:
        merged["learning_rate"] = config["lr"]

    hidden_dims = tuple(merged["hidden_dims"])
    dropout = merged["dropout"]
    if isinstance(dropout, (int, float)):
        dropout = tuple(float(dropout) for _ in hidden_dims)
    else:
        dropout = tuple(dropout)

    if len(dropout) != len(hidden_dims):
        raise ValueError("dropout must be a scalar or match the number of hidden layers.")

    merged["hidden_dims"] = hidden_dims
    merged["dropout"] = dropout
    merged["batch_norm"] = bool(merged.get("batch_norm", False))
    merged["batch_size"] = int(merged["batch_size"])
    merged["learning_rate"] = float(merged["learning_rate"])
    merged["weight_decay"] = float(merged["weight_decay"])
    merged["use_pos_weight"] = bool(merged.get("use_pos_weight", True))
    return merged


def serializable_mlp_config(config):
    """Convert tuple-heavy configs into JSON-friendly plain values."""
    normalized = normalize_mlp_config(config)
    return {
        **normalized,
        "hidden_dims": list(normalized["hidden_dims"]),
        "dropout": list(normalized["dropout"]),
    }


def load_mlp_config(path, fallback=None):
    """Load a saved MLP config JSON, or return shared defaults if it is missing."""
    import json

    path = Path(path)
    if path.exists():
        with path.open() as file:
            return normalize_mlp_config(json.load(file))
    return normalize_mlp_config(fallback)


def save_mlp_config(config, path):
    """Save the MLP config used for training so client runs can reuse it."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(serializable_mlp_config(config), file, indent=2)


class CreditDefaultMLP(nn.Module):
    """Configurable MLP architecture used by global and client-local training."""

    def __init__(
        self,
        input_dim,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        dropout=DEFAULT_DROPOUT,
        use_batch_norm=DEFAULT_USE_BATCH_NORM,
    ):
        super().__init__()
        model_config = normalize_mlp_config(
            {
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "batch_norm": use_batch_norm,
            }
        )
        layers = []
        previous_dim = input_dim

        for hidden_dim, dropout_rate in zip(
            model_config["hidden_dims"], model_config["dropout"]
        ):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if model_config["batch_norm"]:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StateDictCreditDefaultMLP(nn.Module):
    """MLP wrapper reconstructed from a saved ``net.*`` state dict."""

    def __init__(self, state_dict):
        super().__init__()
        self.net = build_net_from_state_dict(state_dict)

    def forward(self, x):
        return self.net(x)


def build_net_from_state_dict(state_dict):
    """Rebuild a Sequential MLP whose module indexes match a saved checkpoint."""
    linear_indices = sorted(
        int(key.split(".")[1])
        for key, value in state_dict.items()
        if key.startswith("net.") and key.endswith(".weight") and value.ndim == 2
    )
    if not linear_indices:
        raise ValueError("No Linear layers found in model_state_dict.")

    batch_norm_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("net.") and key.endswith(".running_mean")
    }
    modules = []
    linear_position = 0
    max_idx = linear_indices[-1]

    for idx in range(max_idx + 1):
        weight = state_dict.get(f"net.{idx}.weight")
        bias = state_dict.get(f"net.{idx}.bias")
        if weight is not None and getattr(weight, "ndim", None) == 2:
            out_features, in_features = weight.shape
            modules.append(nn.Linear(in_features, out_features, bias=bias is not None))
            linear_position += 1
        elif idx in batch_norm_indices:
            modules.append(nn.BatchNorm1d(state_dict[f"net.{idx}.running_mean"].shape[0]))
        elif linear_position < len(linear_indices):
            modules.append(nn.ReLU() if idx % 2 == 0 else nn.Dropout(0.0))
        else:
            modules.append(nn.Identity())

    return nn.Sequential(*modules)


def build_mlp_from_artifact(artifact, strict=True):
    """Build a CreditDefaultMLP from an artifact, falling back to checkpoint shape."""
    config = normalize_mlp_config(artifact.get("model_config"))
    state_dict = artifact["model_state_dict"]
    try:
        model = CreditDefaultMLP(
            input_dim=artifact["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            use_batch_norm=config["batch_norm"],
        )
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        model = StateDictCreditDefaultMLP(state_dict)
        model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, config


def make_dataset(X_array, y_series):
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_series.to_numpy(), dtype=torch.float32).view(-1, 1)
    return TensorDataset(X_tensor, y_tensor)


def make_loader(X_array, y_series, batch_size=DEFAULT_BATCH_SIZE, shuffle=False):
    return DataLoader(make_dataset(X_array, y_series), batch_size=batch_size, shuffle=shuffle)


def make_training_components(
    model,
    y_train,
    device,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    use_pos_weight=True,
):
    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - y_train.sum())
    if positive_count == 0:
        raise ValueError("Training split has no positive examples.")

    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if use_pos_weight
        else nn.BCEWithLogitsLoss()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=DEFAULT_SCHEDULER_FACTOR,
        patience=DEFAULT_SCHEDULER_PATIENCE,
    )
    return criterion, optimizer, scheduler, pos_weight


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_probs = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(xb)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        all_targets.append(yb.detach().cpu().numpy().ravel())

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return avg_loss, auc, pr_auc


def predict_proba(model, loader, device):
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            targets.append(yb.numpy().ravel())
    return np.concatenate(targets), np.concatenate(probs)


def best_f1_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    f1_scores = [f1_score(y_true, y_prob >= threshold, zero_division=0) for threshold in thresholds]
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train_credit_default_mlp(
    data_bundle,
    device,
    model_config=None,
    seed=DEFAULT_SEED,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    verbose=True,
):
    """Train the shared configurable MLP and return the best model plus metadata."""
    set_seed(seed)
    model_config = normalize_mlp_config(model_config)

    X_train = data_bundle["X_train"]
    X_val = data_bundle["X_val"]
    y_train = data_bundle["y_train"]
    y_val = data_bundle["y_val"]

    train_loader = make_loader(X_train, y_train, batch_size=model_config["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=model_config["batch_size"], shuffle=False)

    model = CreditDefaultMLP(
        input_dim=X_train.shape[1],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
        use_batch_norm=model_config["batch_norm"],
    ).to(device)
    criterion, optimizer, scheduler, pos_weight = make_training_components(
        model,
        y_train,
        device,
        learning_rate=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
        use_pos_weight=model_config["use_pos_weight"],
    )

    best_val_auc = -np.inf
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_auc, train_pr_auc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_auc, val_pr_auc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_pr_auc": train_pr_auc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_pr_auc": val_pr_auc,
            }
        )

        if verbose:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} auc {train_auc:.4f} pr_auc {train_pr_auc:.4f} | "
                f"val loss {val_loss:.4f} auc {val_auc:.4f} pr_auc {val_pr_auc:.4f}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping after {epoch} epochs.")
            break

    model.load_state_dict(best_state)
    y_val_true, y_val_prob = predict_proba(model, val_loader, device)
    best_threshold, best_val_f1 = best_f1_threshold(y_val_true, y_val_prob)

    return {
        "model": model,
        "history": history,
        "best_state": best_state,
        "best_val_auc": float(best_val_auc),
        "best_threshold": best_threshold,
        "best_val_f1": best_val_f1,
        "pos_weight": float(pos_weight.item()),
        "batch_size": model_config["batch_size"],
        "model_config": serializable_mlp_config(model_config),
    }
