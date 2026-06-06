"""Train a federated MLP with Flower on the prepared client CSV files.

The script uses FedAvg over the same CreditDefaultMLP architecture used by the
global and client-local notebooks. It writes a final federated model artifact
and a per-client summary CSV that can be picked up by
compare_global_local_models.ipynb.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
except ImportError as exc:  # pragma: no cover - exercised only without Flower.
    raise SystemExit(
        "Flower is not installed. Install it in your project environment with:\n"
        '  pip install -U "flwr[simulation]"\n'
    ) from exc

from credit_data import (
    CATEGORICAL_COLS,
    DEFAULT_TARGET_COL,
    add_features,
    make_one_hot_encoder,
)
from credit_mlp import (
    CreditDefaultMLP,
    DEFAULT_SEED,
    best_f1_threshold,
    load_mlp_config,
    make_loader,
    make_training_components,
    normalize_mlp_config,
    predict_proba,
    run_epoch,
    serializable_mlp_config,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIENT_DATA_DIR = REPO_ROOT / "data" / "IID_cl3_age_unbalanced"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FedAvg with Flower on local credit-default client datasets."
    )
    parser.add_argument("--client-data-dir", type=Path, default=DEFAULT_CLIENT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument(
        "--feature-set",
        choices=["baseline", "engineered"],
        default="engineered",
        help="Feature set to use before shared preprocessing.",
    )
    parser.add_argument(
        "--initialization",
        choices=["average-local", "global", "random"],
        default="average-local",
        help=(
            "Initial server weights. average-local averages compatible local model "
            "artifacts from models/<client-data-dir-name>/ before FL starts."
        ),
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument("--min-fit-clients", type=int, default=None)
    parser.add_argument("--min-evaluate-clients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument("--client-gpus", type=float, default=0.0)
    parser.add_argument(
        "--save-history-json",
        action="store_true",
        help="Also save Flower's raw history object as a JSON-ish text dump.",
    )
    return parser.parse_args()


def discover_client_files(client_data_dir: Path) -> list[Path]:
    nested = sorted(client_data_dir.glob("client_*/client.csv"))
    if nested:
        return nested

    flat = sorted(client_data_dir.glob("client_*.csv"))
    if flat:
        return flat

    raise FileNotFoundError(
        f"No client CSV files found under {client_data_dir}. "
        "Expected client_000/client.csv or client_000.csv."
    )


def client_name_from_path(client_csv: Path) -> str:
    return client_csv.parent.name if client_csv.name == "client.csv" else client_csv.stem


def split_raw_client_data(
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    feature_set: str,
) -> dict[str, object]:
    model_df = add_features(df) if feature_set == "engineered" else df.copy()
    X_raw = model_df.drop(columns=["ID", target_col])
    y_raw = model_df[target_col].astype(np.float32)

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw,
        y_raw,
        test_size=0.30,
        random_state=seed,
        stratify=y_raw,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    return {
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def build_shared_preprocessor(raw_splits: dict[str, dict[str, object]]) -> ColumnTransformer:
    first_split = next(iter(raw_splits.values()))
    X_train_raw = first_split["X_train_raw"]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in X_train_raw.columns]
    numeric_cols = [col for col in X_train_raw.columns if col not in categorical_cols]
    train_frames = [split["X_train_raw"] for split in raw_splits.values()]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", make_one_hot_encoder(), categorical_cols),
        ]
    )
    preprocessor.fit(pd.concat(train_frames, axis=0, ignore_index=True))
    return preprocessor


def transform_client_splits(
    raw_splits: dict[str, dict[str, object]],
    preprocessor: ColumnTransformer,
) -> dict[str, dict[str, object]]:
    bundles = {}
    for client_name, split in raw_splits.items():
        bundles[client_name] = {
            "X_train": preprocessor.transform(split["X_train_raw"]).astype(np.float32),
            "X_val": preprocessor.transform(split["X_val_raw"]).astype(np.float32),
            "X_test": preprocessor.transform(split["X_test_raw"]).astype(np.float32),
            "y_train": split["y_train"],
            "y_val": split["y_val"],
            "y_test": split["y_test"],
        }
    return bundles


def load_client_bundles(
    client_files: list[Path],
    target_col: str,
    seed: int,
    feature_set: str,
) -> tuple[dict[str, dict[str, object]], ColumnTransformer]:
    raw_splits = {}
    for client_csv in client_files:
        client_name = client_name_from_path(client_csv)
        raw_splits[client_name] = split_raw_client_data(
            pd.read_csv(client_csv),
            target_col=target_col,
            seed=seed,
            feature_set=feature_set,
        )

    preprocessor = build_shared_preprocessor(raw_splits)
    return transform_client_splits(raw_splits, preprocessor), preprocessor


def get_model_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [value.cpu().numpy() for value in model.state_dict().values()]


def set_model_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(parameters) != len(state_dict):
        raise ValueError(
            f"Expected {len(state_dict)} parameter arrays, got {len(parameters)}."
        )

    new_state = {}
    for key, value in zip(state_dict.keys(), parameters):
        new_state[key] = torch.tensor(
            value,
            dtype=state_dict[key].dtype,
            device=state_dict[key].device,
        )
    model.load_state_dict(new_state, strict=True)


def safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "test_pr_auc": float(average_precision_score(y_true, y_prob)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["test_roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["test_roc_auc"] = float("nan")
    return metrics


def evaluate_loss(model: torch.nn.Module, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = criterion(model(xb), yb)
            total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def pooled_bundle(client_bundles: dict[str, dict[str, object]], split: str) -> tuple[np.ndarray, pd.Series]:
    X = np.concatenate([bundle[f"X_{split}"] for bundle in client_bundles.values()], axis=0)
    y = pd.concat(
        [bundle[f"y_{split}"] for bundle in client_bundles.values()],
        axis=0,
        ignore_index=True,
    )
    return X, y


def make_model(input_dim: int, model_config: dict) -> CreditDefaultMLP:
    return CreditDefaultMLP(
        input_dim=input_dim,
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
        use_batch_norm=model_config["batch_norm"],
    )


def load_torch_artifact(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compatible_artifact_parameters(
    artifact: dict,
    template_model: torch.nn.Module,
) -> list[np.ndarray] | None:
    if int(artifact.get("input_dim", -1)) != template_model.net[0].in_features:
        return None

    candidate = make_model(
        input_dim=artifact["input_dim"],
        model_config=normalize_mlp_config(artifact.get("model_config")),
    )
    try:
        candidate.load_state_dict(artifact["model_state_dict"], strict=True)
    except RuntimeError:
        return None

    template_shapes = [arr.shape for arr in get_model_parameters(template_model)]
    candidate_parameters = get_model_parameters(candidate)
    if [arr.shape for arr in candidate_parameters] != template_shapes:
        return None
    return candidate_parameters


def initialize_parameters(
    initialization: str,
    template_model: torch.nn.Module,
    models_dir: Path,
    client_data_dir: Path,
) -> list[np.ndarray]:
    if initialization == "random":
        return get_model_parameters(template_model)

    if initialization == "global":
        global_model_path = models_dir / "credit_default_mlp.pt"
        if not global_model_path.exists():
            warnings.warn("Global model artifact not found; using random initialization.")
            return get_model_parameters(template_model)
        artifact = load_torch_artifact(global_model_path)
        parameters = compatible_artifact_parameters(artifact, template_model)
        if parameters is None:
            warnings.warn("Global model artifact is not shape-compatible; using random initialization.")
            return get_model_parameters(template_model)
        return parameters

    local_model_dir = models_dir / client_data_dir.name
    local_artifacts = sorted(local_model_dir.glob("client_*.pt"))
    summary_path = local_model_dir / "client_model_summary.csv"
    train_row_weights = {}
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if {"client", "train_rows"}.issubset(summary_df.columns):
            train_row_weights = dict(zip(summary_df["client"], summary_df["train_rows"]))

    weighted_parameters = None
    total_weight = 0.0
    skipped = []

    for artifact_path in local_artifacts:
        artifact = load_torch_artifact(artifact_path)
        parameters = compatible_artifact_parameters(artifact, template_model)
        if parameters is None:
            skipped.append(artifact_path.name)
            continue

        client_name = artifact_path.stem
        weight = float(train_row_weights.get(client_name, 1))
        if weighted_parameters is None:
            weighted_parameters = [weight * arr.astype(np.float64) for arr in parameters]
        else:
            weighted_parameters = [
                accum + weight * arr.astype(np.float64)
                for accum, arr in zip(weighted_parameters, parameters)
            ]
        total_weight += weight

    if skipped:
        warnings.warn(
            "Skipped local artifacts that were not compatible with the shared "
            f"federated model shape: {', '.join(skipped)}"
        )

    if weighted_parameters is None or total_weight == 0:
        warnings.warn("No compatible local artifacts found; using random initialization.")
        return get_model_parameters(template_model)

    return [(arr / total_weight).astype(np.float32) for arr in weighted_parameters]


class CreditFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_name: str,
        data_bundle: dict[str, object],
        input_dim: int,
        model_config: dict,
        local_epochs: int,
        device: torch.device,
    ):
        self.client_name = client_name
        self.data_bundle = data_bundle
        self.input_dim = input_dim
        self.model_config = model_config
        self.local_epochs = local_epochs
        self.device = device

    def _new_model(self) -> CreditDefaultMLP:
        return make_model(self.input_dim, self.model_config).to(self.device)

    def get_parameters(self, config):  # noqa: D401 - Flower API signature.
        return get_model_parameters(self._new_model())

    def fit(self, parameters, config):  # noqa: D401 - Flower API signature.
        model = self._new_model()
        set_model_parameters(model, parameters)

        train_loader = make_loader(
            self.data_bundle["X_train"],
            self.data_bundle["y_train"],
            batch_size=self.model_config["batch_size"],
            shuffle=True,
        )
        criterion, optimizer, _, _ = make_training_components(
            model,
            self.data_bundle["y_train"],
            self.device,
            learning_rate=self.model_config["learning_rate"],
            weight_decay=self.model_config["weight_decay"],
            use_pos_weight=self.model_config["use_pos_weight"],
        )

        train_loss = train_auc = train_pr_auc = float("nan")
        for _ in range(self.local_epochs):
            train_loss, train_auc, train_pr_auc = run_epoch(
                model,
                train_loader,
                criterion,
                self.device,
                optimizer=optimizer,
            )

        return (
            get_model_parameters(model),
            len(train_loader.dataset),
            {
                "train_loss": float(train_loss),
                "train_roc_auc": float(train_auc),
                "train_pr_auc": float(train_pr_auc),
            },
        )

    def evaluate(self, parameters, config):  # noqa: D401 - Flower API signature.
        model = self._new_model()
        set_model_parameters(model, parameters)

        val_loader = make_loader(
            self.data_bundle["X_val"],
            self.data_bundle["y_val"],
            batch_size=self.model_config["batch_size"],
            shuffle=False,
        )
        criterion, _, _, _ = make_training_components(
            model,
            self.data_bundle["y_train"],
            self.device,
            learning_rate=self.model_config["learning_rate"],
            weight_decay=self.model_config["weight_decay"],
            use_pos_weight=self.model_config["use_pos_weight"],
        )
        loss = evaluate_loss(model, val_loader, criterion, self.device)
        y_true, y_prob = predict_proba(model, val_loader, self.device)
        threshold, val_f1 = best_f1_threshold(y_true, y_prob)

        return (
            float(loss),
            len(val_loader.dataset),
            {
                "val_roc_auc": float(roc_auc_score(y_true, y_prob)),
                "val_pr_auc": float(average_precision_score(y_true, y_prob)),
                "val_f1": float(val_f1),
                "threshold": float(threshold),
            },
        )


def weighted_average_metrics(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated = {}
    for key in metrics[0][1].keys():
        values = [
            num_examples * client_metrics[key]
            for num_examples, client_metrics in metrics
            if key in client_metrics and np.isfinite(client_metrics[key])
        ]
        weights = [
            num_examples
            for num_examples, client_metrics in metrics
            if key in client_metrics and np.isfinite(client_metrics[key])
        ]
        if weights:
            aggregated[key] = float(sum(values) / sum(weights))
    aggregated["examples"] = float(total_examples)
    return aggregated


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )
        if parameters_aggregated is not None:
            self.latest_parameters = parameters_to_ndarrays(parameters_aggregated)
        return parameters_aggregated, metrics_aggregated


def evaluate_final_model(
    model: torch.nn.Module,
    model_config: dict,
    client_bundles: dict[str, dict[str, object]],
    threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for client_name, bundle in client_bundles.items():
        loader = make_loader(
            bundle["X_test"],
            bundle["y_test"],
            batch_size=model_config["batch_size"],
            shuffle=False,
        )
        y_true, y_prob = predict_proba(model, loader, torch.device("cpu"))
        row = {
            "model_family": "Federated MLP",
            "unit": client_name,
            "evaluation_scope": "client_test",
            "test_rows": int(len(y_true)),
            "default_rate": float(np.mean(y_true)),
            "threshold": float(threshold),
        }
        row.update(safe_binary_metrics(y_true, y_prob, threshold))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "federated_model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved federated per-client summary to {summary_path}")
    return summary_df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    client_data_dir = args.client_data_dir.resolve()
    models_dir = args.models_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else models_dir / f"{client_data_dir.name}_federated"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    client_files = discover_client_files(client_data_dir)
    client_bundles, preprocessor = load_client_bundles(
        client_files,
        target_col=args.target_col,
        seed=args.seed,
        feature_set=args.feature_set,
    )
    client_names = list(client_bundles.keys())

    base_config = load_mlp_config(models_dir / "credit_default_mlp_config.json")
    model_config = normalize_mlp_config(base_config)
    if args.batch_size is not None:
        model_config["batch_size"] = int(args.batch_size)

    input_dim = next(iter(client_bundles.values()))["X_train"].shape[1]
    template_model = make_model(input_dim, model_config)
    initial_parameters = initialize_parameters(
        args.initialization,
        template_model,
        models_dir=models_dir,
        client_data_dir=client_data_dir,
    )
    set_model_parameters(template_model, initial_parameters)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device for clients: {device}")
    print(f"Clients: {client_names}")
    print(f"Input dimension: {input_dim}")
    print(f"Model config: {serializable_mlp_config(model_config)}")
    print(f"Initialization: {args.initialization}")
    print(f"Output directory: {output_dir}")

    def client_fn(cid: str):
        client_idx = int(cid)
        client_name = client_names[client_idx]
        return CreditFlowerClient(
            client_name=client_name,
            data_bundle=client_bundles[client_name],
            input_dim=input_dim,
            model_config=model_config,
            local_epochs=args.local_epochs,
            device=device,
        ).to_client()

    num_clients = len(client_names)
    min_fit_clients = args.min_fit_clients or num_clients
    min_evaluate_clients = args.min_evaluate_clients or num_clients

    strategy = TrackingFedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": args.client_cpus, "num_gpus": args.client_gpus},
    )

    final_parameters = strategy.latest_parameters or initial_parameters
    final_model = make_model(input_dim, model_config)
    set_model_parameters(final_model, final_parameters)

    X_val_pooled, y_val_pooled = pooled_bundle(client_bundles, "val")
    val_loader = make_loader(
        X_val_pooled,
        y_val_pooled,
        batch_size=model_config["batch_size"],
        shuffle=False,
    )
    y_val_true, y_val_prob = predict_proba(final_model, val_loader, torch.device("cpu"))
    best_threshold, best_val_f1 = best_f1_threshold(y_val_true, y_val_prob)
    best_val_auc = float(roc_auc_score(y_val_true, y_val_prob))

    X_test_pooled, y_test_pooled = pooled_bundle(client_bundles, "test")
    test_loader = make_loader(
        X_test_pooled,
        y_test_pooled,
        batch_size=model_config["batch_size"],
        shuffle=False,
    )
    y_test_true, y_test_prob = predict_proba(final_model, test_loader, torch.device("cpu"))
    pooled_test_metrics = safe_binary_metrics(y_test_true, y_test_prob, best_threshold)

    summary_df = evaluate_final_model(
        final_model,
        model_config,
        client_bundles,
        threshold=best_threshold,
        output_dir=output_dir,
    )

    artifact = {
        "model_state_dict": final_model.state_dict(),
        "input_dim": int(input_dim),
        "best_threshold": float(best_threshold),
        "feature_set": args.feature_set,
        "use_feature_engineering": args.feature_set == "engineered",
        "client_data_dir": str(client_data_dir),
        "clients": client_names,
        "model_config": serializable_mlp_config(model_config),
        "federated_config": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "fraction_fit": args.fraction_fit,
            "fraction_evaluate": args.fraction_evaluate,
            "initialization": args.initialization,
            "seed": args.seed,
        },
        "validation_metrics": {
            "best_val_auc": best_val_auc,
            "best_val_f1": float(best_val_f1),
        },
        "test_metrics": pooled_test_metrics,
        "client_summary": summary_df.to_dict(orient="records"),
        "preprocessor": preprocessor,
    }
    model_path = output_dir / "federated_mlp.pt"
    torch.save(artifact, model_path)
    print(f"Saved federated model to {model_path}")

    pooled_summary_path = output_dir / "federated_pooled_metrics.json"
    with pooled_summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "validation_metrics": artifact["validation_metrics"],
                "test_metrics": pooled_test_metrics,
                "best_threshold": float(best_threshold),
                "test_rows": int(len(y_test_true)),
                "default_rate": float(np.mean(y_test_true)),
            },
            file,
            indent=2,
        )
    print(f"Saved pooled metrics to {pooled_summary_path}")

    if args.save_history_json:
        history_path = output_dir / "flower_history.txt"
        history_path.write_text(str(history), encoding="utf-8")
        print(f"Saved Flower history to {history_path}")


if __name__ == "__main__":
    main()
