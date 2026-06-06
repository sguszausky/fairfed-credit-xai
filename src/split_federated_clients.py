"""Split the credit dataset into random IID-style federated client pools.

The splitter preserves the global distribution of one or more stratification
columns as closely as possible by shuffling rows inside each stratum and then
spreading every stratum across all clients. Optional client sizes can be used
to create exact unequal client row counts while preserving the stratification
distribution as closely as possible.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path


DEFAULT_INPUT = Path("data/default_of_credit_card_clients.xls")
DEFAULT_OUTPUT_ROOT = Path("data")
DEFAULT_DISTRIBUTION_TYPE = "IID"
DEFAULT_TARGET_COL = "default payment next month"

pd = None


def require_pandas():
    global pd
    if pd is None:
        import pandas as pandas

        pd = pandas
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create random stratified federated client datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the full dataset. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Parent directory where split folders are written. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--distribution-type",
        default=DEFAULT_DISTRIBUTION_TYPE,
        help=(
            "Distribution label used in the output folder name. "
            f"Default: {DEFAULT_DISTRIBUTION_TYPE}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact output directory. Overrides the default "
            "<output-root>/<distribution-type>_cl<num-clients> path."
        ),
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of federated client datasets to create.",
    )
    parser.add_argument(
        "--client-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Exact row counts for each client. Must contain --num-clients "
            "positive integers and sum to at most the number of rows in the dataset. "
            "Defaults to an equal split over all rows."
        ),
    )
    parser.add_argument(
        "--unbalance-feature",
        default=None,
        help=(
            "Feature/column to force into different per-client distributions. "
            "Use together with --unbalance-targets."
        ),
    )
    parser.add_argument(
        "--unbalance-targets",
        default=None,
        help=(
            "Python/JSON-style list with one [value_or_range, percent] entry per "
            "client, for example: '[[(18, 25), 80], [(50, 70), 90]]'. "
            "Numeric ranges are inclusive; non-ranges match exact values."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible shuffling.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help=f"Target column used in summary statistics. Default: {DEFAULT_TARGET_COL}",
    )
    parser.add_argument(
        "--stratify-cols",
        nargs="+",
        default=None,
        help=(
            "Columns whose joint distribution should be preserved. "
            "Defaults to --target-col."
        ),
    )
    parser.add_argument(
        "--excel-header",
        type=int,
        default=1,
        help="Header row for Excel input files. Use 1 for this UCI credit dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.output_root / f"{args.distribution_type}_cl{args.num_clients}"


def read_dataset(path: Path, excel_header: int) -> pd.DataFrame:
    pandas = require_pandas()
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pandas.read_excel(path, header=excel_header)
    if suffix == ".csv":
        return pandas.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def validate_args(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[str]:
    if args.num_clients < 2:
        raise ValueError("--num-clients must be at least 2.")
    if args.num_clients > len(df):
        raise ValueError("--num-clients cannot be larger than the number of rows.")
    if args.client_sizes is not None:
        if len(args.client_sizes) != args.num_clients:
            raise ValueError(
                "--client-sizes must contain exactly --num-clients values."
            )
        if any(size <= 0 for size in args.client_sizes):
            raise ValueError("--client-sizes values must be positive integers.")
        if sum(args.client_sizes) > len(df):
            raise ValueError(
                "--client-sizes must sum to at most the number of rows in the dataset "
                f"({len(df)})."
            )

    stratify_cols = args.stratify_cols or [args.target_col]
    missing_cols = [col for col in [args.target_col, *stratify_cols] if col not in df]
    if missing_cols:
        raise ValueError(f"Missing column(s) in dataset: {missing_cols}")

    validate_unbalance_args(df, args)

    validate_output_dir(output_dir, args.overwrite)

    return stratify_cols


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. "
            "Use --overwrite or choose another --output-dir."
        )


def validate_unbalance_args(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.unbalance_feature is None and args.unbalance_targets is None:
        args.unbalance_specs = None
        args.unbalance_feature_resolved = None
        return
    if args.unbalance_feature is None or args.unbalance_targets is None:
        raise ValueError(
            "--unbalance-feature and --unbalance-targets must be used together."
        )

    feature = resolve_column_name(df, args.unbalance_feature)
    client_sizes = args.client_sizes or equal_client_sizes(len(df), args.num_clients)
    specs = parse_unbalance_targets(
        raw_targets=args.unbalance_targets,
        num_clients=args.num_clients,
        client_sizes=client_sizes,
    )
    for spec in specs:
        matching_rows = count_matching_rows(df, feature, spec["value"])
        required_rows = spec["required_rows"]
        if matching_rows < required_rows:
            raise ValueError(
                f"Not enough data points with {feature}={format_target_value(spec['value'])}: "
                f"need {required_rows}, available {matching_rows}."
            )

    args.unbalance_specs = specs
    args.unbalance_feature_resolved = feature


def resolve_column_name(df: pd.DataFrame, requested_col: str) -> str:
    if requested_col in df:
        return requested_col

    requested_lower = requested_col.lower()
    matches = [col for col in df.columns if str(col).lower() == requested_lower]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Missing column in dataset: {requested_col}")


def parse_unbalance_targets(
    raw_targets: object,
    num_clients: int,
    client_sizes: list[int],
) -> list[dict[str, object]]:
    if isinstance(raw_targets, str):
        try:
            targets = ast.literal_eval(raw_targets)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "--unbalance-targets must be a Python/JSON-style list, for example "
                "'[[(18, 25), 80], [(50, 70), 90]]'."
            ) from exc
    else:
        targets = raw_targets

    if not isinstance(targets, (list, tuple)):
        raise ValueError("--unbalance-targets must be a list.")
    if len(targets) != num_clients:
        raise ValueError(
            "--unbalance-targets must contain exactly --num-clients entries."
        )

    specs = []
    for client_id, item in enumerate(targets):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "Each --unbalance-targets entry must be [value_or_range, percent]."
            )
        value, percent = item
        if not isinstance(percent, (int, float)):
            raise ValueError("Unbalance percentages must be numeric.")
        if percent < 0 or percent > 100:
            raise ValueError("Unbalance percentages must be between 0 and 100.")

        required_rows = round(client_sizes[client_id] * percent / 100)
        specs.append(
            {
                "client_id": client_id,
                "value": normalize_target_value(value),
                "percent": float(percent),
                "required_rows": int(required_rows),
                "outside_rows": int(client_sizes[client_id] - required_rows),
            }
        )

    return specs


def normalize_target_value(value: object) -> object:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        lower, upper = value
        if lower > upper:
            raise ValueError(f"Invalid numeric range {value}: lower bound is larger.")
        return (lower, upper)

    if isinstance(value, str) and "-" in value:
        lower_text, upper_text = value.split("-", 1)
        try:
            lower = float(lower_text.strip())
            upper = float(upper_text.strip())
        except ValueError:
            return value
        if lower > upper:
            raise ValueError(f"Invalid numeric range {value}: lower bound is larger.")
        return (lower, upper)

    return value


def count_matching_rows(df: pd.DataFrame, feature: str, value: object) -> int:
    return int(feature_mask(df, feature, value).sum())


def create_federated_clients(
    df: pd.DataFrame,
    num_clients: int = 5,
    client_sizes: list[int] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
    stratify_cols: list[str] | None = None,
    seed: int = 42,
    unbalance_feature: str | None = None,
    unbalance_targets: object | None = None,
    output_dir: Path | str | None = None,
    distribution_type: str = DEFAULT_DISTRIBUTION_TYPE,
    source_path: Path | str | None = None,
    overwrite: bool = False,
) -> list[pd.DataFrame]:
    """Create federated client datasets from an in-memory DataFrame.

    This is the notebook-friendly API. Pass ``unbalance_targets`` as a normal
    Python list, for example ``[((18, 25), 80), ((50, 70), 90)]``.
    """
    validate_split_sizes(df, num_clients, client_sizes)
    resolved_stratify_cols = stratify_cols or [target_col]
    validate_columns(df, target_col, resolved_stratify_cols)

    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    if resolved_output_dir is not None:
        validate_output_dir(resolved_output_dir, overwrite)

    resolved_feature = None
    specs = None
    if unbalance_feature is None and unbalance_targets is None:
        clients = split_stratified(
            df=df,
            num_clients=num_clients,
            stratify_cols=resolved_stratify_cols,
            seed=seed,
            client_sizes=client_sizes,
        )
    elif unbalance_feature is None or unbalance_targets is None:
        raise ValueError(
            "unbalance_feature and unbalance_targets must be used together."
        )
    else:
        resolved_feature = resolve_column_name(df, unbalance_feature)
        resolved_client_sizes = client_sizes or equal_client_sizes(len(df), num_clients)
        specs = parse_unbalance_targets(
            raw_targets=unbalance_targets,
            num_clients=num_clients,
            client_sizes=resolved_client_sizes,
        )
        validate_unbalance_specs(df, resolved_feature, specs)

        clients = split_feature_unbalanced(
            df=df,
            feature=resolved_feature,
            specs=specs,
            seed=seed,
        )

    if resolved_output_dir is not None:
        write_clients(
            clients=clients,
            source_path=Path(source_path) if source_path is not None else None,
            output_dir=resolved_output_dir,
            distribution_type=distribution_type,
            target_col=target_col,
            stratify_cols=resolved_stratify_cols,
            seed=seed,
            unbalance_feature=resolved_feature,
            unbalance_specs=specs,
        )

    return clients


def validate_split_sizes(
    df: pd.DataFrame,
    num_clients: int,
    client_sizes: list[int] | None,
) -> None:
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")
    if num_clients > len(df):
        raise ValueError("num_clients cannot be larger than the number of rows.")
    if client_sizes is None:
        return
    if len(client_sizes) != num_clients:
        raise ValueError("client_sizes must contain exactly num_clients values.")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("client_sizes values must be positive integers.")
    if sum(client_sizes) > len(df):
        raise ValueError(
            "client_sizes must sum to at most the number of rows in the dataset "
            f"({len(df)})."
        )


def validate_columns(
    df: pd.DataFrame,
    target_col: str,
    stratify_cols: list[str],
) -> None:
    missing_cols = [col for col in [target_col, *stratify_cols] if col not in df]
    if missing_cols:
        raise ValueError(f"Missing column(s) in dataset: {missing_cols}")


def validate_unbalance_specs(
    df: pd.DataFrame,
    feature: str,
    specs: list[dict[str, object]],
) -> None:
    for spec in specs:
        matching_rows = count_matching_rows(df, feature, spec["value"])
        required_rows = spec["required_rows"]
        if matching_rows < required_rows:
            raise ValueError(
                f"Not enough data points with {feature}={format_target_value(spec['value'])}: "
                f"need {required_rows}, available {matching_rows}."
            )


def split_stratified(
    df: pd.DataFrame,
    num_clients: int,
    stratify_cols: list[str],
    seed: int,
    client_sizes: list[int] | None = None,
) -> list[pd.DataFrame]:
    rng = random.Random(seed)
    client_parts: list[list[pd.DataFrame]] = [[] for _ in range(num_clients)]

    grouped = list(df.groupby(stratify_cols, sort=False, dropna=False))
    stratum_sizes = [len(stratum) for _, stratum in grouped]
    if client_sizes is None:
        client_sizes = equal_client_sizes(len(df), num_clients)
    rows_to_assign = sum(client_sizes)
    stratum_client_counts = allocate_strata_to_clients(
        stratum_sizes=stratum_sizes,
        client_sizes=client_sizes,
        rows_to_assign=rows_to_assign,
        rng=rng,
    )

    for (_, stratum), client_counts in zip(grouped, stratum_client_counts):
        start = 0
        shuffled_indices = list(stratum.index)
        rng.shuffle(shuffled_indices)
        shuffled = stratum.loc[shuffled_indices]

        for client_id, part_size in enumerate(client_counts):
            part = shuffled.iloc[start : start + part_size]
            start += part_size
            if not part.empty:
                client_parts[client_id].append(part)

    clients = []
    for parts in client_parts:
        if parts:
            client_df = pd.concat(parts, axis=0)
            shuffled_indices = list(client_df.index)
            rng.shuffle(shuffled_indices)
            client_df = client_df.loc[shuffled_indices]
        else:
            client_df = df.iloc[0:0]
        clients.append(client_df.reset_index(drop=True))

    return clients


def split_feature_unbalanced(
    df: pd.DataFrame,
    feature: str,
    specs: list[dict[str, object]],
    seed: int,
) -> list[pd.DataFrame]:
    rng = random.Random(seed)
    remaining_indices = set(df.index)
    selected_by_client: list[list[object]] = [[] for _ in specs]

    required_order = sorted(
        range(len(specs)),
        key=lambda client_id: count_matching_rows(df, feature, specs[client_id]["value"]),
    )
    for client_id in required_order:
        spec = specs[client_id]
        required_rows = int(spec["required_rows"])
        if required_rows == 0:
            continue

        matching_indices = matching_remaining_indices(
            df=df,
            feature=feature,
            value=spec["value"],
            remaining_indices=remaining_indices,
        )
        if len(matching_indices) < required_rows:
            raise ValueError(
                f"Not enough data points with {feature}={format_target_value(spec['value'])}: "
                f"need {required_rows}, available {len(matching_indices)}."
            )

        selected_indices = rng.sample(matching_indices, required_rows)
        selected_by_client[client_id].extend(selected_indices)
        remaining_indices.difference_update(selected_indices)

    for client_id, spec in enumerate(specs):
        outside_rows = int(spec["outside_rows"])
        if outside_rows == 0:
            continue

        outside_indices = outside_remaining_indices(
            df=df,
            feature=feature,
            value=spec["value"],
            remaining_indices=remaining_indices,
        )
        if len(outside_indices) < outside_rows:
            raise ValueError(
                f"Not enough data points outside {feature}={format_target_value(spec['value'])}: "
                f"need {outside_rows}, available {len(outside_indices)}."
            )

        selected_indices = rng.sample(outside_indices, outside_rows)
        selected_by_client[client_id].extend(selected_indices)
        remaining_indices.difference_update(selected_indices)

    clients = []
    for client_indices in selected_by_client:
        rng.shuffle(client_indices)
        clients.append(df.loc[client_indices].reset_index(drop=True))

    return clients


def matching_remaining_indices(
    df: pd.DataFrame,
    feature: str,
    value: object,
    remaining_indices: set[object],
) -> list[object]:
    mask = feature_mask(df.loc[list(remaining_indices)], feature, value)
    return list(mask[mask].index)


def outside_remaining_indices(
    df: pd.DataFrame,
    feature: str,
    value: object,
    remaining_indices: set[object],
) -> list[object]:
    mask = feature_mask(df.loc[list(remaining_indices)], feature, value)
    return list(mask[~mask].index)


def feature_mask(df: pd.DataFrame, feature: str, value: object):
    series = df[feature]
    if is_numeric_range(value):
        lower, upper = value
        return series.between(lower, upper, inclusive="both")
    return series == value


def is_numeric_range(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def format_target_value(value: object) -> str:
    if is_numeric_range(value):
        lower, upper = value
        return f"{lower}-{upper}"
    return str(value)


def equal_client_sizes(total_rows: int, num_clients: int) -> list[int]:
    base_size, remainder = divmod(total_rows, num_clients)
    return [base_size + int(client_id < remainder) for client_id in range(num_clients)]


def allocate_strata_to_clients(
    stratum_sizes: list[int],
    client_sizes: list[int],
    rows_to_assign: int,
    rng: random.Random,
) -> list[list[int]]:
    total_rows = sum(stratum_sizes)
    remaining_client_sizes = client_sizes.copy()
    selected_stratum_sizes = allocate_sampled_stratum_sizes(
        stratum_sizes=stratum_sizes,
        rows_to_assign=rows_to_assign,
        total_rows=total_rows,
        rng=rng,
    )
    allocations: list[list[int]] = []
    selected_after_current = rows_to_assign

    for stratum_size, selected_stratum_size in zip(
        stratum_sizes, selected_stratum_sizes
    ):
        selected_after_current -= selected_stratum_size
        counts = allocate_single_stratum(
            stratum_size=selected_stratum_size,
            client_sizes=client_sizes,
            remaining_client_sizes=remaining_client_sizes,
            rows_after_current=selected_after_current,
            selected_total_rows=rows_to_assign,
            rng=rng,
        )
        allocations.append(counts)
        remaining_client_sizes = [
            remaining - count
            for remaining, count in zip(remaining_client_sizes, counts)
        ]

    return allocations


def allocate_sampled_stratum_sizes(
    stratum_sizes: list[int],
    rows_to_assign: int,
    total_rows: int,
    rng: random.Random,
) -> list[int]:
    sampled_sizes = [0 for _ in stratum_sizes]
    remaining_rows = rows_to_assign

    while remaining_rows > 0:
        best_stratum = max(
            (
                stratum_id
                for stratum_id, stratum_size in enumerate(stratum_sizes)
                if sampled_sizes[stratum_id] < stratum_size
            ),
            key=lambda stratum_id: (
                (rows_to_assign * stratum_sizes[stratum_id] / total_rows)
                - sampled_sizes[stratum_id],
                rng.random(),
            ),
        )
        sampled_sizes[best_stratum] += 1
        remaining_rows -= 1

    return sampled_sizes


def allocate_single_stratum(
    stratum_size: int,
    client_sizes: list[int],
    remaining_client_sizes: list[int],
    rows_after_current: int,
    selected_total_rows: int,
    rng: random.Random,
) -> list[int]:
    lower_bounds = [
        max(0, remaining - rows_after_current)
        for remaining in remaining_client_sizes
    ]
    upper_bounds = [min(remaining, stratum_size) for remaining in remaining_client_sizes]
    counts = lower_bounds.copy()
    remaining_in_stratum = stratum_size - sum(counts)

    while remaining_in_stratum > 0:
        best_client = max(
            (
                client_id
                for client_id in range(len(client_sizes))
                if counts[client_id] < upper_bounds[client_id]
            ),
            key=lambda client_id: (
                (stratum_size * client_sizes[client_id] / selected_total_rows)
                - counts[client_id],
                rng.random(),
            ),
        )
        counts[best_client] += 1
        remaining_in_stratum -= 1

    return counts


def distribution_table(df: pd.DataFrame, cols: list[str]) -> list[dict[str, object]]:
    counts = (
        df.groupby(cols, sort=True, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["share"] = counts["count"] / len(df)
    return counts.to_dict(orient="records")


def summarize_client(
    client_id: int,
    client_df: pd.DataFrame,
    target_col: str,
    stratify_cols: list[str],
) -> dict[str, object]:
    target_counts = client_df[target_col].value_counts(dropna=False).sort_index()
    summary: dict[str, object] = {
        "client_id": client_id,
        "rows": int(len(client_df)),
        "target_mean": float(client_df[target_col].mean()),
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "stratify_distribution": distribution_table(client_df, stratify_cols),
    }
    return summary


def write_clients(
    clients: list[pd.DataFrame],
    source_path: Path | None,
    output_dir: Path,
    distribution_type: str,
    target_col: str,
    stratify_cols: list[str],
    seed: int,
    unbalance_feature: str | None = None,
    unbalance_specs: list[dict[str, object]] | None = None,
) -> None:
    pandas = require_pandas()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for client_id, client_df in enumerate(clients):
        client_dir = output_dir / f"client_{client_id:03d}"
        client_dir.mkdir(parents=True, exist_ok=True)
        client_path = client_dir / "client.csv"
        client_df.to_csv(client_path, index=False)
        summary = summarize_client(client_id, client_df, target_col, stratify_cols)
        summary["path"] = str(client_path.relative_to(output_dir))
        summaries.append(summary)

    summary_df = pandas.DataFrame(
        {
            "client_id": item["client_id"],
            "rows": item["rows"],
            "target_mean": item["target_mean"],
            "path": item["path"],
        }
        for item in summaries
    )
    summary_df.to_csv(output_dir / "client_summary.csv", index=False)

    manifest = {
        "source_path": str(source_path) if source_path is not None else None,
        "distribution_type": distribution_type,
        "output_dir": str(output_dir),
        "num_clients": len(clients),
        "client_sizes": [int(len(client_df)) for client_df in clients],
        "seed": seed,
        "target_col": target_col,
        "stratify_cols": stratify_cols,
        "unbalance_feature": unbalance_feature,
        "unbalance_targets": serialize_unbalance_specs(unbalance_specs),
        "total_rows": int(sum(len(client_df) for client_df in clients)),
        "global_target_mean": float(
            pandas.concat(clients, axis=0, ignore_index=True)[target_col].mean()
        ),
        "clients": summaries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    write_distribution_command(
        output_dir=output_dir,
        target_col=target_col,
        stratify_cols=stratify_cols,
        unbalance_feature=unbalance_feature,
    )


def write_distribution_command(
    output_dir: Path,
    target_col: str,
    stratify_cols: list[str],
    unbalance_feature: str | None,
) -> None:
    columns = []
    for column in [target_col, *stratify_cols, unbalance_feature]:
        if column is not None and column not in columns:
            columns.append(column)

    command = build_distribution_command(output_dir, columns)
    text = (
        "Run this from the project root to inspect the generated client distributions:\n\n"
        f"{command}\n"
    )
    (output_dir / "show_distributions.txt").write_text(text, encoding="utf-8")


def build_distribution_command(output_dir: Path, columns: list[str]) -> str:
    code = (
        "from pathlib import Path; "
        "import pandas as pd; "
        f"out=Path({str(output_dir)!r}); "
        f"cols={columns!r}; "
        "print(pd.read_csv(out/'client_summary.csv').to_string(index=False)); "
        "print(); "
        "files=sorted(out.glob('client_*/client.csv')); "
        "[print(p.parent.name, '\\n', "
        "pd.read_csv(p)[[c for c in cols if c in pd.read_csv(p).columns]]"
        ".value_counts(normalize=True).rename('share').reset_index().to_string(index=False), "
        "'\\n') for p in files]"
    )
    return f"python3 -c {code!r}"


def serialize_unbalance_specs(
    specs: list[dict[str, object]] | None,
) -> list[dict[str, object]] | None:
    if specs is None:
        return None

    serialized = []
    for spec in specs:
        serialized.append(
            {
                "client_id": spec["client_id"],
                "value": spec["value"],
                "percent": spec["percent"],
                "required_rows": spec["required_rows"],
                "outside_rows": spec["outside_rows"],
            }
        )
    return serialized


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    df = read_dataset(args.input, args.excel_header)
    stratify_cols = validate_args(df, args, output_dir)

    if args.unbalance_specs is not None:
        clients = split_feature_unbalanced(
            df=df,
            feature=args.unbalance_feature_resolved,
            specs=args.unbalance_specs,
            seed=args.seed,
        )
    else:
        clients = split_stratified(
            df=df,
            num_clients=args.num_clients,
            stratify_cols=stratify_cols,
            seed=args.seed,
            client_sizes=args.client_sizes,
        )
    write_clients(
        clients=clients,
        source_path=args.input,
        output_dir=output_dir,
        distribution_type=args.distribution_type,
        target_col=args.target_col,
        stratify_cols=stratify_cols,
        seed=args.seed,
        unbalance_feature=args.unbalance_feature_resolved,
        unbalance_specs=args.unbalance_specs,
    )

    print(f"Wrote {len(clients)} client datasets to {output_dir}")
    print(f"Rows per client: {[len(client_df) for client_df in clients]}")
    print(f"Global {args.target_col!r} mean: {df[args.target_col].mean():.4f}")


if __name__ == "__main__":
    main()
