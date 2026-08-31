"""CEL dataset-provider adapter.

CEL-native ``MethodDataset`` and dataframe codecs remain on ``CelDatasetAdapter``.
The portable ``PreparedDataset`` returned by ``prepare()`` contains neither.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import (
    DatasetProvenance,
    FeatureDomains,
    FeatureSchema,
    PreparedDataset,
)
from experiments.zeroshot_cf.datasets.base import DatasetSpec
from experiments.zeroshot_cf.vendor_setup import CEL_REPO, PINNED_CEL_REVISION

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = (
        contiguous.dtype.str.encode()
        + str(contiguous.shape).encode()
        + contiguous.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _schema_identity(schema: FeatureSchema) -> dict[str, Any]:
    """Return the complete canonical schema payload used by provenance."""
    return {
        "names": list(schema.names),
        "numerical": list(schema.numerical),
        "categorical_groups": [
            {"name": group.name, "columns": list(group.columns)}
            for group in schema.categorical_groups
        ],
        "actionable_scalars": list(schema.actionable_scalars),
        "actionable_groups": [
            {"name": group.name, "columns": list(group.columns)}
            for group in schema.actionable_groups
        ],
        "immutable": list(schema.immutable),
        "domains": {
            "lower": _sha256_array(schema.domains.lower),
            "upper": _sha256_array(schema.domains.upper),
            "discrete": {
                str(column): _sha256_array(values)
                for column, values in sorted(schema.domains.discrete.items())
            },
        },
    }


def _dataset_fingerprint(
    *,
    name: str,
    source_revision: str,
    source_hashes: dict[str, str],
    preprocessing_id: str,
    split_id: str,
    split_seed: int,
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
) -> str:
    """Hash every prepared input and the complete portable output contract."""
    identity = {
        "name": name,
        "provider": CelDatasetProvider.provider_id,
        "revision": source_revision,
        "source_hashes": dict(sorted(source_hashes.items())),
        "preprocessing_id": preprocessing_id,
        "split_id": split_id,
        "split_seed": split_seed,
        "arrays": {key: _sha256_array(value) for key, value in sorted(arrays.items())},
        "schema": _schema_identity(schema),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CelDatasetAdapter:
    """Focused compatibility access to CEL-native transforms and metadata."""

    prepared: PreparedDataset
    method_dataset: Any
    numerical_features_indices: tuple[int, ...]
    categorical_features_indices: tuple[int, ...]
    split_variant: str
    n_dropped_rows: int
    preprocessing_variant: str

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.method_dataset.inverse_transform(X)


class CelDatasetProvider:
    """Prepare the pinned CEL classification datasets."""

    provider_id = "cel"

    def prepare(self, spec: DatasetSpec) -> PreparedDataset:
        return self.prepare_adapter(spec).prepared

    def prepare_adapter(self, spec: DatasetSpec) -> CelDatasetAdapter:
        # Kept lazy so importing portable contracts never imports CEL.
        from cel.datasets.file_dataset import FileDataset
        from cel.datasets.method_dataset import MethodDataset
        from cel.preprocessing.base import PreprocessingContext
        from cel.preprocessing.pipeline import PreprocessingPipeline
        from cel.preprocessing.scalers import MinMaxScalingStep
        from sklearn.model_selection import train_test_split

        config_path = CEL_REPO / "config" / "datasets" / f"{spec.name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        config = yaml.safe_load(config_path.read_text())
        raw_data_path = CEL_REPO / config["raw_data_path"]
        file_dataset = FileDataset(config_path=config_path)
        n_dropped_rows = 0
        preprocessing_variant = "original"
        if spec.name == "heloc" and spec.drop_heloc_all_minus9:
            all_minus9 = np.all(np.asarray(file_dataset.X) == -9, axis=1)
            n_dropped_rows = int(all_minus9.sum())
            keep = ~all_minus9
            file_dataset.X = file_dataset.X[keep]
            file_dataset.y = file_dataset.y[keep]
            file_dataset.raw_data = file_dataset.raw_data.loc[keep].reset_index(
                drop=True
            )
            preprocessing_variant = "drop_heloc_all_minus9"

        preprocessing = PreprocessingPipeline([("minmax", MinMaxScalingStep())])
        method_dataset = MethodDataset(
            file_dataset,
            preprocessing_pipeline=preprocessing
            if spec.validation_fraction == 0
            else None,
        )
        X_validation: np.ndarray
        y_validation: np.ndarray
        split_variant = "train_test_80_20"
        if spec.validation_fraction > 0:
            X_train_raw, X_validation_raw, y_train, y_validation = train_test_split(
                method_dataset.X_train_raw,
                method_dataset.y_train,
                test_size=spec.validation_fraction,
                random_state=spec.split_seed,
                stratify=method_dataset.y_train,
            )
            n_validation = len(X_validation_raw)
            evaluation_raw = np.concatenate(
                [X_validation_raw, method_dataset.X_test_raw]
            )
            evaluation_y = np.concatenate([y_validation, method_dataset.y_test])
            context = PreprocessingContext(
                X_train=X_train_raw,
                X_test=evaluation_raw,
                y_train=y_train,
                y_test=evaluation_y,
                categorical_indices=file_dataset.categorical_features_indices,
                continuous_indices=file_dataset.numerical_features_indices,
            )
            preprocessing.fit(context)
            transformed = preprocessing.transform(context)
            X_train = transformed.X_train.astype(np.float64)
            X_validation = transformed.X_test[:n_validation].astype(np.float64)
            X_test = transformed.X_test[n_validation:].astype(np.float64)
            y_train = np.asarray(y_train, dtype=np.int64)
            y_validation = np.asarray(y_validation, dtype=np.int64)
            y_test = method_dataset.y_test.astype(np.int64)

            method_dataset.preprocessing_pipeline = preprocessing
            method_dataset.X_train_raw = X_train_raw.copy()
            method_dataset.X_test_raw = method_dataset.X_test_raw.copy()
            method_dataset.X_train = X_train
            method_dataset.X_test = X_test
            method_dataset.y_train = y_train
            method_dataset.y_test = y_test
            split_variant = (
                f"train_val_test_{0.8 * (1 - spec.validation_fraction):.2f}_"
                f"{0.8 * spec.validation_fraction:.2f}_0.20"
            )
        else:
            X_train = method_dataset.X_train.astype(np.float64)
            X_validation = np.empty((0, X_train.shape[1]), dtype=np.float64)
            X_test = method_dataset.X_test.astype(np.float64)
            y_train = method_dataset.y_train.astype(np.int64)
            y_validation = np.empty(0, dtype=np.int64)
            y_test = method_dataset.y_test.astype(np.int64)

        feature_names = tuple(method_dataset.features)
        feature_to_index = {name: index for index, name in enumerate(feature_names)}
        raw_groups = getattr(file_dataset, "one_hot_feature_groups", {})
        categorical_groups = tuple(
            OneHotActionGroup(
                group_name,
                tuple(feature_to_index[name] for name in member_names),
            )
            for group_name, member_names in raw_groups.items()
        )
        declared_actionable = set(method_dataset.actionable_features)
        grouped_columns = {
            column for group in categorical_groups for column in group.columns
        }
        ordered_immutable: tuple[int, ...] | None = None
        actionable_groups = tuple(
            group
            for group in categorical_groups
            if all(
                feature_names[column] in declared_actionable for column in group.columns
            )
        )
        if categorical_groups:
            actionable_scalars = tuple(
                index
                for index, name in enumerate(feature_names)
                if index not in grouped_columns and name in declared_actionable
            )
        elif spec.name == "heloc":
            actionability = yaml.safe_load(
                (CONFIGS_DIR / "heloc_actionability.yaml").read_text()
            )
            immutable_feature_names = tuple(actionability["immutable_features"])
            immutable_names = set(immutable_feature_names)
            ordered_immutable = tuple(
                feature_names.index(name) for name in immutable_feature_names
            )
            actionable_scalars = tuple(
                index
                for index, name in enumerate(feature_names)
                if name not in immutable_names
            )
        else:
            actionable_scalars = tuple(range(len(feature_names)))
        actionable_columns = set(actionable_scalars)
        actionable_columns.update(
            column for group in actionable_groups for column in group.columns
        )
        immutable = ordered_immutable or tuple(
            index
            for index in range(len(feature_names))
            if index not in actionable_columns
        )

        lower = np.nanmin(X_train, axis=0)
        upper = np.nanmax(X_train, axis=0)
        discrete = {
            index: values
            for index in range(X_train.shape[1])
            if 0
            < len(values := np.unique(X_train[:, index][~np.isnan(X_train[:, index])]))
            <= 20
        }
        domains = FeatureDomains(lower=lower, upper=upper, discrete=discrete)
        schema = FeatureSchema(
            names=feature_names,
            numerical=tuple(
                int(index) for index in method_dataset.numerical_features_indices
            ),
            categorical_groups=categorical_groups,
            actionable_scalars=actionable_scalars,
            actionable_groups=actionable_groups,
            immutable=immutable,
            domains=domains,
        )
        source_hashes = {
            "config": _sha256_file(config_path),
            "raw_data": _sha256_file(raw_data_path),
        }
        if spec.name == "heloc":
            source_hashes["actionability"] = _sha256_file(
                CONFIGS_DIR / "heloc_actionability.yaml"
            )
        preprocessing_id = f"minmax_train_only:{preprocessing_variant}"
        split_id = f"{split_variant}:seed={spec.split_seed}"
        fingerprint = _dataset_fingerprint(
            name=spec.name,
            source_revision=PINNED_CEL_REVISION,
            source_hashes=source_hashes,
            preprocessing_id=preprocessing_id,
            split_id=split_id,
            split_seed=spec.split_seed,
            arrays={
                "X_train": X_train,
                "y_train": y_train,
                "X_validation": X_validation,
                "y_validation": y_validation,
                "X_test": X_test,
                "y_test": y_test,
            },
            schema=schema,
        )
        provenance = DatasetProvenance(
            provider=self.provider_id,
            source_revision=PINNED_CEL_REVISION,
            source_hashes=source_hashes,
            preprocessing_id=preprocessing_id,
            split_id=split_id,
            fingerprint=fingerprint,
        )
        prepared = PreparedDataset(
            name=spec.name,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            X_test=X_test,
            y_test=y_test,
            schema=schema,
            provenance=provenance,
        )
        return CelDatasetAdapter(
            prepared=prepared,
            method_dataset=method_dataset,
            numerical_features_indices=tuple(method_dataset.numerical_features_indices),
            categorical_features_indices=tuple(
                method_dataset.categorical_features_indices
            ),
            split_variant=split_variant,
            n_dropped_rows=n_dropped_rows,
            preprocessing_variant=preprocessing_variant,
        )
