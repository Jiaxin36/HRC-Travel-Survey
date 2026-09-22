"""Configuration and validation of existing survey and LLM-response tables."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import numpy as np
import pandas as pd


# Config

def _as_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values or ())

@dataclass(frozen=True)
class ColumnConfig:
    id_col: str
    stratify_col: str
    condition_categorical: tuple[str, ...]
    condition_numeric: tuple[str, ...]
    outcome_categorical: tuple[str, ...]
    outcome_numeric: tuple[str, ...]
    metric_ordered: tuple[str, ...]
    metric_nominal: tuple[str, ...]
    ordered_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    category_levels: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], response_schema: Sequence[Mapping[str, Any]]) -> "ColumnConfig":
        categories = {item['source_variable']: tuple(item['answer_codes'].values()) for item in response_schema}
        # Derive fixed bounds from the questionnaire, never from observed sample extrema.
        ranges = {key: (float(min(categories[key])), float(max(categories[key])))
                  for key in data.get('metric_ordered', [])}
        result = cls(
            id_col=str(data["id_col"]),
            stratify_col=str(data["stratify_col"]),
            condition_categorical=_as_tuple(data.get("condition_categorical")),
            condition_numeric=_as_tuple(data.get("condition_numeric")),
            outcome_categorical=_as_tuple(data.get("outcome_categorical")),
            outcome_numeric=_as_tuple(data.get("outcome_numeric")),
            metric_ordered=_as_tuple(data.get("metric_ordered")),
            metric_nominal=_as_tuple(data.get("metric_nominal")),
            ordered_ranges=ranges,
            category_levels=categories,
        )
        result.validate()
        return result

    @property
    def condition_cols(self) -> tuple[str, ...]:
        return self.condition_categorical + self.condition_numeric

    @property
    def outcome_cols(self) -> tuple[str, ...]:
        return self.outcome_categorical + self.outcome_numeric

    @property
    def metric_cols(self) -> tuple[str, ...]:
        return self.metric_ordered + self.metric_nominal

    def validate(self) -> None:
        named_sets = {
            "condition_categorical": self.condition_categorical,
            "condition_numeric": self.condition_numeric,
            "outcome_categorical": self.outcome_categorical,
            "outcome_numeric": self.outcome_numeric,
            "metric_ordered": self.metric_ordered,
            "metric_nominal": self.metric_nominal,
        }
        for name, values in named_sets.items():
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate columns in {name}: {values}")

        if set(self.condition_categorical) & set(self.condition_numeric):
            raise ValueError("Condition categorical and numeric columns overlap.")
        if set(self.outcome_categorical) & set(self.outcome_numeric):
            raise ValueError("Outcome categorical and numeric columns overlap.")
        if set(self.metric_ordered) & set(self.metric_nominal):
            raise ValueError("Metric ordered and nominal columns overlap.")
        if set(self.metric_cols) != set(self.outcome_cols):
            raise ValueError(
                "Metric columns must describe exactly the configured outcome columns."
            )

        for column in self.metric_ordered:
            if column not in self.ordered_ranges:
                raise ValueError(f"Missing fixed range for ordered outcome {column!r}.")
            low, high = self.ordered_ranges[column]
            if not low < high:
                raise ValueError(f"Invalid ordered range for {column!r}: {(low, high)}")

        for column in self.outcome_categorical:
            if column not in self.category_levels:
                raise ValueError(
                    f"Missing fixed category support for categorical outcome {column!r}."
                )
            if len(self.category_levels[column]) < 2:
                raise ValueError(f"Outcome {column!r} needs at least two categories.")

@dataclass(frozen=True)
class TVAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 16
    learning_rate: float = 1e-3
    epochs: int = 60
    batch_size: int = 128
    kl_weight: float = 0.1
    categorical_sampling: str = "argmax"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TVAEConfig":
        result = cls(**dict(data))
        result.validate()
        return result

    def validate(self) -> None:
        if min(self.hidden_dim, self.latent_dim, self.epochs, self.batch_size) <= 0:
            raise ValueError("TVAE dimensions, epochs, and batch size must be positive.")
        if self.learning_rate <= 0 or self.kl_weight < 0:
            raise ValueError("learning_rate must be positive and kl_weight nonnegative.")
        if self.categorical_sampling not in {"argmax", "sample"}:
            raise ValueError("categorical_sampling must be 'argmax' or 'sample'.")

@dataclass(frozen=True)
class CalibrationConfig:
    gamma: float = 0.5
    candidate_pool_size: int = 128
    score_scaling: str = "rank"
    temperature: float = 1.0
    posterior_mean_for_human_score: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationConfig":
        result = cls(**dict(data))
        result.validate()
        return result

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive.")
        if self.score_scaling not in {"rank", "robust_z", "minmax"}:
            raise ValueError("Unsupported score_scaling.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

@dataclass(frozen=True)
class ProjectConfig:
    columns: ColumnConfig
    model: TVAEConfig
    calibration: CalibrationConfig
    response_schema: tuple[Mapping[str, Any], ...]
    profile_labels: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        schema = tuple(data['response_schema'])
        qids = [item['qid'] for item in schema]
        variables = [item['source_variable'] for item in schema]
        if len(set(qids)) != len(qids) or len(set(variables)) != len(variables):
            raise ValueError('Duplicate response keys or source variables')
        if set(variables) != set(data['columns']['outcome_categorical']):
            raise ValueError('Response schema must cover all categorical outcomes exactly once')
        for item in schema:
            codes = list(item['answer_codes'].values())
            if len(codes) < 2 or len(set(codes)) != len(codes) or any(type(code) is not int for code in codes):
                raise ValueError('Each response requires distinct integer codes for its options')
            if any(not isinstance(text, str) or not text.strip() for text in item['answer_codes']):
                raise ValueError('Answer options must be nonempty strings')
        return cls(
            columns=ColumnConfig.from_dict(data["columns"], schema),
            model=TVAEConfig.from_dict(data["model"]),
            calibration=CalibrationConfig.from_dict(data["calibration"]),
            response_schema=schema,
            profile_labels=data['profile_labels'],
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Data

ROOT = Path(__file__).resolve().parent

FAMILIES = ('DeepSeek', 'Qwen', 'GPT4', 'GPT5')

def validate_outcomes(frame, config, name):
    for field in config.columns.outcome_cols:
        if field not in frame:
            raise ValueError(f'{name}: missing outcome {field}')
        allowed = config.columns.category_levels.get(field)
        if allowed and not frame[field].isin(allowed).all():
            raise ValueError(f'{name}: invalid outcome code in {field}')

def validate_human(frame, config, name, *, require_stratum=False):
    c = config.columns
    fields = list(dict.fromkeys([c.id_col, *c.condition_cols, *c.outcome_cols,
                               *([c.stratify_col] if require_stratum else [])]))
    if set(fields)-set(frame):
        raise ValueError(f'{name}: required analysis fields are missing')
    if frame.empty or frame[fields].isna().any().any():
        raise ValueError(f'{name}: analysis fields must be complete and nonempty')
    if frame[c.id_col].duplicated().any() or frame[c.id_col].astype(str).str.strip().eq('').any():
        raise ValueError(f'{name}: IDs must be unique and nonempty')
    numeric = frame[fields].select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f'{name}: numeric fields must be finite')
    for field in c.condition_numeric:
        if not np.isfinite(pd.to_numeric(frame[field], errors='raise').to_numpy()).all():
            raise ValueError(f'{name}: invalid numeric condition')
    validate_outcomes(frame, config, name)
    return fields

def prepare(country, pool_path, benchmark_path, output, *, llm_dir=None):
    config = ProjectConfig.load(ROOT/f'configs/{country}.json')
    c = config.columns
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new private output directory')
    pool = pd.read_csv(pool_path, dtype={c.id_col: str})
    benchmark = pd.read_csv(benchmark_path, dtype={c.id_col: str})
    pool_fields = validate_human(pool, config, 'calibration pool', require_stratum=True)
    benchmark_fields = validate_human(benchmark, config, 'benchmark')
    if set(pool[c.id_col]) & set(benchmark[c.id_col]):
        raise ValueError('Calibration/benchmark ID overlap')
    responses = {}
    hashes = {'calibration_pool': hashlib.sha256(Path(pool_path).read_bytes()).hexdigest(),
              'benchmark': hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest()}
    if llm_dir is not None:
        for family in FAMILIES:
            path = Path(llm_dir)/f'{family}.csv'
            frame = pd.read_csv(path, dtype={c.id_col: str})
            if c.id_col not in frame or frame[c.id_col].duplicated().any() or set(frame[c.id_col]) != set(benchmark[c.id_col]):
                raise ValueError(f'{family}: response IDs must exactly match benchmark IDs')
            validate_outcomes(frame, config, family)
            responses[family] = frame[[c.id_col, *c.outcome_cols]]
            hashes[family] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    (output/'llm').mkdir()
    pool[pool_fields].to_csv(output/'calibration_pool.csv', index=False)
    benchmark[benchmark_fields].to_csv(output/'benchmark.csv', index=False)
    for family, frame in responses.items():
        frame.to_csv(output/f'llm/{family}.csv', index=False)
    manifest = {'country': country, 'calibration_pool_n': len(pool), 'benchmark_n': len(benchmark),
                'llm_families_imported': list(responses), 'source_sha256': hashes,
                'split_preserved': True, 'household_separation_requires_upstream_validation': country == 'england'}
    (output/'preparation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate existing processed tables without changing their split.')
    parser.add_argument('--country', choices=['china', 'england'], required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--llm-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.country, args.pool, args.benchmark, args.output, llm_dir=args.llm_dir)
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}))
