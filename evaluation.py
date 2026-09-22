"""Comparison methods, evaluation metrics and nested sampling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from hrc import CandidatePool, SeedPlan
from data_utils import ProjectConfig


# Metrics

@dataclass(frozen=True)
class MixedTypeSchema:
    ordered: tuple[str, ...]
    nominal: tuple[str, ...]
    ordered_ranges: Mapping[str, tuple[float, float]]
    nominal_categories: Mapping[str, tuple[Any, ...]]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.ordered + self.nominal

class MixedTypeEmbedding:
    """One fit, many transforms for mixed-type distributional metrics."""

    def __init__(self, schema: MixedTypeSchema) -> None:
        self.schema = schema
        self._ranges: dict[str, tuple[float, float]] = {}
        self._categories: dict[str, tuple[Any, ...]] = {}
        self._fitted = False

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.schema.columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing metric columns: {missing}")
        counts = frame.loc[:, list(self.schema.columns)].isna().sum()
        bad = counts[counts > 0]
        if not bad.empty:
            raise ValueError(f"Metric input contains missing values: {bad.to_dict()}")

    def fit(self, reference: pd.DataFrame) -> "MixedTypeEmbedding":
        self._validate_columns(reference)
        for column in self.schema.ordered:
            if column in self.schema.ordered_ranges:
                low, high = self.schema.ordered_ranges[column]
            else:
                values = pd.to_numeric(reference[column], errors="raise").to_numpy(float)
                low, high = float(values.min()), float(values.max())
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(f"Invalid scaling range for {column!r}: {(low, high)}")
            self._ranges[column] = (float(low), float(high))

        for column in self.schema.nominal:
            categories = self.schema.nominal_categories.get(column)
            if categories is None:
                categories = tuple(pd.unique(reference[column]))
            if len(categories) < 2:
                raise ValueError(f"Nominal column {column!r} needs at least two categories.")
            self._categories[column] = tuple(categories)

        self._fitted = True
        return self

    @property
    def output_dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("Embedding must be fitted first.")
        return len(self.schema.ordered) + sum(
            len(self._categories[column]) for column in self.schema.nominal
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedding must be fitted first.")
        self._validate_columns(frame)
        parts: list[np.ndarray] = []

        for column in self.schema.ordered:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
            low, high = self._ranges[column]
            if np.any(values < low) or np.any(values > high):
                observed = (float(values.min()), float(values.max()))
                raise ValueError(
                    f"Ordered values for {column!r} fall outside fixed range "
                    f"{(low, high)}: observed {observed}."
                )
            parts.append(((values - low) / (high - low)).reshape(-1, 1))

        for column in self.schema.nominal:
            categories = self._categories[column]
            lookup = {value: index for index, value in enumerate(categories)}
            codes = frame[column].map(lookup)
            if codes.isna().any():
                unknown = sorted(set(frame.loc[codes.isna(), column].tolist()), key=str)
                raise ValueError(f"Unknown categories in {column!r}: {unknown}")
            one_hot = np.zeros((len(frame), len(categories)), dtype=float)
            one_hot[np.arange(len(frame)), codes.to_numpy(dtype=int)] = 1.0
            parts.append(one_hot)

        if not parts:
            return np.empty((len(frame), 0), dtype=float)
        return np.ascontiguousarray(np.hstack(parts), dtype=float)

def _mean_pairwise_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    total = 0.0
    count = 0
    for start in range(0, len(left), chunk_size):
        delta = left[start : start + chunk_size, None, :] - right[None, :, :]
        distances = np.sqrt(np.sum(np.square(delta), axis=2))
        total += float(distances.sum())
        count += int(distances.size)
    return total / count

def energy_statistic(left: np.ndarray, right: np.ndarray) -> float:
    """Biased sample energy statistic matching the manuscript's n^-2 formula."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Energy inputs must be 2-D arrays with equal feature dimension.")
    value = (
        2.0 * _mean_pairwise_distance(left, right)
        - _mean_pairwise_distance(left, left)
        - _mean_pairwise_distance(right, right)
    )
    return float(max(value, 0.0))

def sqrt_energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Square root of :func:`energy_statistic`, matching the legacy code values."""
    return float(np.sqrt(energy_statistic(left, right)))

def bias_corrected_cramers_v(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left, right], axis=1).dropna()
    if paired.empty:
        return float("nan")
    table = pd.crosstab(paired.iloc[:, 0], paired.iloc[:, 1]).to_numpy()
    rows, columns = table.shape
    n = int(table.sum())
    if n <= 1 or min(rows, columns) <= 1:
        return float("nan")

    row_totals = table.sum(axis=1, keepdims=True)
    column_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / n
    chi2 = float(np.sum(np.square(table - expected) / expected))
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(phi2_corrected / denominator))

def correlation_ratio(ordered: pd.Series, nominal: pd.Series) -> float:
    paired = pd.concat([ordered, nominal], axis=1).dropna()
    if paired.empty:
        return float("nan")
    values = pd.to_numeric(paired.iloc[:, 0], errors="raise").to_numpy(float)
    groups = paired.iloc[:, 1].to_numpy()
    grand_mean = float(values.mean())
    total = float(np.square(values - grand_mean).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for group in pd.unique(groups):
        group_values = values[groups == group]
        between += len(group_values) * (float(group_values.mean()) - grand_mean) ** 2
    return float(np.sqrt(max(between, 0.0) / total))

def association_matrix(
    frame: pd.DataFrame,
    ordered: Sequence[str],
    nominal: Sequence[str],
) -> pd.DataFrame:
    ordered = tuple(ordered)
    nominal = tuple(nominal)
    columns = ordered + nominal
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing association columns: {missing}")

    matrix = np.eye(len(columns), dtype=float)
    for row_index, left_name in enumerate(columns):
        for column_index in range(row_index + 1, len(columns)):
            right_name = columns[column_index]
            if left_name in ordered and right_name in ordered:
                paired = frame.loc[:, [left_name, right_name]].dropna()
                left_rank = paired[left_name].rank(method="average").to_numpy(float)
                right_rank = paired[right_name].rank(method="average").to_numpy(float)
                left_centered = left_rank - left_rank.mean()
                right_centered = right_rank - right_rank.mean()
                denominator = float(
                    np.sqrt(
                        np.square(left_centered).sum()
                        * np.square(right_centered).sum()
                    )
                )
                value = (
                    float(np.dot(left_centered, right_centered) / denominator)
                    if denominator > 0
                    # A constant synthetic outcome is a complete association
                    # collapse.  Returning zero keeps the affected pair in the
                    # Tier-2 error instead of silently dropping it as NaN.
                    else 0.0
                )
            elif left_name in nominal and right_name in nominal:
                value = bias_corrected_cramers_v(frame[left_name], frame[right_name])
            elif left_name in ordered:
                value = correlation_ratio(frame[left_name], frame[right_name])
            else:
                value = correlation_ratio(frame[right_name], frame[left_name])
            matrix[row_index, column_index] = value
            matrix[column_index, row_index] = value
    return pd.DataFrame(matrix, index=columns, columns=columns)

@dataclass(frozen=True)
class AssociationComparison:
    reference: pd.DataFrame
    synthetic: pd.DataFrame
    difference: pd.DataFrame
    mae: float
    rmse: float
    max_abs: float

def compare_association(
    reference_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    ordered: Sequence[str],
    nominal: Sequence[str],
) -> AssociationComparison:
    reference = association_matrix(reference_frame, ordered, nominal)
    synthetic = association_matrix(synthetic_frame, ordered, nominal)
    difference = synthetic - reference
    upper = np.triu_indices_from(difference.to_numpy(), k=1)
    values = difference.to_numpy()[upper]
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        mae = rmse = max_abs = float("nan")
    else:
        mae = float(np.mean(np.abs(finite)))
        rmse = float(np.sqrt(np.mean(np.square(finite))))
        max_abs = float(np.max(np.abs(finite)))
    return AssociationComparison(reference, synthetic, difference, mae, rmse, max_abs)


# Baselines

@dataclass(frozen=True)
class WeightingDiagnostics:
    method: str
    n_human: int
    n_synthetic: int
    n_features: int
    weight_min: float
    weight_max: float
    weight_mean: float
    effective_sample_size: float
    effective_sample_fraction: float
    clipped_low_fraction: float
    clipped_high_fraction: float
    converged: bool | None = None
    iterations: int | None = None
    max_margin_error_before_clipping: float | None = None
    max_margin_error_after_clipping: float | None = None
    structural_zero_cells: int | None = None
    oof_auc: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class _RakingMargin:
    name: str
    human_codes: np.ndarray
    synthetic_codes: np.ndarray
    levels: tuple[str, ...]
    target: np.ndarray

def _validate_frames(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    categorical = tuple(categorical_cols)
    numerical = tuple(numerical_cols)
    all_columns = categorical + numerical
    if not all_columns:
        raise ValueError("At least one weighting feature is required.")
    if len(all_columns) != len(set(all_columns)):
        raise ValueError("Categorical and numerical weighting features overlap.")
    for name, frame in (("human", human), ("synthetic", synthetic)):
        missing = sorted(set(all_columns) - set(frame.columns))
        if missing:
            raise KeyError(f"{name} frame is missing weighting features: {missing}")
        nulls = frame.loc[:, list(all_columns)].isna().sum()
        bad = nulls[nulls > 0]
        if not bad.empty:
            raise ValueError(f"{name} weighting features contain missing values: {bad.to_dict()}")
    if len(human) < 2 or len(synthetic) < 2:
        raise ValueError("Human and synthetic samples must each contain at least two rows.")
    return categorical, numerical

def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("weights must be a non-empty one-dimensional array.")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("weights must be finite and nonnegative.")
    total = float(result.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum.")
    return result * (len(result) / total)

def effective_sample_size(weights: np.ndarray) -> float:
    normalised = _normalise_weights(weights)
    denominator = float(np.sum(np.square(normalised)))
    return float(normalised.sum() ** 2 / denominator)

def _clip_and_summarise(
    weights: np.ndarray,
    *,
    clip_low: float,
    clip_high: float,
) -> tuple[np.ndarray, float, float]:
    if not 0 <= clip_low <= 1.0 <= clip_high or clip_low == clip_high:
        raise ValueError("weight_clip must satisfy 0 <= low <= 1 <= high.")
    normalised = _normalise_weights(weights)
    low_fraction = float(np.mean(normalised < clip_low))
    high_fraction = float(np.mean(normalised > clip_high))
    # Find one common scale whose clipped values retain mean weight one. This
    # enforces the declared bounds after normalisation (a simple clip followed
    # by renormalisation can exceed the upper bound again).
    target_sum = float(len(normalised))
    scale_low = 0.0
    scale_high = 1.0
    while float(np.clip(normalised * scale_high, clip_low, clip_high).sum()) < target_sum:
        scale_high *= 2.0
    for _ in range(100):
        scale_mid = (scale_low + scale_high) / 2.0
        current = float(np.clip(normalised * scale_mid, clip_low, clip_high).sum())
        if current < target_sum:
            scale_low = scale_mid
        else:
            scale_high = scale_mid
    clipped = np.clip(normalised * scale_high, clip_low, clip_high)
    return clipped, low_fraction, high_fraction

def _categorical_codes(series: pd.Series) -> np.ndarray:
    # CSV type inference may load the same coded category as integer in one file
    # and float in another. Canonicalise integral numeric values so 1 and 1.0
    # remain the same survey category.
    def token(value: Any) -> str:
        if isinstance(value, (int, np.integer)):
            return f"num:{int(value)}"
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return f"num:{int(number)}" if number.is_integer() else f"num:{number:.17g}"
        return f"text:{value}"

    return np.asarray([token(value) for value in series.to_numpy()], dtype=object)

def _numeric_bin_codes(
    human: pd.Series,
    synthetic: pd.Series,
    *,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("numeric_bins must be at least two.")
    human_values = human.to_numpy(dtype=float)
    synthetic_values = synthetic.to_numpy(dtype=float)
    if not np.isfinite(human_values).all() or not np.isfinite(synthetic_values).all():
        raise ValueError(f"Numerical raking feature {human.name!r} is not finite.")
    interior = np.quantile(human_values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    interior = np.unique(interior)
    edges = np.r_[-np.inf, interior, np.inf]
    human_codes = np.searchsorted(edges[1:-1], human_values, side="right")
    synthetic_codes = np.searchsorted(edges[1:-1], synthetic_values, side="right")
    return (
        np.asarray([f"bin:{value}" for value in human_codes], dtype=object),
        np.asarray([f"bin:{value}" for value in synthetic_codes], dtype=object),
    )

def _build_raking_margins(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
    numeric_bins: int,
    smoothing: float,
) -> list[_RakingMargin]:
    if smoothing < 0:
        raise ValueError("raking smoothing must be nonnegative.")
    margins: list[_RakingMargin] = []
    for column in categorical_cols:
        human_codes = _categorical_codes(human[column])
        synthetic_codes = _categorical_codes(synthetic[column])
        levels = tuple(sorted(set(human_codes) | set(synthetic_codes)))
        counts = np.asarray([(human_codes == level).sum() for level in levels], dtype=float)
        target = (counts + smoothing) / (len(human_codes) + smoothing * len(levels))
        margins.append(_RakingMargin(column, human_codes, synthetic_codes, levels, target))
    for column in numerical_cols:
        human_codes, synthetic_codes = _numeric_bin_codes(
            human[column], synthetic[column], bins=numeric_bins
        )
        levels = tuple(sorted(set(human_codes) | set(synthetic_codes)))
        counts = np.asarray([(human_codes == level).sum() for level in levels], dtype=float)
        target = (counts + smoothing) / (len(human_codes) + smoothing * len(levels))
        margins.append(
            _RakingMargin(f"{column}__quantile_bin", human_codes, synthetic_codes, levels, target)
        )
    return margins

def _max_margin_error(weights: np.ndarray, margins: Sequence[_RakingMargin]) -> float:
    normalised = _normalise_weights(weights)
    total = float(normalised.sum())
    discrepancies: list[float] = []
    for margin in margins:
        current = np.asarray(
            [normalised[margin.synthetic_codes == level].sum() / total for level in margin.levels]
        )
        discrepancies.extend(np.abs(current - margin.target).tolist())
    return float(max(discrepancies, default=0.0))

def raking_ipf_weights(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
    numeric_bins: int = 5,
    smoothing: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    weight_clip: tuple[float, float] = (0.05, 20.0),
) -> tuple[np.ndarray, WeightingDiagnostics]:
    """Estimate record weights by iterative proportional fitting.

    Categorical margins use their observed levels. Numerical features are first
    discretised at human-sample quantiles. A small configurable pseudocount
    stabilises rare categories. Final clipping is reported explicitly because it
    can reintroduce a small marginal discrepancy.
    """
    categorical, numerical = _validate_frames(
        human,
        synthetic,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive.")
    margins = _build_raking_margins(
        human,
        synthetic,
        categorical_cols=categorical,
        numerical_cols=numerical,
        numeric_bins=numeric_bins,
        smoothing=smoothing,
    )
    structural_zero_cells = sum(
        int(not np.any(margin.synthetic_codes == level))
        for margin in margins
        for level, target in zip(margin.levels, margin.target)
        if target > 0
    )

    weights = np.ones(len(synthetic), dtype=float)
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        for margin in margins:
            total = float(weights.sum())
            for level, target in zip(margin.levels, margin.target):
                mask = margin.synthetic_codes == level
                current_mass = float(weights[mask].sum())
                if current_mass > 0 and target > 0:
                    weights[mask] *= float(target) * total / current_mass
        weights = _normalise_weights(weights)
        if _max_margin_error(weights, margins) <= tolerance:
            converged = True
            break

    before_clipping = _max_margin_error(weights, margins)
    clipped, low_fraction, high_fraction = _clip_and_summarise(
        weights, clip_low=float(weight_clip[0]), clip_high=float(weight_clip[1])
    )
    after_clipping = _max_margin_error(clipped, margins)
    ess = effective_sample_size(clipped)
    diagnostics = WeightingDiagnostics(
        method="Raking/IPF",
        n_human=len(human),
        n_synthetic=len(synthetic),
        n_features=len(margins),
        weight_min=float(clipped.min()),
        weight_max=float(clipped.max()),
        weight_mean=float(clipped.mean()),
        effective_sample_size=ess,
        effective_sample_fraction=float(ess / len(clipped)),
        clipped_low_fraction=low_fraction,
        clipped_high_fraction=high_fraction,
        converged=converged,
        iterations=iterations,
        max_margin_error_before_clipping=before_clipping,
        max_margin_error_after_clipping=after_clipping,
        structural_zero_cells=structural_zero_cells,
    )
    return clipped, diagnostics

def _design_matrix(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
) -> tuple[np.ndarray, int]:
    combined = pd.concat(
        [
            human.loc[:, [*categorical_cols, *numerical_cols]],
            synthetic.loc[:, [*categorical_cols, *numerical_cols]],
        ],
        ignore_index=True,
    )
    parts: list[np.ndarray] = []
    if numerical_cols:
        numeric = combined.loc[:, list(numerical_cols)].to_numpy(dtype=float)
        mean = numeric.mean(axis=0)
        scale = numeric.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        parts.append((numeric - mean) / scale)
    if categorical_cols:
        tokens = pd.DataFrame(
            {
                column: _categorical_codes(combined[column])
                for column in categorical_cols
            }
        )
        dummies = pd.get_dummies(tokens, columns=list(categorical_cols), dtype=float)
        parts.append(dummies.to_numpy(dtype=float))
    matrix = np.column_stack(parts)
    if not np.isfinite(matrix).all():
        raise ValueError("Density-ratio design matrix contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=float), int(matrix.shape[1])

def density_ratio_weights(
    human: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
    seed: int,
    n_splits: int = 5,
    regularization_c: float = 1.0,
    max_iterations: int = 2000,
    probability_clip: float = 1e-6,
    weight_clip: tuple[float, float] = (0.05, 20.0),
) -> tuple[np.ndarray, WeightingDiagnostics]:
    """Estimate p(human)/p(synthetic) with cross-fitted logistic odds."""
    categorical, numerical = _validate_frames(
        human,
        synthetic,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
    if not 2 <= n_splits <= min(len(human), len(synthetic)):
        raise ValueError("n_splits must be between 2 and the smaller class size.")
    if regularization_c <= 0 or max_iterations <= 0:
        raise ValueError("regularization_c and max_iterations must be positive.")
    if not 0 < probability_clip < 0.5:
        raise ValueError("probability_clip must lie in (0, 0.5).")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    features, n_features = _design_matrix(
        human,
        synthetic,
        categorical_cols=categorical,
        numerical_cols=numerical,
    )
    labels = np.r_[np.ones(len(human), dtype=int), np.zeros(len(synthetic), dtype=int)]
    probabilities = np.full(len(labels), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    fold_iterations: list[int] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(features, labels)):
        classifier = LogisticRegression(
            C=float(regularization_c),
            max_iter=int(max_iterations),
            solver="lbfgs",
            random_state=int(seed) + fold,
        )
        classifier.fit(features[train_index], labels[train_index])
        fold_iterations.append(int(np.max(classifier.n_iter_)))
        probabilities[test_index] = classifier.predict_proba(features[test_index])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Cross-fitted density-ratio probabilities are incomplete.")

    oof_auc = float(roc_auc_score(labels, probabilities))
    synthetic_probability = np.clip(
        probabilities[len(human) :], probability_clip, 1.0 - probability_clip
    )
    empirical_prior_correction = len(synthetic) / len(human)
    raw_weights = (
        synthetic_probability / (1.0 - synthetic_probability)
    ) * empirical_prior_correction
    clipped, low_fraction, high_fraction = _clip_and_summarise(
        raw_weights, clip_low=float(weight_clip[0]), clip_high=float(weight_clip[1])
    )
    ess = effective_sample_size(clipped)
    diagnostics = WeightingDiagnostics(
        method="Density-ratio weighting",
        n_human=len(human),
        n_synthetic=len(synthetic),
        n_features=n_features,
        weight_min=float(clipped.min()),
        weight_max=float(clipped.max()),
        weight_mean=float(clipped.mean()),
        effective_sample_size=ess,
        effective_sample_fraction=float(ess / len(clipped)),
        clipped_low_fraction=low_fraction,
        clipped_high_fraction=high_fraction,
        converged=all(iterations < max_iterations for iterations in fold_iterations),
        iterations=max(fold_iterations),
        oof_auc=oof_auc,
    )
    return clipped, diagnostics

def weighted_resample_indices(
    weights: np.ndarray,
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    normalised = _normalise_weights(weights)
    if n <= 0:
        raise ValueError("The resampled output size must be positive.")
    probabilities = normalised / normalised.sum()
    rng = np.random.default_rng(int(seed))
    return rng.choice(len(normalised), size=int(n), replace=True, p=probabilities).astype(int)


# Sample size

def stratified_nested_order(
    frame: pd.DataFrame,
    *,
    stratify_col: str,
    seed: int,
) -> np.ndarray:
    """Return a deterministic order whose prefixes preserve stratum proportions.

    Records are shuffled within strata. A weighted-deficit scheduler then chooses
    the next stratum, so every smaller training set is a strict prefix of every
    larger one for the same seed.
    """
    if stratify_col not in frame.columns:
        raise KeyError(f"Missing stratification column {stratify_col!r}.")
    if frame[stratify_col].isna().any():
        raise ValueError(f"Stratification column {stratify_col!r} contains missing values.")

    rng = np.random.default_rng(int(seed))
    groups = sorted(pd.unique(frame[stratify_col]), key=str)
    queues: dict[Any, list[int]] = {}
    for group in groups:
        indices = np.flatnonzero(frame[stratify_col].to_numpy() == group)
        queues[group] = rng.permutation(indices).astype(int).tolist()

    total = len(frame)
    proportions = {group: len(queues[group]) / total for group in groups}
    selected = {group: 0 for group in groups}
    positions = {group: 0 for group in groups}
    order: list[int] = []
    for step in range(total):
        eligible = [
            group for group in groups if positions[group] < len(queues[group])
        ]
        group = max(
            eligible,
            key=lambda value: (
                (step + 1) * proportions[value] - selected[value],
                -groups.index(value),
            ),
        )
        order.append(queues[group][positions[group]])
        positions[group] += 1
        selected[group] += 1
    return np.asarray(order, dtype=int)

def uniform_tvae_sample(
    benchmark: pd.DataFrame,
    *,
    config: ProjectConfig,
    seeds: SeedPlan,
    pools: Mapping[Any, CandidatePool],
    outcome_encoder: Any,
) -> pd.DataFrame:
    id_col = config.columns.id_col
    selected: list[np.ndarray] = []
    for record_key in benchmark[id_col].array:
        pool = pools[record_key]
        rng = np.random.default_rng(
            seeds.for_stage("sample_size_sweep", "pure_tvae", record_key)
        )
        selected.append(pool.encoded[int(rng.integers(len(pool.encoded)))])
    decoded = outcome_encoder.inverse_transform(np.vstack(selected))
    return pd.concat(
        [benchmark.loc[:, [id_col]].reset_index(drop=True), decoded], axis=1
    )


def metric_schema(config: ProjectConfig) -> MixedTypeSchema:
    columns = config.columns
    return MixedTypeSchema(
        ordered=columns.metric_ordered,
        nominal=columns.metric_nominal,
        ordered_ranges=columns.ordered_ranges,
        nominal_categories={column: columns.category_levels[column] for column in columns.metric_nominal},
    )
