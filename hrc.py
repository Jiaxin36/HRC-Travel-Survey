"""Conditional TVAE training and hybrid reweighted calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import random
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
import torch.nn.functional as functional
from data_utils import CalibrationConfig, ProjectConfig, TVAEConfig


# Seeds

def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable 32-bit seed from a base seed and semantic keys."""
    payload = json.dumps(
        [int(base_seed), *parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)

def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed process-level RNGs. Per-operation seeds are still preferred."""
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

@dataclass(frozen=True)
class SeedPlan:
    base_seed: int
    run_key: str

    def for_stage(self, stage: str, *parts: Any) -> int:
        return derive_seed(self.base_seed, self.run_key, stage, *parts)

    @property
    def data_sample(self) -> int:
        return self.for_stage("data_sample")

    @property
    def model_init(self) -> int:
        return self.for_stage("model_init")

    @property
    def data_loader(self) -> int:
        return self.for_stage("data_loader")

    @property
    def training_latent(self) -> int:
        return self.for_stage("training_latent")

    def candidate(self, record_key: Any) -> int:
        return self.for_stage("candidate", record_key)

    def selection(self, record_key: Any, gamma: float) -> int:
        return self.for_stage("selection", record_key, f"{gamma:.12g}")


# Encoding

def _make_one_hot_encoder(
    *, categories: list[list[Any]] | str, handle_unknown: str
) -> OneHotEncoder:
    kwargs = {
        "categories": categories,
        "handle_unknown": handle_unknown,
        "dtype": np.float32,
    }
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(sparse=False, **kwargs)

@dataclass(frozen=True)
class EncoderLayout:
    numeric_dim: int
    categorical_sizes: tuple[int, ...]

    @property
    def output_dim(self) -> int:
        return self.numeric_dim + sum(self.categorical_sizes)

class TabularEncoder:
    """Fit-once encoder shared by training, calibration, and inverse transform.

    Fixed categorical supports prevent output dimensions from changing when a
    small training subset happens to omit a valid response category.
    """

    def __init__(
        self,
        *,
        categorical_cols: Sequence[str],
        numerical_cols: Sequence[str],
        categorical_levels: Mapping[str, Sequence[Any]] | None = None,
        handle_unknown: str = "error",
    ) -> None:
        self.categorical_cols = tuple(categorical_cols)
        self.numerical_cols = tuple(numerical_cols)
        self.columns = self.categorical_cols + self.numerical_cols
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Categorical and numerical column definitions overlap.")
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        self.categorical_levels = {
            key: tuple(value) for key, value in (categorical_levels or {}).items()
        }
        if self.categorical_cols and self.categorical_levels:
            missing = set(self.categorical_cols) - set(self.categorical_levels)
            if missing:
                raise ValueError(f"Missing fixed categorical levels for: {sorted(missing)}")
            categories: list[list[Any]] | str = [
                list(self.categorical_levels[column])
                for column in self.categorical_cols
            ]
        else:
            categories = "auto"

        self._one_hot = (
            _make_one_hot_encoder(
                categories=categories,
                handle_unknown=handle_unknown,
            )
            if self.categorical_cols
            else None
        )
        self._scaler = StandardScaler() if self.numerical_cols else None
        self._fitted = False

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing encoder columns: {missing}")
        null_counts = frame.loc[:, list(self.columns)].isna().sum()
        bad = null_counts[null_counts > 0]
        if not bad.empty:
            raise ValueError(f"Encoder input contains missing values: {bad.to_dict()}")

    def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
        self._validate_frame(frame)
        if self._scaler is not None:
            self._scaler.fit(frame.loc[:, list(self.numerical_cols)])
        if self._one_hot is not None:
            self._one_hot.fit(frame.loc[:, list(self.categorical_cols)])
        self._fitted = True
        return self

    @property
    def layout(self) -> EncoderLayout:
        if not self._fitted:
            raise RuntimeError("Encoder must be fitted before accessing its layout.")
        sizes = (
            tuple(len(categories) for categories in self._one_hot.categories_)
            if self._one_hot is not None
            else ()
        )
        return EncoderLayout(len(self.numerical_cols), sizes)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder must be fitted before transform.")
        self._validate_frame(frame)
        parts: list[np.ndarray] = []
        if self._scaler is not None:
            parts.append(
                np.asarray(
                    self._scaler.transform(frame.loc[:, list(self.numerical_cols)]),
                    dtype=np.float32,
                )
            )
        if self._one_hot is not None:
            parts.append(
                np.asarray(
                    self._one_hot.transform(frame.loc[:, list(self.categorical_cols)]),
                    dtype=np.float32,
                )
            )
        if not parts:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.ascontiguousarray(np.hstack(parts), dtype=np.float32)

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Encoder must be fitted before inverse_transform.")
        array = np.asarray(encoded, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.layout.output_dim:
            raise ValueError(
                f"Expected encoded shape (*, {self.layout.output_dim}), got {array.shape}."
            )

        result: dict[str, Any] = {}
        numeric_end = self.layout.numeric_dim
        if self._scaler is not None:
            numeric = self._scaler.inverse_transform(array[:, :numeric_end])
            for index, column in enumerate(self.numerical_cols):
                result[column] = numeric[:, index]
        if self._one_hot is not None:
            categorical = self._one_hot.inverse_transform(array[:, numeric_end:])
            for index, column in enumerate(self.categorical_cols):
                result[column] = categorical[:, index]

        return pd.DataFrame(result).loc[:, list(self.columns)]


# Model

def _torch_generator(seed: int, device: torch.device) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator

class ConditionalTVAE(nn.Module):
    """Conditional tabular VAE with per-variable reconstruction balancing."""

    def __init__(
        self,
        *,
        input_dim: int,
        condition_dim: int,
        numeric_dim: int,
        categorical_sizes: Sequence[int],
        config: TVAEConfig,
        init_seed: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim)
        self.numeric_dim = int(numeric_dim)
        self.categorical_sizes = tuple(int(value) for value in categorical_sizes)
        self.config = config
        self.device = torch.device(device)

        if self.numeric_dim + sum(self.categorical_sizes) != self.input_dim:
            raise ValueError("Numeric and categorical layout does not match input_dim.")

        # Parameter initialization is independent of all prior Torch operations.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim + self.condition_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
            self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(config.latent_dim + self.condition_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, self.input_dim),
            )
        self.to(self.device)

    def encode(self, outcomes: torch.Tensor, conditions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(torch.cat([outcomes, conditions], dim=1))
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        log_variance: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        epsilon = torch.randn(
            mu.shape,
            dtype=mu.dtype,
            device=mu.device,
            generator=generator,
        )
        return mu + epsilon * torch.exp(0.5 * log_variance)

    def decode(self, latent: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, conditions], dim=1))

    def forward(
        self,
        outcomes: torch.Tensor,
        conditions: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encode(outcomes, conditions)
        latent = self.reparameterize(mu, log_variance, generator=generator)
        return self.decode(latent, conditions), mu, log_variance

    def per_sample_reconstruction_score(
        self,
        reconstruction_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return a variable-count-weighted reconstruction score per row."""
        n_rows = targets.shape[0]
        numeric_score = torch.zeros(n_rows, device=targets.device)
        if self.numeric_dim:
            numeric_score = torch.square(
                targets[:, : self.numeric_dim]
                - reconstruction_logits[:, : self.numeric_dim]
            ).mean(dim=1)

        categorical_parts: list[torch.Tensor] = []
        start = self.numeric_dim
        for size in self.categorical_sizes:
            logits = reconstruction_logits[:, start : start + size]
            target_index = targets[:, start : start + size].argmax(dim=1)
            categorical_parts.append(
                functional.cross_entropy(logits, target_index, reduction="none")
            )
            start += size
        categorical_score = (
            torch.stack(categorical_parts, dim=1).mean(dim=1)
            if categorical_parts
            else torch.zeros(n_rows, device=targets.device)
        )

        numeric_variables = self.numeric_dim
        categorical_variables = len(self.categorical_sizes)
        total_variables = numeric_variables + categorical_variables
        if total_variables == 0:
            raise ValueError("The outcome encoder has no variables.")
        numeric_weight = numeric_variables / total_variables
        categorical_weight = categorical_variables / total_variables
        return numeric_weight * numeric_score + categorical_weight * categorical_score

    def loss(
        self,
        reconstruction_logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction = self.per_sample_reconstruction_score(
            reconstruction_logits, targets
        ).mean()
        kl = -0.5 * torch.mean(
            1.0 + log_variance - mu.square() - log_variance.exp()
        )
        total = reconstruction + self.config.kl_weight * kl
        return total, reconstruction, kl

    def fit_model(
        self,
        outcomes: np.ndarray,
        conditions: np.ndarray,
        *,
        data_loader_seed: int,
        latent_seed: int,
    ) -> list[dict[str, float]]:
        x = torch.as_tensor(outcomes, dtype=torch.float32)
        c = torch.as_tensor(conditions, dtype=torch.float32)
        if len(x) != len(c):
            raise ValueError("Outcome and condition row counts differ.")
        dataset = torch.utils.data.TensorDataset(x, c)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(int(data_loader_seed))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
        )
        latent_generator = _torch_generator(latent_seed, self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        history: list[dict[str, float]] = []

        self.train()
        for epoch in range(self.config.epochs):
            totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
            seen = 0
            for x_batch, c_batch in loader:
                x_batch = x_batch.to(self.device)
                c_batch = c_batch.to(self.device)
                reconstruction_logits, mu, log_variance = self.forward(
                    x_batch,
                    c_batch,
                    generator=latent_generator,
                )
                total, reconstruction, kl = self.loss(
                    reconstruction_logits, x_batch, mu, log_variance
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()

                batch_size = len(x_batch)
                seen += batch_size
                totals["loss"] += float(total.detach().cpu()) * batch_size
                totals["reconstruction"] += float(reconstruction.detach().cpu()) * batch_size
                totals["kl"] += float(kl.detach().cpu()) * batch_size

            history.append(
                {
                    "epoch": float(epoch + 1),
                    **{name: value / seen for name, value in totals.items()},
                }
            )
        return history

    def _hard_or_sample_categorical(
        self,
        logits: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        output = logits.clone()
        start = self.numeric_dim
        for size in self.categorical_sizes:
            block = logits[:, start : start + size]
            if self.config.categorical_sampling == "argmax":
                indices = block.argmax(dim=1)
            else:
                probabilities = functional.softmax(block, dim=1)
                indices = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                ).squeeze(1)
            output[:, start : start + size] = functional.one_hot(
                indices, num_classes=size
            ).to(dtype=logits.dtype)
            start += size
        return output

    @torch.no_grad()
    def sample_given_condition(
        self,
        condition: np.ndarray,
        *,
        n: int,
        seed: int,
    ) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive.")
        self.eval()
        condition_array = np.asarray(condition, dtype=np.float32).reshape(1, -1)
        if condition_array.shape[1] != self.condition_dim:
            raise ValueError("Condition dimension does not match the model.")
        generator = _torch_generator(seed, self.device)
        condition_tensor = torch.as_tensor(
            condition_array, dtype=torch.float32, device=self.device
        ).repeat(n, 1)
        latent = torch.randn(
            (n, self.config.latent_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        logits = self.decode(latent, condition_tensor)
        output = self._hard_or_sample_categorical(logits, generator=generator)
        return output.cpu().numpy()

    @torch.no_grad()
    def score_candidates(
        self,
        candidates: np.ndarray,
        conditions: np.ndarray,
        *,
        posterior_mean: bool = True,
        seed: int | None = None,
    ) -> np.ndarray:
        self.eval()
        candidate_tensor = torch.as_tensor(
            candidates, dtype=torch.float32, device=self.device
        )
        condition_tensor = torch.as_tensor(
            conditions, dtype=torch.float32, device=self.device
        )
        mu, log_variance = self.encode(candidate_tensor, condition_tensor)
        if posterior_mean:
            latent = mu
        else:
            if seed is None:
                raise ValueError("A seed is required for stochastic candidate scoring.")
            latent = self.reparameterize(
                mu,
                log_variance,
                generator=_torch_generator(seed, self.device),
            )
        logits = self.decode(latent, condition_tensor)
        return self.per_sample_reconstruction_score(
            logits, candidate_tensor
        ).cpu().numpy()

    def metadata(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "condition_dim": self.condition_dim,
            "numeric_dim": self.numeric_dim,
            "categorical_sizes": self.categorical_sizes,
            "config": asdict(self.config),
        }


# Scoring

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without requiring SciPy."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks

def scale_lower_is_better(values: np.ndarray, method: str = "rank") -> np.ndarray:
    """Put lower-is-better scores on a comparable, dimensionless scale."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Scores must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Scores contain non-finite values.")

    if method == "rank":
        if len(array) == 1:
            return np.zeros(1, dtype=float)
        return (_average_ranks(array) - 1.0) / (len(array) - 1.0)

    if method == "minmax":
        low = float(array.min())
        high = float(array.max())
        if high <= low:
            return np.zeros_like(array)
        return (array - low) / (high - low)

    if method == "robust_z":
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        if mad <= np.finfo(float).eps:
            return np.zeros_like(array)
        return np.clip((array - median) / (1.4826 * mad), -8.0, 8.0)

    raise ValueError(f"Unknown score scaling method: {method!r}")

def hybrid_selection_probabilities(
    llm_distance: np.ndarray,
    human_score: np.ndarray,
    *,
    gamma: float,
    scaling: str = "rank",
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    llm_scaled = scale_lower_is_better(llm_distance, scaling)
    human_scaled = scale_lower_is_better(human_score, scaling)
    if llm_scaled.shape != human_scaled.shape:
        raise ValueError("LLM and human scores must have the same shape.")

    energy = gamma * llm_scaled + (1.0 - gamma) * human_scaled
    logits = -energy / temperature
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Hybrid weights cannot be normalized.")
    return weights / total, llm_scaled, human_scaled


# Calibration

@dataclass(frozen=True)
class CandidatePool:
    encoded: np.ndarray
    human_score: np.ndarray
    candidate_seed: int

class HRCCalibrator:
    def __init__(
        self,
        *,
        model: ConditionalTVAE,
        condition_encoder: TabularEncoder,
        outcome_encoder: TabularEncoder,
        condition_cols: tuple[str, ...],
        outcome_cols: tuple[str, ...],
        config: CalibrationConfig,
        seed_plan: SeedPlan,
    ) -> None:
        self.model = model
        self.condition_encoder = condition_encoder
        self.outcome_encoder = outcome_encoder
        self.condition_cols = condition_cols
        self.outcome_cols = outcome_cols
        self.config = config
        self.seed_plan = seed_plan


    @staticmethod
    def _one_row(record: pd.Series, columns: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame([{column: record[column] for column in columns}])


    def generate_candidate_pool(
        self,
        record: pd.Series,
        *,
        record_key: Any,
    ) -> CandidatePool:
        condition = self.condition_encoder.transform(
            self._one_row(record, self.condition_cols)
        )[0]
        candidate_seed = self.seed_plan.candidate(record_key)
        candidates = self.model.sample_given_condition(
            condition,
            n=self.config.candidate_pool_size,
            seed=candidate_seed,
        )
        condition_matrix = np.repeat(condition[None, :], len(candidates), axis=0)
        human_score = self.model.score_candidates(
            candidates,
            condition_matrix,
            posterior_mean=self.config.posterior_mean_for_human_score,
            seed=(
                None
                if self.config.posterior_mean_for_human_score
                else self.seed_plan.for_stage("human_score", record_key)
            ),
        )
        return CandidatePool(candidates, human_score, candidate_seed)


# Pipeline

@dataclass
class HRCBundle:
    condition_encoder: TabularEncoder
    outcome_encoder: TabularEncoder
    model: ConditionalTVAE
    calibrator: HRCCalibrator
    training_history: list[dict[str, float]]
    seed_plan: SeedPlan

def fit_hrc_bundle(
    training_frame: pd.DataFrame,
    *,
    config: ProjectConfig,
    seed_plan: SeedPlan,
    device: str,
) -> HRCBundle:
    columns = config.columns
    outcome_levels = {column: config.columns.category_levels[column] for column in columns.outcome_categorical}
    condition_encoder = TabularEncoder(
        categorical_cols=columns.condition_categorical,
        numerical_cols=columns.condition_numeric,
        handle_unknown='ignore',
    ).fit(training_frame)
    outcome_encoder = TabularEncoder(
        categorical_cols=columns.outcome_categorical,
        numerical_cols=columns.outcome_numeric,
        categorical_levels=outcome_levels,
        handle_unknown='error',
    ).fit(training_frame)
    encoded_conditions = condition_encoder.transform(training_frame)
    encoded_outcomes = outcome_encoder.transform(training_frame)
    model = ConditionalTVAE(
        input_dim=outcome_encoder.layout.output_dim,
        condition_dim=condition_encoder.layout.output_dim,
        numeric_dim=outcome_encoder.layout.numeric_dim,
        categorical_sizes=outcome_encoder.layout.categorical_sizes,
        config=config.model,
        init_seed=seed_plan.model_init,
        device=device,
    )
    history = model.fit_model(
        encoded_outcomes, encoded_conditions,
        data_loader_seed=seed_plan.data_loader,
        latent_seed=seed_plan.training_latent,
    )
    calibrator = HRCCalibrator(
        model=model,
        condition_encoder=condition_encoder,
        outcome_encoder=outcome_encoder,
        condition_cols=columns.condition_cols,
        outcome_cols=columns.outcome_cols,
        config=config.calibration,
        seed_plan=seed_plan,
    )
    return HRCBundle(condition_encoder, outcome_encoder, model, calibrator, history, seed_plan)
