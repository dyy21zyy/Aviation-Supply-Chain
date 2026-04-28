from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import beta


PROJECT_DIRNAME = "aviation_sideinfo_case"


def project_root() -> Path:
    """Return project root path (.../aviation_sideinfo_case)."""
    return Path(__file__).resolve().parents[1]


def ensure_directories() -> Tuple[Path, Path, Path]:
    """Ensure project, data, and outputs directories exist and return them."""
    root = project_root()
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return root, data_dir, outputs_dir


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file with basic path validation."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {file_path}")
    return pd.read_csv(file_path)


def beta_prior_mean(a: float, b: float) -> float:
    """Return the mean of a Beta(a, b) prior."""
    if a <= 0 or b <= 0:
        raise ValueError("Beta parameters a and b must be positive.")
    return a / (a + b)


def beta_prior_ci(a: float, b: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Return a central (1-alpha) credible interval for Beta(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("Beta parameters a and b must be positive.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    lower = beta.ppf(alpha / 2.0, a, b)
    upper = beta.ppf(1.0 - alpha / 2.0, a, b)
    return lower, upper


def scaled_beta_pdf(x_0_100: float, a: float, b: float) -> float:
    """Evaluate a Beta(a, b) density on x in [0,100] after scaling by z=x/100."""
    if a <= 0 or b <= 0:
        raise ValueError("Beta parameters a and b must be positive.")
    if not np.isfinite(x_0_100):
        raise ValueError("x_0_100 must be finite.")
    if x_0_100 < 0 or x_0_100 > 100:
        raise ValueError("x_0_100 must lie within [0, 100].")

    z = np.clip(x_0_100 / 100.0, 1e-6, 1 - 1e-6)
    return beta.pdf(z, a, b) / 100.0


def get_categorical_likelihood(
    signal: str, state_value: int, likelihood_df: pd.DataFrame
) -> Tuple[float, float]:
    """Return P(signal=state_value|R=1), P(signal=state_value|R=0)."""
    required_cols = {
        "signal",
        "state_value",
        "likelihood_given_disruption",
        "likelihood_given_no_disruption",
    }
    missing = required_cols.difference(likelihood_df.columns)
    if missing:
        raise ValueError(f"likelihood_df is missing required columns: {sorted(missing)}")

    match = likelihood_df[
        (likelihood_df["signal"] == signal)
        & (likelihood_df["state_value"] == state_value)
    ]
    if match.empty:
        raise ValueError(
            f"Missing likelihood for signal-state pair: signal='{signal}', state_value={state_value}"
        )

    row = match.iloc[0]
    l_r1 = float(row["likelihood_given_disruption"])
    l_r0 = float(row["likelihood_given_no_disruption"])
    if l_r1 < 0 or l_r0 < 0:
        raise ValueError("Likelihood values must be non-negative.")
    return l_r1, l_r0


def bayesian_posterior(prior: float, likelihood_R1: float, likelihood_R0: float) -> float:
    """Compute posterior P(R=1|signals) by Bayes rule."""
    if not (0 <= prior <= 1):
        raise ValueError("prior must lie in [0, 1].")
    if likelihood_R1 < 0 or likelihood_R0 < 0:
        raise ValueError("Likelihoods must be non-negative.")

    denom = prior * likelihood_R1 + (1 - prior) * likelihood_R0
    if denom <= 0:
        raise ValueError("Posterior denominator must be positive.")
    return (prior * likelihood_R1) / denom


def decision_threshold(action_cost: float, disruption_loss: float, action_effectiveness: float) -> float:
    """Return threshold tau = C_A / (e * C_D)."""
    denom = action_effectiveness * disruption_loss
    if denom <= 0:
        raise ValueError("action_effectiveness * disruption_loss must be positive.")
    return action_cost / denom


def expected_loss_act(
    p: float,
    action_cost: float,
    disruption_loss: float,
    action_effectiveness: float,
) -> float:
    """Return expected loss when taking resilience action."""
    return action_cost + p * (1 - action_effectiveness) * disruption_loss


def expected_loss_no_act(p: float, disruption_loss: float) -> float:
    """Return expected loss when not taking resilience action."""
    return p * disruption_loss
