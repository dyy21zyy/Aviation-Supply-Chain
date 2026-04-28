from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    bayesian_posterior,
    beta_prior_ci,
    beta_prior_mean,
    decision_threshold,
    get_categorical_likelihood,
    load_csv,
    scaled_beta_pdf,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    signals = load_csv(data_dir / "antwerp_case_signals.csv").sort_values("event_day")
    likelihoods = load_csv(data_dir / "likelihood_assumptions.csv")
    cost_scenarios = load_csv(data_dir / "cost_scenarios.csv")

    prior_a, prior_b = 2, 18
    prior_mean = beta_prior_mean(prior_a, prior_b)
    prior_ci_lower, prior_ci_upper = beta_prior_ci(prior_a, prior_b, alpha=0.05)

    rows = []
    for _, row in signals.iterrows():
        s1 = int(row["S1_industrial_action"])
        s2 = float(row["S2_congestion_index"])
        s3 = int(row["S3_logistics_advisory"])

        l_s1_r1, l_s1_r0 = get_categorical_likelihood(
            signal="S1_industrial_action",
            state_value=s1,
            likelihood_df=likelihoods,
        )
        l_s3_r1, l_s3_r0 = get_categorical_likelihood(
            signal="S3_logistics_advisory",
            state_value=s3,
            likelihood_df=likelihoods,
        )

        # S2 is a constructed scaled congestion proxy (not an official port index).
        l_s2_r1 = scaled_beta_pdf(s2, a=6, b=2)
        l_s2_r0 = scaled_beta_pdf(s2, a=2, b=6)

        likelihood_r1 = l_s1_r1 * l_s2_r1 * l_s3_r1
        likelihood_r0 = l_s1_r0 * l_s2_r0 * l_s3_r0
        if likelihood_r0 <= 0:
            raise ValueError("likelihood_R0 must be positive to compute likelihood ratio.")
        likelihood_ratio = likelihood_r1 / likelihood_r0

        posterior_independent = bayesian_posterior(prior_mean, likelihood_r1, likelihood_r0)

        # Illustrative dependence adjustment only; this is NOT an estimated copula.
        dependence_factor_r1 = 0.85
        dependence_factor_r0 = 0.95
        adjusted_likelihood_r1 = likelihood_r1 * dependence_factor_r1
        adjusted_likelihood_r0 = likelihood_r0 * dependence_factor_r0
        posterior_adjusted = bayesian_posterior(
            prior_mean,
            adjusted_likelihood_r1,
            adjusted_likelihood_r0,
        )

        rows.append(
            {
                "event_day": int(row["event_day"]),
                "date_label": row["date_label"],
                "S1_industrial_action": s1,
                "S2_congestion_index": s2,
                "S3_logistics_advisory": s3,
                "prior_mean": prior_mean,
                "prior_ci_lower": prior_ci_lower,
                "prior_ci_upper": prior_ci_upper,
                "likelihood_R1": likelihood_r1,
                "likelihood_R0": likelihood_r0,
                "likelihood_ratio": likelihood_ratio,
                "posterior_independent": posterior_independent,
                "posterior_dependence_adjusted": posterior_adjusted,
            }
        )

    results = pd.DataFrame(rows).sort_values("event_day")
    results.to_csv(outputs_dir / "table_bayesian_results.csv", index=False)

    base = cost_scenarios[cost_scenarios["scenario"] == "base"]
    if base.empty:
        raise ValueError("Base scenario is required in cost_scenarios.csv")
    base_row = base.iloc[0]
    base_threshold = decision_threshold(
        action_cost=float(base_row["action_cost"]),
        disruption_loss=float(base_row["disruption_loss"]),
        action_effectiveness=float(base_row["action_effectiveness"]),
    )

    plt.figure(figsize=(8, 5))
    plt.plot(results["event_day"], results["posterior_independent"], marker="o", label="Posterior (independent)")
    plt.plot(
        results["event_day"],
        results["posterior_dependence_adjusted"],
        marker="o",
        linestyle="--",
        label="Posterior (dependence-adjusted, illustrative)",
    )
    plt.axhline(base_threshold, linestyle=":", label="Base decision threshold")
    plt.xlabel("event_day")
    plt.ylabel("posterior disruption probability")
    plt.title("Event-based Bayesian risk update: Antwerp-Bruges illustration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir / "posterior_plot.png", dpi=200)
    plt.close()

    print("Created outputs/table_bayesian_results.csv and outputs/posterior_plot.png")


if __name__ == "__main__":
    main()
