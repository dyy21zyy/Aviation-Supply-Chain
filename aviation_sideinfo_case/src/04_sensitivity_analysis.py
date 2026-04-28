from pathlib import Path

import pandas as pd

from utils import decision_threshold, load_csv


def earliest_action_for_threshold(events: pd.DataFrame, threshold: float):
    eligible = events[events["posterior_dependence_adjusted"] > threshold].sort_values("event_day")
    if eligible.empty:
        return "no_action", "no_action"
    first = eligible.iloc[0]
    return int(first["event_day"]), first["date_label"]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    data_dir = project_root / "data"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    bayes_results = load_csv(outputs_dir / "table_bayesian_results.csv").sort_values("event_day")
    cost_scenarios = load_csv(data_dir / "cost_scenarios.csv")

    rows = []

    # Scenario-based sensitivity summary.
    for _, sc in cost_scenarios.iterrows():
        threshold = decision_threshold(
            float(sc["action_cost"]),
            float(sc["disruption_loss"]),
            float(sc["action_effectiveness"]),
        )
        day, label = earliest_action_for_threshold(bayes_results, threshold)
        rows.append(
            {
                "action_cost": float(sc["action_cost"]),
                "disruption_loss": float(sc["disruption_loss"]),
                "action_effectiveness": float(sc["action_effectiveness"]),
                "threshold": threshold,
                "earliest_action_day": day,
                "earliest_action_label": label,
            }
        )

    # Grid sensitivity.
    action_cost_grid = [100000, 200000, 300000, 500000]
    disruption_loss_grid = [1000000, 3000000, 5000000, 10000000]
    effectiveness_grid = [0.3, 0.5, 0.7, 1.0]

    for c_a in action_cost_grid:
        for c_d in disruption_loss_grid:
            for eff in effectiveness_grid:
                threshold = decision_threshold(c_a, c_d, eff)
                day, label = earliest_action_for_threshold(bayes_results, threshold)
                rows.append(
                    {
                        "action_cost": float(c_a),
                        "disruption_loss": float(c_d),
                        "action_effectiveness": float(eff),
                        "threshold": threshold,
                        "earliest_action_day": day,
                        "earliest_action_label": label,
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(outputs_dir / "table_sensitivity.csv", index=False)
    print("Created outputs/table_sensitivity.csv")


if __name__ == "__main__":
    main()
