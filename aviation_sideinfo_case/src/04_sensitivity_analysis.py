import pandas as pd

from utils import decision_threshold, ensure_directories, load_csv


def earliest_action(events: pd.DataFrame, threshold: float):
    eligible = events[events["posterior_dependence_adjusted"] > threshold].sort_values("event_day")
    if eligible.empty:
        return "no_action", "no_action"
    first = eligible.iloc[0]
    return int(first["event_day"]), first["date_label"]


def main() -> None:
    _, _, outputs_dir = ensure_directories()

    bayes_results = load_csv(outputs_dir / "table_bayesian_results.csv").sort_values("event_day")

    action_cost_grid = [100000, 200000, 300000, 500000]
    disruption_loss_grid = [1000000, 3000000, 5000000, 10000000]
    effectiveness_grid = [0.3, 0.5, 0.7, 1.0]

    rows = []
    for action_cost in action_cost_grid:
        for disruption_loss in disruption_loss_grid:
            for effectiveness in effectiveness_grid:
                threshold = decision_threshold(action_cost, disruption_loss, effectiveness)
                earliest_day, earliest_label = earliest_action(bayes_results, threshold)
                rows.append(
                    {
                        "action_cost": float(action_cost),
                        "disruption_loss": float(disruption_loss),
                        "action_effectiveness": float(effectiveness),
                        "threshold": threshold,
                        "earliest_action_day": earliest_day,
                        "earliest_action_label": earliest_label,
                    }
                )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(outputs_dir / "table_sensitivity.csv", index=False)
    print("Created table_sensitivity.csv")


if __name__ == "__main__":
    main()
