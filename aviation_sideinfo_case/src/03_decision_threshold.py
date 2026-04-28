import pandas as pd

from utils import (
    decision_threshold,
    ensure_directories,
    expected_loss_act,
    expected_loss_no_act,
    load_csv,
)


def main() -> None:
    _, data_dir, outputs_dir = ensure_directories()

    bayes_results = load_csv(outputs_dir / "table_bayesian_results.csv").sort_values("event_day")
    cost_scenarios = load_csv(data_dir / "cost_scenarios.csv")

    rows = []
    for _, scenario_row in cost_scenarios.iterrows():
        scenario = scenario_row["scenario"]
        c_a = float(scenario_row["action_cost"])
        c_d = float(scenario_row["disruption_loss"])
        eff = float(scenario_row["action_effectiveness"])

        threshold = decision_threshold(c_a, c_d, eff)

        for _, event_row in bayes_results.iterrows():
            p = float(event_row["posterior_dependence_adjusted"])
            loss_act = expected_loss_act(p, c_a, c_d, eff)
            loss_no_act = expected_loss_no_act(p, c_d)
            decision = "act" if p > threshold else "do_not_act"
            net_expected_benefit = loss_no_act - loss_act

            rows.append(
                {
                    "scenario": scenario,
                    "event_day": int(event_row["event_day"]),
                    "date_label": event_row["date_label"],
                    "posterior_dependence_adjusted": p,
                    "threshold": threshold,
                    "decision": decision,
                    "expected_loss_act": loss_act,
                    "expected_loss_no_act": loss_no_act,
                    "net_expected_benefit": net_expected_benefit,
                }
            )

    decision_results = pd.DataFrame(rows).sort_values(["scenario", "event_day"])
    decision_results.to_csv(outputs_dir / "table_decision_results.csv", index=False)
    print("Created table_decision_results.csv")


if __name__ == "__main__":
    main()
