from pathlib import Path

import pandas as pd

from utils import (
    decision_threshold,
    expected_loss_act,
    expected_loss_no_act,
    load_csv,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    data_dir = project_root / "data"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    bayes_results = load_csv(outputs_dir / "table_bayesian_results.csv").sort_values("event_day")
    cost_scenarios = load_csv(data_dir / "cost_scenarios.csv")

    rows = []
    for _, sc in cost_scenarios.iterrows():
        scenario = sc["scenario"]
        c_a = float(sc["action_cost"])
        c_d = float(sc["disruption_loss"])
        eff = float(sc["action_effectiveness"])
        threshold = decision_threshold(c_a, c_d, eff)

        for _, ev in bayes_results.iterrows():
            p = float(ev["posterior_dependence_adjusted"])
            loss_act = expected_loss_act(p, c_a, c_d, eff)
            loss_no_act = expected_loss_no_act(p, c_d)
            decision = "act" if p > threshold else "do_not_act"
            net_benefit = loss_no_act - loss_act

            rows.append(
                {
                    "scenario": scenario,
                    "event_day": int(ev["event_day"]),
                    "date_label": ev["date_label"],
                    "posterior_dependence_adjusted": p,
                    "threshold": threshold,
                    "decision": decision,
                    "expected_loss_act": loss_act,
                    "expected_loss_no_act": loss_no_act,
                    "net_expected_benefit": net_benefit,
                }
            )

    out = pd.DataFrame(rows).sort_values(["scenario", "event_day"])
    out.to_csv(outputs_dir / "table_decision_results.csv", index=False)
    print("Created outputs/table_decision_results.csv")


if __name__ == "__main__":
    main()
