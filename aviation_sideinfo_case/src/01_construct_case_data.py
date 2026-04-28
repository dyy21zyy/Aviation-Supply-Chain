import pandas as pd

from utils import ensure_directories


def main() -> None:
    _, data_dir, output_dir = ensure_directories()

    # Event-based coded illustration signals inspired by public reporting.
    # S2_congestion_index is a constructed scaled congestion proxy (0-100),
    # not an official port index.
    antwerp_case = pd.DataFrame(
        [
            {
                "event_day": -3,
                "date_label": "early_warning",
                "public_information": "Early public indication of work-to-rule or industrial-action risk among harbour pilots.",
                "S1_industrial_action": 1,
                "S2_congestion_index": 35,
                "S3_logistics_advisory": 0,
                "disruption_state": 0,
            },
            {
                "event_day": 0,
                "date_label": "disruption_reported",
                "public_information": "Industrial action materially disrupts port operations; reported vessel handling falls below normal level.",
                "S1_industrial_action": 2,
                "S2_congestion_index": 75,
                "S3_logistics_advisory": 0,
                "disruption_state": 1,
            },
            {
                "event_day": 1,
                "date_label": "carrier_advisory",
                "public_information": "Carrier or logistics advisory reports expected delays or operational disruption.",
                "S1_industrial_action": 2,
                "S2_congestion_index": 82,
                "S3_logistics_advisory": 1,
                "disruption_state": 1,
            },
            {
                "event_day": 7,
                "date_label": "backlog_persistent",
                "public_information": "Backlog remains elevated; public reports indicate many vessels waiting and reduced operating capacity.",
                "S1_industrial_action": 2,
                "S2_congestion_index": 95,
                "S3_logistics_advisory": 1,
                "disruption_state": 1,
            },
        ]
    )

    likelihood_assumptions = pd.DataFrame(
        [
            {
                "signal": "S1_industrial_action",
                "state_value": 0,
                "likelihood_given_disruption": 0.10,
                "likelihood_given_no_disruption": 0.80,
                "explanation": "Event-based coded illustration: no warning/action is uncommon under disruption.",
            },
            {
                "signal": "S1_industrial_action",
                "state_value": 1,
                "likelihood_given_disruption": 0.30,
                "likelihood_given_no_disruption": 0.17,
                "explanation": "Event-based coded illustration: warning/tension can occur in both states.",
            },
            {
                "signal": "S1_industrial_action",
                "state_value": 2,
                "likelihood_given_disruption": 0.60,
                "likelihood_given_no_disruption": 0.03,
                "explanation": "Event-based coded illustration: confirmed action is more likely under disruption.",
            },
            {
                "signal": "S3_logistics_advisory",
                "state_value": 0,
                "likelihood_given_disruption": 0.35,
                "likelihood_given_no_disruption": 0.92,
                "explanation": "Event-based coded illustration: no advisory is more likely with no disruption.",
            },
            {
                "signal": "S3_logistics_advisory",
                "state_value": 1,
                "likelihood_given_disruption": 0.65,
                "likelihood_given_no_disruption": 0.08,
                "explanation": "Event-based coded illustration: advisory is more likely under disruption.",
            },
        ]
    )

    cost_scenarios = pd.DataFrame(
        [
            {
                "scenario": "base",
                "action_cost": 200000,
                "disruption_loss": 5000000,
                "action_effectiveness": 0.70,
                "explanation": "Baseline cost-sensitive decision scenario.",
            },
            {
                "scenario": "high_action_cost",
                "action_cost": 500000,
                "disruption_loss": 5000000,
                "action_effectiveness": 0.70,
                "explanation": "Action is more expensive than base.",
            },
            {
                "scenario": "low_effectiveness",
                "action_cost": 200000,
                "disruption_loss": 5000000,
                "action_effectiveness": 0.30,
                "explanation": "Action prevents a smaller share of disruption loss.",
            },
            {
                "scenario": "low_disruption_loss",
                "action_cost": 200000,
                "disruption_loss": 1000000,
                "action_effectiveness": 0.70,
                "explanation": "Disruption consequence is lower than base.",
            },
            {
                "scenario": "severe_disruption_loss",
                "action_cost": 200000,
                "disruption_loss": 10000000,
                "action_effectiveness": 0.70,
                "explanation": "Disruption consequence is higher than base.",
            },
        ]
    )

    antwerp_case.to_csv(data_dir / "antwerp_case_signals.csv", index=False)
    likelihood_assumptions.to_csv(data_dir / "likelihood_assumptions.csv", index=False)
    cost_scenarios.to_csv(data_dir / "cost_scenarios.csv", index=False)

    antwerp_case.to_csv(output_dir / "table_event_timeline.csv", index=False)

    print("Created event-based coded illustration input CSVs and table_event_timeline.csv")


if __name__ == "__main__":
    main()
