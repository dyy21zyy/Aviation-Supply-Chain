# aviation_sideinfo_case

## 1) Project title
**Event-based Bayesian side-information illustration for aviation engine-component supply disruption risk**  
Case context: *2025 Antwerp-Bruges harbour pilots industrial action and potential knock-on disruption risk for aviation-related component flows.*

## 2) Research purpose
This repository provides a **small, reproducible numerical illustration** of how manually coded side-information signals can update disruption-risk beliefs under Bayes' rule and then map those beliefs into cost-sensitive resilience decisions.

## 3) Important disclaimer
This project is an **event-based numerical illustration**, **not** a full empirical validation.

- Signal values are **manually coded illustrative variables** motivated by publicly reported event evidence.
- The codebase does **not** use proprietary aviation shipment data.
- The codebase does **not** scrape websites and does **not** call APIs.
- Results should be interpreted as methodological demonstration, not production forecasting.

## 4) Data description
All input data are generated locally by `src/01_construct_case_data.py`.

1. `data/antwerp_case_signals.csv`
   - Event timeline with coded side-information signals.
   - Includes:
     - `S1_industrial_action` (ordinal: 0/1/2)
     - `S2_congestion_index` (0-100) as a **constructed scaled congestion proxy**, not an official port index
     - `S3_logistics_advisory` (binary)
     - `disruption_state` (binary illustrative event-state marker)

2. `data/likelihood_assumptions.csv`
   - Categorical likelihood assumptions for `S1` and `S3` under disruption vs no disruption.

3. `data/cost_scenarios.csv`
   - Cost and effectiveness scenarios used for decision-threshold calculations.

## 5) Method
- Prior disruption risk: 
  \[
  \pi \sim \text{Beta}(2,18),\quad E[\pi]=0.10
  \]
- Likelihood model:
  - Categorical likelihoods for `S1_industrial_action` and `S3_logistics_advisory`
  - Beta likelihood for scaled congestion proxy `S2/100`
    - \(S2|R=1 \sim \text{Beta}(6,2)\)
    - \(S2|R=0 \sim \text{Beta}(2,6)\)
- Posterior update uses Bayes rule for each event day.
- Optional dependence-adjusted posterior multiplies the independent likelihood with fixed factors (illustrative correction only, **not** an estimated Gaussian copula).
- Cost-sensitive decision threshold:
  \[
  \tau = \frac{C_A}{e\,C_D}
  \]
  where action is recommended when posterior risk exceeds \(\tau\).

## 6) How to run
From the project root (`aviation_sideinfo_case/`):

```bash
python src/01_construct_case_data.py
python src/02_bayesian_update.py
python src/03_decision_threshold.py
python src/04_sensitivity_analysis.py
```

## 7) Output files
- `outputs/table_event_timeline.csv`  
  Copy of the illustrative event timeline and coded signals.
- `outputs/table_bayesian_results.csv`  
  Prior summary, likelihood terms, likelihood ratio, independent posterior, and dependence-adjusted posterior by event day.
- `outputs/posterior_plot.png`  
  Posterior trajectories over event day with base decision threshold line.
- `outputs/table_decision_results.csv`  
  Scenario-by-event expected loss comparison and act/do-not-act decisions.
- `outputs/table_sensitivity.csv`  
  Earliest action day under scenario and grid-based parameter sensitivity.

## 8) Interpretation notes for journal use
- Treat this as a **transparent numerical thought experiment** showing the workflow from public-event evidence to decision support.
- Emphasize that the values are manually coded to illustrate Bayesian side-information mechanics.
- Use the sensitivity table to discuss how intervention timing depends on action cost, potential disruption loss, and mitigation effectiveness.

## 9) Limitations
- Single-event illustration
- Manually coded signals
- No estimated copula (only illustrative dependence adjustment)
- No XGBoost benchmark
- No claim of causal identification
- Full validation would require a multi-event panel dataset with richer operational outcomes

## Public-event motivation (context only)
The coding scheme is motivated by publicly reported facts about the 2025 Antwerp-Bruges harbour pilots industrial action, including reports that the port often handles roughly 60-80 vessels/day but processed 31 on an affected day, along with reports of vessel delays/diversions, carrier advisories, and persistent vessel queues with reduced operating capacity. These points are used only to motivate side-information design; they are not ingested as automated raw feeds.
