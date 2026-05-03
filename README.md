# PaySim Fraud Triage — Chain-Aware Mobile-Money Fraud Detection

**Course / research-style ML project** on the synthetic **PaySim** mobile-money dataset: exploratory analysis, **leakage-aware** feature design, **no-chain vs chain-aware** model comparison, **probability calibration**, **cost-sensitive** triage rules, and a **Streamlit** deployment for interactive and batch scoring.

![Chain-aware fraud detection & triage framework (split-safe & interpretable)](assets/ml_framework_chain_aware.png)

![End-to-end ML flowchart — no-chain vs chain-aware pipeline through calibration & Streamlit demo](assets/ml_flow_diagram_chain_aware.png)

*Above: proposed solution infographic, then detailed flowchart — PaySim → EDA → stratified split → split-safe chain discovery & features → preprocessing → model comparison → **calibration** (reference export: **`catboost_plain_sigmoid`**) → triage; plus validation (SHAP/LIME, bootstrap CI, error analysis, drift/PSI) and Streamlit deployment.*

---

## Abstract (problem & goal)

PaySim simulates digital payment flows with extreme **class imbalance** (fraud is rare). The goal is not only high discrimination (e.g. PR-AUC) but **decision-ready** outputs: calibrated fraud probabilities and a **three-way triage** (approve / review / block) aligned with asymmetric **false-positive vs false-negative** costs. This repository implements that pipeline end-to-end and documents **why** each modeling choice was made.

---

## Contributions & novelty

1. **Chain-aware behavioral features (domain-motivated)**  
   Fraud in mobile money often involves **multi-hop** patterns (e.g. **TRANSFER** followed by **CASH_OUT**). We encode this without using the label by grouping rows on **`(step, amount)`**, counting co-occurring types, and defining:
   - **`chain_size`**: number of rows in the `(step, amount)` group.
   - **`is_chain_member`**: group contains both **TRANSFER** and **CASH_OUT**, with **`chain_size` ≤ 12** (`CHAIN_SIZE_CAP`) to avoid treating huge accidental collisions as “chains.”  

   These features are **not** derived from `isFraud`; they are structural signals aligned with known PaySim fraud narratives.

2. **Controlled A/B: no-chain vs chain-aware**  
   In `01_eda_paysim.ipynb` (**§12**), we train and evaluate the same model families **with** and **without** chain columns, producing an explicit **`*_no_chain` vs `*_chain`** comparison table (e.g. PR-AUC, recall). That supports the claim that chain features **help** under the same split and preprocessing, rather than ad-hoc tuning.

3. **Calibration + triage, not raw scores**  
   Boosted/tree models can be **poorly calibrated**. The notebook calibrates **finalists** (RF, CatBoost, XGBoost, …). In the **documented full run**, **`catboost_plain_sigmoid`** (calibrated CatBoost) wins; **`FINAL_MODEL_KEY`** always records whichever finalist actually wins your run. The notebook exports **`artifacts/feature_metadata.json`**, and the app scores with that **calibrated** pipeline. **`build_artifacts.py`** alone stays within **RF** calibration variants for a lightweight fallback. Thresholds for **GREEN / YELLOW / RED** come from exported metadata.

4. **Cost-aware policy transparency**  
   The UI surfaces a simple cost model (**false-positive cost = 5**, **false-negative cost = 500**) so reviewers see that thresholds reflect **business asymmetry**, not arbitrary cutoffs.

5. **Leakage-aware baseline design**  
   We **drop** `isFlaggedFraud` from features (rule-like flag aligned with fraud), **drop** high-cardinality IDs (`nameOrig`, `nameDest`) for the tabular baseline, and use **`log_amount`** instead of raw **`amount`** in the feature matrix to avoid redundant scaling signals. Post-transaction balances are kept for this academic setting but **flagged** in the notebook as an **operational caveat** for real-time deployment.

---

## Methodology (how we decided)

| Stage | Decision | Rationale |
|--------|-----------|-----------|
| **Split** | Stratified train/test (`test_size=0.2`, fixed `RANDOM_STATE`) | Preserve rare fraud rate in both sets; reproducibility. |
| **Preprocessing** | `ColumnTransformer`: `StandardScaler` on numeric, `OneHotEncoder(handle_unknown="ignore")` on `type` | Linearly sensitive models need scaling; trees still receive consistent numeric inputs; unknown categories at inference. |
| **Engineered numeric features** | `orig_delta`, `dest_delta`, `orig_residual`, zero-balance flags, `log_amount` | Captures balance consistency and scale skew; documented in EDA. |
| **Chain features** | Groupby `(step, amount)` + TRANSFER ∧ CASH_OUT + cap | Domain pattern; cap limits noise from massive groups. |
| **Models compared** | LR, RF, XGBoost, LightGBM, CatBoost (optional), BRF, GNB, … (notebook §12) | Same chain-aware pipeline; finalists vary by run. |
| **Final scorer** | **Reference export:** **`catboost_plain_sigmoid`** (calibrated CatBoost) via **`final_model_key`** + **`calibration_model_file_map`**. Optional **`build_artifacts.py`**: RF-family calibration only. | App loads metadata-driven scorer; **Tree SHAP** targets the **deploy model’s inner tree** when supported (**CatBoost**/XGB/RF unwrap). **`rf_plain_base.joblib`** remains a packaged **fallback** only. |
| **Triage** | Three-way **GREEN / YELLOW / RED** from exported thresholds (`review_threshold`, `block_threshold`, `moderate_cutoff` in `feature_metadata.json`) | Values depend on your notebook §12.9b run; chain escalation rule matches that export. |

### Additional methods added for robustness / clarity

- **SMOTE (optional)** in the notebook experiments to mitigate extreme class imbalance during training.
- **Class-weight (optional)** in the notebook experiments to compare cost-sensitive learning behavior vs SMOTE.
- **LIME vs SHAP (notebook-only)** local explanation cross-check for one blocked transaction:
  - SHAP is model-faithful local attribution (primary explanation).
  - LIME is a perturbation-based surrogate (used only as an academic robustness comparison).
- **Bootstrap PR-AUC CI (notebook-only)** after final calibration to quantify uncertainty of PR-AUC on the untouched test split.
- **Error Analysis (notebook-only)** section on false negatives and false positives for the final deployed/demo policy, including compact tables and grouped summaries.

**Important methodology note (academic honesty):** in `01_eda_paysim.ipynb` and `build_artifacts.py`, `chain_size` and `is_chain_member` are computed **after** the stratified train/test split, **separately on training vs test rows** (so the test set does not inform training-side group statistics). A **production** system would still materialize chain state from **transaction history up to decision time** in a streaming-safe way; the Streamlit **manual** path can use fallback chain fields as documented in the UI.

---

## Results & evidence

Quantitative metrics (**PR-AUC**, confusion matrices, ROC, calibration curves, **SHAP** where run) live in **`01_eda_paysim.ipynb`** after the training cells. **Figures below** illustrate the Streamlit prototype and triage story; cite the notebook for tables and plots used in your report.

The Streamlit app has 5 tabs:
- **Command Center:** deployment snapshot, policy-at-a-glance logic, and quick navigation overview for the demo.
- **Dashboard:** single-transaction scoring + triage decision panel with SHAP-driven local risk drivers (optional SHAP visual).
- **Batch upload:** upload a CSV to score many transactions + bucket summary (SHAP disabled for speed).
- **Drift Monitor (monitoring-only):** early vs late windows (`step <= 400` vs `step > 400`) with feature drift (PSI table + summary chart) and PR-AUC early/late comparison; **no retraining**.
- **Model Card:** shows `MODEL_CARD.md` (deployed system documentation: model, calibration, triage policy, drift monitoring, and limitations).

### Optional local LLM explanation layer (Ollama)

If enabled in Streamlit, the LLM is used only as an **analyst-style explanation assistant** after prediction is complete.

**Academic honesty note:**  
**Analyst summary is generated by a local LLM for explanation only. Fraud score and final action come from the calibrated ML pipeline.**

Pipeline placement:

`transaction input → preprocessing → calibrated finalist (e.g. `catboost_plain_sigmoid`) → calibrated fraud probability + triage bucket → Tree SHAP on the **same inner booster** when explodable (else `rf_plain_base` fallback) → reasons → optional LLM analyst summary`

This keeps the LLM strictly in the explanation layer; it does not change model training, calibration, thresholds, or triage decisions.

---

## Figure gallery

### A) Streamlit app screens

The prototype has **five tabs**: Command Center, Dashboard, Batch upload, Drift Monitor, and Model Card. Key screens are shown below (batch scoring still uses CSV upload + bucket summary as in deployment).

**Command Center** — deployment snapshot, hero banner with decision pipeline, policy at a glance, and quick orientation:
![Command Center tab](assets/streamlit_tab_command_center.png)

**Dashboard** — manual transaction scoring, triage panel, SHAP explanations, and optional analyst summary:
![Dashboard tab](assets/streamlit_tab_dashboard.png)

**Batch upload** — CSV upload + bucket summary (fast scoring without SHAP):
![Batch upload tab](assets/image-3d6b6a66-2581-46c5-b3dd-a92d91142cf3.png)

**Drift Monitor** — PSI feature drift table, summary chart, optional PR-AUC early vs late comparison (monitoring-only):
![Drift Monitor tab](assets/streamlit_tab_drift_monitor.png)

**Model Card** — renders `MODEL_CARD.md` with intended use, data, thresholds, and limitations:
![Model Card tab](assets/streamlit_tab_model_card.png)

### B) Notebook outputs (modeling evidence)

**Reliability / calibration comparison (RF vs XGB; uncalibrated/sigmoid/isotonic):**
![Calibration reliability plots](assets/image-1b21f940-7778-4311-8abb-f61e53b0b090.png)

**Threshold sweep and selected operating threshold:**
![Threshold sweep table](assets/image-adc02d3a-d3db-4ed4-bb25-fde25313d4df.png)

**Triage buckets after chain-escalation policy:**
![Triage bucket distribution](assets/image-4f8a7d39-62e6-4cda-a5de-e9695ef46ee3.png)

**SHAP global summary (feature impact):**
![SHAP summary](assets/image-df7adac4-79f9-4057-a503-cda45689f81c.png)

---

## Tech stack

| Area | Tools |
|------|--------|
| Language | Python 3 |
| Analysis | Jupyter, pandas, numpy |
| ML | scikit-learn, joblib; notebook also uses **imbalanced-learn (SMOTE)** where configured, **XGBoost**, **LightGBM**, **CatBoost** (optional), **SHAP**, plotting libraries |
| App | Streamlit |
| Data | PaySim CSV (`PS_20174392719_1491204439457_log.csv`) — **gitignored by default** (large) |

**Install (app only):**

```bash
pip install -r requirements.txt
```

`requirements.txt` already includes **SHAP** (used in both Streamlit and the notebook).

**Notebook extras (install as needed — LIME is notebook-only):**

```bash
pip install jupyter matplotlib seaborn imbalanced-learn xgboost lightgbm catboost lime
```

---

## Repository layout

```
Group_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── build_artifacts.py
├── artifacts/                 # preprocessor, calibrated joblib(s), optional RF stub for SHAP fallback, feature_metadata.json (see below)
├── assets/                    # banner SVG + README screenshots
├── 01_eda_paysim.ipynb
├── MODEL_CARD.md
└── PS_20174392719_1491204439457_log.csv   # local only unless you use LFS / remove gitignore
```

---

## Quick start

```bash
cd Group_Project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place **`PS_20174392719_1491204439457_log.csv`** in the project root (see [PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) or your course mirror), then:

```bash
python build_artifacts.py
streamlit run app.py
```

### Streamlit, artifacts, and “dynamic” scoring

`app.py` **does not train** models. It loads:

- **`artifacts/preprocessor_paysim.joblib`** — same transforms as the notebook.
- **`artifacts/feature_metadata.json`** — **`final_model_key`** plus **`calibration_model_file_map`**. The **checked-in reference export** uses **`catboost_plain_sigmoid`** → `catboost_plain_sigmoid_calibrated.joblib`; another rerun may list XGBoost or RF variants instead.
- **The calibrated pipeline** file named by that map — this is the **fraud probability** used for triage on Dashboard, Batch, and Drift Monitor.
- **`artifacts/rf_plain_base.joblib`** — **backup** tree for **Tree SHAP** / uncalibrated “tree probability” if the deploy object can’t be explained in your environment; **normally** the app unwraps **CatBoost/XGB/RF** from the calibrated pipeline first.

**Two ways to populate `artifacts/`**

1. **Recommended (full notebook zoo):** run **`01_eda_paysim.ipynb`** through **§12.9b** and execute the **artifact export** cell (writes `feature_metadata.json`, all relevant `*_calibrated.joblib` files, and keeps maps aligned with CatBoost / XGB / RF).
2. **Quick RF-only fallback:** run **`python build_artifacts.py`** — trains RF + RF calibration variants only; fine for a fast demo, **not** a drop-in replacement for a CatBoost-led notebook export.

**Auto-rebuild on Streamlit start** (`AUTO_REBUILD_IF_STALE`, default on): if the **notebook file** is newer than artifact mtimes, the app runs **`build_artifacts.py`** only — it does **not** execute Jupyter. After a real modeling change, **export from the notebook** (or run `build_artifacts.py` intentionally) so `feature_metadata.json` matches the models you care about.

---

## Batch CSV (production-style inputs)

Required columns:

`step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, **`chain_size`**, **`is_chain_member`**

Batch scoring assumes chain fields are **precomputed** (mirroring offline EDA). Manual demo mode can use **fallback** chain values; the UI states this limitation.

---

## Limitations (for reports & defense)

- **Synthetic data:** PaySim is not live production traffic; generalization claims must be qualified.  
- **Chain timing:** offline evaluation uses **split-safe** chain features (per split after `train_test_split`); a live system still needs **streaming-safe** chain state from history at decision time.  
- **Post-transaction balances:** available in the dataset; real-time systems may not have the same fields at decision time.  
- **YELLOW bucket:** with current calibration and scores, moderate-risk rows can be sparse; triage logic is still correct and documented.

---

## Data note

`PS_20174392719_1491204439457_log.csv` is large and gitignored by default. Use Git LFS only if you want to version the raw file.
