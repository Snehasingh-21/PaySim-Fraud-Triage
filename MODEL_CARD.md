# Model Card: PaySim Fraud Triage (Chain-Aware RF + Dynamic RF-Family Calibration)

## 1. Summary
This model is designed to triage **synthetic mobile-money transactions** from the **PaySim** dataset into a three-way policy:

- **GREEN (ALLOW)**: low fraud risk
- **YELLOW (REVIEW)**: uncertain / potentially fraudulent
- **RED (BLOCK)**: high fraud risk (or escalated fraud risk under chain context)

The deployed score is a **calibrated fraud probability** from the final RF-family calibration path selected for that run. Calibrated probabilities are used to apply an asymmetric, cost-sensitive decision policy with explicit operational thresholds.

## 2. Intended use
Use this model for **decision support** in an educational / prototype setting:

1. Compute a **calibrated probability** that a transaction is fraud.
2. Convert the probability into a **triage bucket** (ALLOW / REVIEW / BLOCK).
3. Provide **local explainability** (SHAP) to support human review.

In this repository, the Streamlit app implements the triage policy and surfaces local evidence for each prediction.

## 3. Not intended for
This model is **not** intended for:

- Automated, fully autonomous blocking in a real production system without additional operational controls.
- Claims of real-world accuracy on live traffic. The underlying data is synthetic and should be treated as a research/teaching artifact.

## 4. Data
**Dataset:** PaySim synthetic mobile-money transactions  
**Scale (as reported in the project):** ~6.3M transactions with extreme class imbalance (~0.13% fraud rate).

## 5. Features and chain-aware context
The feature set includes:

- Timing / numeric structure: `step`, `log_amount`
- Balance-consistency signals:
  - `orig_delta = oldbalanceOrg - newbalanceOrig` (sender balance change)
  - `dest_delta = newbalanceDest - oldbalanceDest` (receiver balance change)
  - `orig_residual = orig_delta - amount`
  - zero-balance flags (sender/receiver before/after)
- Transaction type: categorical `type` (one-hot encoded)
- **Chain-aware features** (domain-motivated; computed without using the label):
  - `chain_size`: group size under `(step, amount)` with a **cap** (`CHAIN_SIZE_CAP = 12`)
  - `is_chain_member`: group contains both `TRANSFER` and `CASH_OUT` within the capped chain size

Important modeling note (academic honesty):
In the **research notebook** and in **`build_artifacts.py`**, `chain_size` and `is_chain_member` are computed **after** the train/test split, **separately within the training and test rows** (so test rows do not inform training-side chain group statistics). A **production** system must still compute chain state in a **streaming-safe** way from transaction history up to decision time; the app’s **manual/batch** modes may use precomputed or fallback chain fields as documented in the UI.

## 6. Model
Base model:
- **Random Forest** classifier (RF)

Model lineage (to avoid confusion):
In the notebook we compare multiple model families (including Logistic Regression and XGBoost) to show *why* chain features and calibration help. However, the **deployed base estimator** used by the app is fixed to **RF**; calibration choice is selected dynamically inside RF-family variants (uncalibrated/sigmoid/isotonic) from notebook calibration results.

Calibration:
- **Dynamic RF-family calibration selection** (Brier-first, then PR-AUC, then ROC-AUC tie-break) among:
  - `rf_plain_uncalibrated`
  - `rf_plain_sigmoid`
  - `rf_plain_isotonic`

Deployment output:
- `P(fraud=1 | x)` after the selected RF-family calibration path for that run

## 7. Decision policy (cost-sensitive triage)
The final action is derived from calibrated probability and the chain escalation rule.

Policy parameters (**from notebook export**, also in `artifacts/feature_metadata.json`):

- false positive cost = **5**
- false negative cost = **500**
- `review_threshold` = **0.05**
- `block_threshold` = **0.25**
- `moderate_cutoff` = **0.05**
- `operating_threshold` (cost-optimal scalar search) = **0.15**
- `chain_size_cap` = **12**

Triage rule (3-way):

- If `p < 0.05` → **GREEN**
- Else if `0.05 <= p < 0.25` → **YELLOW**
- Else `p >= 0.25` → **RED**

Chain escalation rule:

- If **chain signal is active** (`is_chain_member = 1`) and `p >= moderate_cutoff` (0.05),
  then the bucket can be escalated to **RED** (even when not already in the `p >= 0.25` band).

## 8. Evaluation and evidence
### Key metrics (deployed model) + supporting evidence

Values below reflect **the same notebook run** that exported `feature_metadata.json` (selected scorer: **`rf_plain_sigmoid`**).

| Metric | Value |
|--------|-------|
| PR-AUC (test, calibrated RF) | 0.998508 |
| Brier Score | 3.276363e-06 |
| ROC-AUC | 0.999645 |
| Fraud captured in RED (before → after escalation) | 99.76% → 99.76% |
| Legitimate allowed (GREEN) | 100.000% |
| Logistic Regression ΔPR-AUC (chain vs no-chain; notebook §12.5) | +0.010 |

**Bootstrap PR-AUC (95% CI, notebook §12.9a):**

| Statistic | Value |
|----------|-------|
| PR-AUC point estimate | 0.9985080292924411 |
| Bootstrap mean PR-AUC | 0.9984951330558315 |
| 95% CI lower | 0.9966562712663735 |
| 95% CI upper | 0.9998761515900116 |
| Valid bootstrap runs | 300 / 300 |

### Model comparison context (notebook evidence, not deployment)

The notebook comparison set now includes **Logistic Regression, Random Forest, XGBoost, and LightGBM** under the same chain-aware setup and reporting pipeline.  
This comparison is included to justify model-selection decisions; it does **not** change the deployed decision-maker in this project.

Deployed decision-maker remains:
- **Random Forest + selected RF-family calibration path** for fraud probability and triage bucket.

### Bootstrap uncertainty for PR-AUC (95% CI)

The notebook computes a **bootstrap 95% confidence interval** for the deployed scorer’s **PR-AUC** on the *untouched test set* (Section `12.9a`), keyed to **`FINAL_MODEL_KEY`** for that run. The rows below are a **legacy PDF snapshot** (isotonic-aligned export before dynamic selection emphasized sigmoid here); rerun §12.9a after your current `FINAL_MODEL_KEY` and replace this table—or omit until refreshed.

| Statistic | Value (legacy snapshot; refresh after rerun) |
|----------|-------|
| PR-AUC point estimate | 0.998239 |
| Bootstrap mean PR-AUC | 0.998200 |
| 95% CI lower bound | 0.996213 |
| 95% CI upper bound | 0.999633 |
| Valid bootstrap runs used | 300 / 300 |

**Optional (from your screenshot):** if you want the CI numbers/table as an image, place your histogram/table image into `assets/` and embed it only when the file is present (to avoid broken placeholders in the app view).

### Evidence in the notebook

The notebook (`01_eda_paysim.ipynb`) contains further evidence for:

- Discrimination metrics (including PR-AUC for imbalanced evaluation)
- No-chain vs chain-aware A/B comparison
- Calibration comparison
- Local and global interpretability experiments (SHAP)
- Local robustness cross-check (SHAP vs LIME) for one blocked case
- Final error analysis section on **false negatives** and **false positives** (counts, rates, top examples, grouped summaries)
- A monitoring-only drift analysis (PSI and PR-AUC delta over time windows)

## 9. Explainability
Primary deployed explanation:
- **Local SHAP**: per-transaction feature attribution used to produce human-readable risk drivers.

Notebook robustness check:
- **LIME vs SHAP (local comparison)** is included to demonstrate agreement on key drivers (not used in the deployed app).

Optional UI explanation layer (Streamlit):
- A local **Ollama-based LLM analyst summary** can be generated on demand to translate model evidence into short analyst-style text.
- This LLM output is **explanation-only** and is never used to compute probability, bucket, thresholding, calibration, or final action.
- Academic honesty statement: **Fraud score and final action come from the calibrated ML pipeline (RF + selected RF-family calibration), not from the LLM.**

## 10. Drift monitoring (monitoring-only)
The Streamlit “Drift Monitor” tab implements monitoring-only behavior (no retraining):

- Early window: `step <= 400`
- Late window: `step > 400`
- Feature drift metric: **PSI** using repository-defined thresholds:
  - PSI < 0.10 -> **GREEN**
  - 0.10 <= PSI <= 0.20 -> **YELLOW**
  - PSI > 0.20 -> **RED**
- Performance drift metric: PR-AUC early vs late
  - If `abs(delta) > 0.05` -> recommend retraining (warning message)
  - Else -> model considered stable (success message)

## 11. Limitations
- Synthetic data: generalization to real production traffic is not guaranteed.
- Chain state construction differs between offline research and a streaming production pipeline (documented above).
- Some operational signals (e.g., post-transaction balances) may be unavailable at true decision time in a real system.
- Calibration and thresholds are tuned for this repository's pipeline and should be revalidated if the data pipeline changes.

## 12. Versioning / artifacts
The deployed artifacts are loaded from `artifacts/`:
- preprocessor
- RF base model
- calibrated model (selected RF-family path for the built artifact set)
- feature metadata (thresholds and feature names)

If you rebuild artifacts using `build_artifacts.py`, the Streamlit app should remain consistent because it reads thresholds and feature mappings from `feature_metadata.json`.

You can instead run the **`01_eda_paysim.ipynb` §12.9b tail cell**, which writes the same artifact directory from the live notebook session and refreshes this model card’s §7–§8 block (everything up to **“### Model comparison context …”**) so it matches those exports.

