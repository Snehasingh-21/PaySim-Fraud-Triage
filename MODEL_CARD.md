# Model Card: PaySim Fraud Triage (Chain-Aware + Calibrated Deploy Scorer)

## 1. Summary
This model is designed to triage **synthetic mobile-money transactions** from the **PaySim** dataset into a three-way policy:

- **GREEN (ALLOW)**: low fraud risk
- **YELLOW (REVIEW)**: uncertain / potentially fraudulent
- **RED (BLOCK)**: high fraud risk (or escalated fraud risk under chain context)

The deployed score is a **calibrated fraud probability** from whichever **`final_model_key`** is stored in **`artifacts/feature_metadata.json`** for that export (notebook §12.9b / `build_artifacts.py`). Calibrated probabilities drive an asymmetric, cost-sensitive triage policy.

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

**Fraud probability (triage):** the Streamlit app loads the **calibrated** scorer keyed by **`final_model_key`** in `feature_metadata.json` (via **`calibration_model_file_map`**). The usual flagship path is **`01_eda_paysim.ipynb` §12.9b**, which selects a finalist (often **`catboost_plain_sigmoid`**) from the calibrated comparison table—**that object is what drives buckets and policy.**

**Refreshing without the notebook (`build_artifacts.py`):** the script retrains/refreshes the **RF-family** preprocessor and joblibs (`rf_plain_base.joblib`, sigmoid/isotonic RF calibrators, merged calibration table rows). It **does not automatically switch deploy to RF**: **`final_model_key` is chosen** after **`PAYSIM_FINAL_MODEL_KEY`** (if set), otherwise **prior finalist** if its calibrator file still exists under `artifacts/`, otherwise the **RF** pick that `build_artifacts` just produced. So you only end on **`rf_plain_sigmoid`** when there is **no surviving non-RF scorer on disk** (or when you explicitly set the env var).

**Local SHAP in the app:** `TreeExplainer` runs on the **fitted tree model inside the deploy calibrator** (e.g. **CatBoost** unwrapped from `CalibratedClassifierCV`) whenever SHAP accepts that object—**baseline probability** (`base_probability` in outputs) tracks that same tree. **`artifacts/rf_plain_base.joblib`** is **only loaded** so the Streamlit bundle always has **one** RF skeleton for SHAP/testing when CatBoost/XGB/LightGBM cannot be explained in-process; **it is not the default SHAP target** when CatBoost unwrap works.

**Notebook training breadth:** Section **12** compares LR, RF, boosting, CatBoost, etc.; export picks **`FINAL_MODEL_KEY`** and the matching deploy calibrator. **`build_artifacts.py`** refreshes RF-family joblibs and **merges** calibration tables against prior **`feature_metadata.json`**—**deploy can stay CatBoost/XGB** when those calibrator files are still present under `artifacts/`.

## 7. Decision policy (cost-sensitive triage)
The final action is derived from calibrated probability and the chain escalation rule.

Policy parameters (**from notebook export**, also in `artifacts/feature_metadata.json`):

- false positive cost = **5**
- false negative cost = **500**
- `review_threshold` = **0.05**
- `block_threshold` = **0.15**
- `moderate_cutoff` = **0.05**
- `operating_threshold` (cost-optimal scalar search) = **0.05**
- `chain_size_cap` = **12**

Triage rule (3-way):

- If `p < 0.05` → **GREEN**
- Else if `0.05 <= p < 0.15` → **YELLOW**
- Else `p >= 0.15` → **RED**

Chain escalation rule:

- If **chain signal is active** (`is_chain_member = 1`) and `p >= moderate_cutoff` (0.05),
  then the bucket can be escalated to **RED** (even when not already in the `p >= 0.15` band).

## 8. Evaluation and evidence
### Key metrics (deployed model) + supporting evidence

Values below reflect **the same notebook run** that exported `feature_metadata.json` (selected scorer: **`catboost_plain_sigmoid`**).

| Metric | Value |
|--------|-------|
| PR-AUC (test, calibrated deploy model) | 0.998554 |
| Brier Score | 3.143698e-06 |
| ROC-AUC | 0.999905 |
| Fraud captured in RED (before → after escalation) | 99.76% → 99.76% |
| Legitimate allowed (GREEN) | 100.000% |
| Logistic Regression ΔPR-AUC (chain vs no-chain; notebook §12.5) | +0.010 |

**Bootstrap PR-AUC (95% CI, notebook §12.9a):**

| Statistic | Value |
|----------|-------|
| PR-AUC point estimate | 0.9985543777615832 |
| Bootstrap mean PR-AUC | 0.9985199532798703 |
| 95% CI lower | 0.996327499809248 |
| 95% CI upper | 0.9999501386694014 |
| Valid bootstrap runs | 300 / 300 |

### Model comparison context (notebook evidence, not deployment)

The notebook compares **multiple families** (including boosting, **CatBoost**, **Balanced RF**, **Gaussian NB**, etc.) under the same chain-aware pipeline. The **deployed** scorer is whatever **`final_model_key`** + **`calibration_model_file_map`** reference after export — **not** a fixed legacy RF path unless that key wins or you used **`build_artifacts.py`** alone.

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
- Academic honesty statement: **Fraud score and final action come from the calibrated pipeline referenced by `final_model_key` / `calibration_model_file_map`, not from the LLM.**

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
The Streamlit app loads from `artifacts/`:
- **Preprocessor** — `preprocessor_paysim.joblib` (column order + encoding from the training pipeline).
- **Deploy scorer (calibrated)** — whatever file `calibration_model_file_map[final_model_key]` points to (e.g. `catboost_plain_sigmoid_calibrated.joblib` for the current notebook-led export). This drives **probability + triage buckets**.
- **Tree model for Tree SHAP + `base_probability`** — the Streamlit loader **unwraps** the calibrated object to its inner booster (typically **CatBoostClassifier** today). **`rf_plain_base.joblib`** is a **standalone RF** kept on disk **only as a SHAP/runtime fallback** if the unwrap path cannot be wrapped by `shap.TreeExplainer` on your Python/SHAP/CatBoost build.
- **Feature metadata** — `feature_metadata.json` (thresholds, feature names, `final_model_key`, calibration table, optional bootstrap/triage snapshots).

If you rebuild artifacts using `build_artifacts.py`, the Streamlit app should remain consistent because it reads thresholds and feature mappings from `feature_metadata.json`.

You can instead run the **`01_eda_paysim.ipynb` §12.9b tail cell**, which writes the same artifact directory from the live notebook session and refreshes this model card’s §7–§8 block (everything up to **“### Model comparison context …”**) so it matches those exports.

