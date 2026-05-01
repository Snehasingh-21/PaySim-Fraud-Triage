"""
Export Streamlit-ready artifacts + feature_metadata.json from notebook objects.

After Sections 12.4, 12.8, and 12.9b in `01_eda_paysim.ipynb`, run the export cell that calls
`sync_from_notebook_session(...)`.

Writes `artifacts/*.joblib` and `artifacts/feature_metadata.json` so Streamlit matches the notebook
session. Restart Streamlit after export (cached resource load).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

RF_CALIBRATION_ROWS = ["rf_plain_uncalibrated", "rf_plain_sigmoid", "rf_plain_isotonic"]

DEFAULT_COST_FP = 5
DEFAULT_COST_FN = 500


def _project_root_fallback(root: Path | None) -> Path:
    if root is not None:
        return root.expanduser().resolve()
    return Path.cwd().resolve()


def _selected_row_for_metrics(table: pd.DataFrame, model_key: str) -> pd.Series | None:
    if table is None or table.empty or "model" not in table.columns:
        return None
    sub = table[table["model"].astype(str) == str(model_key)]
    return None if sub.empty else sub.iloc[0]


def _build_metadata_payload(
    *,
    input_feature_columns: list[str],
    processed_feature_names: list[str],
    triage_thresholds: Mapping[str, float],
    chain_size_cap: int,
    final_model_key: str,
    final_model_reason: str,
    calibration_comparison_table: pd.DataFrame,
    bootstrap_prauc: Mapping[str, Any] | None,
    triage_snapshot: Mapping[str, Any] | None,
    cost_fp: int,
    cost_fn: int,
) -> dict[str, Any]:
    tbl = calibration_comparison_table.copy().reset_index(drop=True)
    rf_tbl = tbl[tbl["model"].isin(RF_CALIBRATION_ROWS)].copy()
    if rf_tbl.empty:
        raise ValueError(
            "calibration_comparison_table must include rf_plain_uncalibrated, "
            "rf_plain_sigmoid, and rf_plain_isotonic rows."
        )

    metadata: dict[str, Any] = {
        "input_feature_columns": list(input_feature_columns),
        "processed_feature_names": list(processed_feature_names),
        "triage_thresholds": {k: float(v) for k, v in dict(triage_thresholds).items()},
        "chain_size_cap": int(chain_size_cap),
        "final_model_key": str(final_model_key),
        "final_model_reason": str(final_model_reason),
        "calibration_selection_rule": "lowest_brier_then_higher_pr_auc_then_higher_roc_auc_within_rf_family",
        "calibration_comparison_table": rf_tbl.to_dict(orient="records"),
        "calibration_model_file_map": {
            "rf_plain_uncalibrated": "rf_plain_base.joblib",
            "rf_plain_sigmoid": "rf_plain_sigmoid_calibrated.joblib",
            "rf_plain_isotonic": "rf_plain_isotonic_calibrated.joblib",
        },
        "cost_sensitive_policy": {"cost_fp": int(cost_fp), "cost_fn": int(cost_fn)},
        "exported_from_notebook": True,
        "notes": (
            "Artifacts synced from Jupyter notebook session; preprocessor + RF + RF-family "
            "calibration + triage thresholds match the notebook evaluation run."
        ),
    }

    if triage_snapshot:
        metadata["triage_snapshot"] = dict(triage_snapshot)
    if bootstrap_prauc:
        metadata["bootstrap_prauc_snapshot"] = dict(bootstrap_prauc)

    row = _selected_row_for_metrics(rf_tbl, final_model_key)
    if row is not None:
        metadata["selected_calibration_metrics_test"] = {
            "model": final_model_key,
            "brier": float(row["brier"]),
            "pr_auc": float(row["pr_auc"]),
            "roc_auc": float(row["roc_auc"]),
        }

    return metadata


def _write_auto_model_card_segment(meta: dict[str, Any]) -> str:
    tnames = meta["triage_thresholds"]
    rev = float(tnames["review_threshold"])
    blk = float(tnames["block_threshold"])
    mod = float(tnames["moderate_cutoff"])
    op = float(tnames.get("operating_threshold", rev))
    cap = int(meta["chain_size_cap"])
    fk = str(meta["final_model_key"])

    cs = meta.get("cost_sensitive_policy", {})
    cf = int(cs.get("cost_fp", DEFAULT_COST_FP))
    cn = int(cs.get("cost_fn", DEFAULT_COST_FN))

    sel = meta.get("selected_calibration_metrics_test", {})
    pr = sel.get("pr_auc")
    bri = sel.get("brier")
    roc = sel.get("roc_auc")

    pr_s = f"{float(pr):.6f}" if pr is not None else "(n/a)"
    roc_s = f"{float(roc):.6f}" if roc is not None else "(n/a)"

    def _fmt_brier(x: float | None) -> str:
        if x is None:
            return "(n/a)"
        return f"{float(x):.6e}"

    tsnap = meta.get("triage_snapshot") or {}

    fc_after = tsnap.get("fraud_capture_in_red_after_pct")
    fc_before = tsnap.get("fraud_capture_in_red_before_pct")
    legit_green = tsnap.get("legitimate_in_green_pct")

    if fc_before is not None and fc_after is not None:
        fc_cell = f"{float(fc_before):.2f}% → {float(fc_after):.2f}%"
    elif fc_after is not None:
        fc_cell = f"{float(fc_after):.2f}%"
    else:
        fc_cell = "(rerun notebook export cell)"

    leg_cell = f"{float(legit_green):.3f}%" if legit_green is not None else "(rerun notebook export cell)"

    boot = meta.get("bootstrap_prauc_snapshot") or {}
    boot_block = "**Bootstrap §12.9a:** values appear here when export includes bootstrap snapshot."
    if boot:
        boot_block = (
            "**Bootstrap PR-AUC (95% CI, notebook §12.9a):**\n\n"
            "| Statistic | Value |\n|----------|-------|\n"
            f"| PR-AUC point estimate | {boot.get('pr_auc_point', '—')} |\n"
            f"| Bootstrap mean PR-AUC | {boot.get('pr_auc_bootstrap_mean', '—')} |\n"
            f"| 95% CI lower | {boot.get('ci_lower', '—')} |\n"
            f"| 95% CI upper | {boot.get('ci_upper', '—')} |\n"
            f"| Valid bootstrap runs | {boot.get('valid_runs', '—')} "
            f"/ {boot.get('n_boot_requested', '—')} |"
        )

    return f"""## 7. Decision policy (cost-sensitive triage)
The final action is derived from calibrated probability and the chain escalation rule.

Policy parameters (**from notebook export**, also in `artifacts/feature_metadata.json`):

- false positive cost = **{cf}**
- false negative cost = **{cn}**
- `review_threshold` = **{rev:.2f}**
- `block_threshold` = **{blk:.2f}**
- `moderate_cutoff` = **{mod:.2f}**
- `operating_threshold` (cost-optimal scalar search) = **{op:.2f}**
- `chain_size_cap` = **{cap}**

Triage rule (3-way):

- If `p < {rev:.2f}` → **GREEN**
- Else if `{rev:.2f} <= p < {blk:.2f}` → **YELLOW**
- Else `p >= {blk:.2f}` → **RED**

Chain escalation rule:

- If **chain signal is active** (`is_chain_member = 1`) and `p >= moderate_cutoff` ({mod:.2f}),
  then the bucket can be escalated to **RED** (even when not already in the `p >= {blk:.2f}` band).

## 8. Evaluation and evidence
### Key metrics (deployed model) + supporting evidence

Values below reflect **the same notebook run** that exported `feature_metadata.json` (selected scorer: **`{fk}`**).

| Metric | Value |
|--------|-------|
| PR-AUC (test, calibrated RF) | {pr_s} |
| Brier Score | {_fmt_brier(float(bri) if bri is not None else None)} |
| ROC-AUC | {roc_s} |
| Fraud captured in RED (before → after escalation) | {fc_cell} |
| Legitimate allowed (GREEN) | {leg_cell} |
| Logistic Regression ΔPR-AUC (chain vs no-chain; notebook §12.5) | +0.010 |

{boot_block}
"""


def rewrite_model_card_eval_section(model_card_path: Path, meta: dict[str, Any]) -> None:
    text = model_card_path.read_text(encoding="utf-8")
    segment = _write_auto_model_card_segment(meta).strip()
    pattern = (
        r"## 7\. Decision policy \(cost-sensitive triage\)\n"
        r"[\s\S]*?\n(?=### Model comparison context \(notebook evidence, not deployment\)\n)"
    )
    if not re.search(pattern, text):
        raise ValueError(
            "Could not find MODEL_CARD.md span to replace (expected ## 7 ... before "
            "'### Model comparison context')."
        )
    text_new = re.sub(pattern, segment + "\n\n", text, count=1)
    model_card_path.write_text(text_new, encoding="utf-8")


def sync_from_notebook_session(
    *,
    preprocessor: Any,
    rf_base_model: Any,
    x_train_columns: list[str],
    processed_feature_names: list[str],
    final_model_key: str,
    final_model_reason: str,
    calibration_comparison_table: pd.DataFrame,
    rf_calibrator_by_key: Mapping[str, Any],
    triage_thresholds: Mapping[str, float],
    chain_size_cap: int,
    y_true: Any,
    final_bucket_labels: Any,
    base_bucket_labels: Any | None = None,
    project_root: Path | None = None,
    bootstrap_prauc: Mapping[str, Any] | None = None,
    cost_fp: int = DEFAULT_COST_FP,
    cost_fn: int = DEFAULT_COST_FN,
    rewrite_model_card: bool = True,
) -> Path:
    root = _project_root_fallback(project_root)
    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)

    fk = str(final_model_key)
    yt = np.asarray(y_true).astype(int).ravel()
    fb = np.asarray(final_bucket_labels).astype(str).ravel()

    total_fraud = int((yt == 1).sum())
    total_legit = int((yt == 0).sum())

    fraud_capture_after = float(100.0 * ((yt == 1) & (fb == "RED")).sum() / max(total_fraud, 1))
    legit_green_pct = float(100.0 * ((yt == 0) & (fb == "GREEN")).sum() / max(total_legit, 1))

    fraud_capture_before: float | None = None
    if base_bucket_labels is not None:
        bb = np.asarray(base_bucket_labels).astype(str).ravel()
        if len(bb) != len(yt):
            raise ValueError("base_bucket_labels length must match y_true.")
        before_red = int(((yt == 1) & (bb == "RED")).sum())
        fraud_capture_before = float(100.0 * before_red / max(total_fraud, 1))

    triage_snapshot = {
        "fraud_capture_in_red_after_pct": fraud_capture_after,
        "fraud_capture_in_red_before_pct": fraud_capture_before,
        "legitimate_in_green_pct": legit_green_pct,
    }

    joblib.dump(preprocessor, out_dir / "preprocessor_paysim.joblib")
    joblib.dump(rf_base_model, out_dir / "rf_plain_base.joblib")

    for k in ("rf_plain_sigmoid", "rf_plain_isotonic"):
        if k not in rf_calibrator_by_key:
            raise ValueError(
                f"rf_calibrator_by_key missing {k}. Re-run calibration cell that fills "
                "NOTEBOOK_RF_CALIBRATORS_FOR_STREAMLIT."
            )
        joblib.dump(rf_calibrator_by_key[k], out_dir / f"{k}_calibrated.joblib")

    if fk == "rf_plain_uncalibrated":
        selected = rf_base_model
    elif fk in rf_calibrator_by_key:
        selected = rf_calibrator_by_key[fk]
    else:
        raise ValueError(f"Unsupported FINAL_MODEL_KEY for export: {fk}")

    joblib.dump(selected, out_dir / "rf_selected_calibrated.joblib")

    meta = _build_metadata_payload(
        input_feature_columns=list(x_train_columns),
        processed_feature_names=list(processed_feature_names),
        triage_thresholds=triage_thresholds,
        chain_size_cap=chain_size_cap,
        final_model_key=fk,
        final_model_reason=str(final_model_reason),
        calibration_comparison_table=calibration_comparison_table,
        bootstrap_prauc=bootstrap_prauc,
        triage_snapshot=triage_snapshot,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )

    (out_dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if rewrite_model_card:
        mc = root / "MODEL_CARD.md"
        if mc.exists():
            try:
                rewrite_model_card_eval_section(mc, meta)
            except ValueError as err:
                warnings.warn(
                    f"Artifacts written, but MODEL_CARD.md was not auto-updated ({err}). Edit §7 headline manually.",
                    RuntimeWarning,
                    stacklevel=1,
                )

    return out_dir


__all__ = ["sync_from_notebook_session", "rewrite_model_card_eval_section"]
