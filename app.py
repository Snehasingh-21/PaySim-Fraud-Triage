import json
import re
import subprocess
from typing import Any, Optional
import base64
import hashlib
import os
import sys
from html import escape
from pathlib import Path
from datetime import datetime
from urllib import error as urlerror
from urllib import request as urlrequest

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass


_APP_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = _APP_ROOT / "artifacts"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor_paysim.joblib"
BASE_MODEL_PATH = ARTIFACT_DIR / "rf_plain_base.joblib"
META_PATH = ARTIFACT_DIR / "feature_metadata.json"
HEADER_IMAGE_PATH = _APP_ROOT / "assets" / "fraud_triage_header.svg"
TITLE_ICON_PATH = _APP_ROOT / "assets" / "title_calibration_icon.png"
COMMAND_CENTER_IMAGE_PATH = _APP_ROOT / "assets" / "command_center_heroic_local.svg"
MODEL_CARD_PATH = _APP_ROOT / "MODEL_CARD.md"
COST_FP = 5
COST_FN = 500
# Bump when the analyst prompt changes so cached summaries are not reused across app versions.
OLLAMA_ANALYST_PROMPT_VERSION = 5
BUILD_SCRIPT_PATH = _APP_ROOT / "build_artifacts.py"

# When notebook is newer than artifacts (or required files missing), optionally run RF `build_artifacts.py`.
# Deploy finalist stays dynamic (CatBoost preserved if `.joblib` + map remain after merge). Disable with AUTO_REBUILD_IF_STALE=0.
AUTO_REBUILD_IF_STALE_DEFAULT = "1"


def _auto_rebuild_timeout_seconds() -> Optional[int]:
    """
    Seconds for subprocess.run on build_artifacts.py. None = no limit.
    AUTO_REBUILD_TIMEOUT_SEC: default 7200 (2h). Use 0, negative, or "none"/"inf" for unlimited.
    """
    raw_default = "7200"
    raw = str(os.getenv("AUTO_REBUILD_TIMEOUT_SEC", raw_default)).strip().lower()
    if raw in {"none", "inf", "infinite"}:
        return None
    try:
        n = int(raw)
    except ValueError:
        n = int(raw_default)
    if n <= 0:
        return None
    return n


def _resolve_paysim_csv() -> Path:
    """Default CSV next to app; override with PAYSIM_CSV or PAYSIM_DATA_PATH (abs or relative to app)."""
    for key in ("PAYSIM_CSV", "PAYSIM_DATA_PATH"):
        raw = os.getenv(key)
        if raw:
            p = Path(raw).expanduser()
            return p.resolve() if p.is_absolute() else (_APP_ROOT / p).resolve()
    name = os.getenv("PAYSIM_CSV_NAME", "PS_20174392719_1491204439457_log.csv")
    return (_APP_ROOT / name).resolve()


def _resolve_notebook_path() -> Path:
    """Notebook that drives auto-rebuild staleness; override with PAYSIM_NOTEBOOK."""
    raw = os.getenv("PAYSIM_NOTEBOOK")
    if raw:
        p = Path(raw).expanduser()
        return p.resolve() if p.is_absolute() else (_APP_ROOT / p).resolve()
    return (_APP_ROOT / "01_eda_paysim.ipynb").resolve()


DATA_PATH = _resolve_paysim_csv()
NOTEBOOK_PATH = _resolve_notebook_path()


def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _artifacts_missing() -> bool:
    required = [PREPROCESSOR_PATH, BASE_MODEL_PATH, META_PATH]
    return any(not p.exists() for p in required)


def _latest_mtime(paths: list[Path]) -> float:
    existing = [p.stat().st_mtime for p in paths if p.exists()]
    return max(existing) if existing else 0.0


def _artifacts_stale() -> bool:
    """
    True when saved artifacts are older than the EDA notebook.

    Only the notebook timestamp drives auto-rebuild (not build_artifacts.py), so editing
    the build script alone does not force a long rebuild on Streamlit start—only saving
    the notebook after real modeling changes does.
    """
    artifact_paths = [META_PATH, PREPROCESSOR_PATH, BASE_MODEL_PATH]
    artifact_paths.extend(sorted(ARTIFACT_DIR.glob("*.joblib")))
    if not NOTEBOOK_PATH.exists():
        return False
    return _latest_mtime([NOTEBOOK_PATH]) > _latest_mtime(artifact_paths)


def maybe_auto_rebuild_artifacts() -> tuple[bool, str]:
    """
    Run RF ``build_artifacts.py`` when stale/missing unless ``AUTO_REBUILD_IF_STALE=0``.
    ``feature_metadata.json`` merge keeps CatBoost/XGB deploy when those joblibs + map survive the run.

    ``needs_rebuild`` = missing required paths OR notebook newer than artifact mtimes (see ``_artifacts_stale``).
    If PaySim CSV is missing, skips quietly.
    """
    if not _env_flag("AUTO_REBUILD_IF_STALE", AUTO_REBUILD_IF_STALE_DEFAULT):
        return False, ""
    if not BUILD_SCRIPT_PATH.exists():
        return False, ""

    missing = _artifacts_missing()
    stale = _artifacts_stale()
    needs_rebuild = missing or stale
    if not needs_rebuild:
        return False, ""

    if not DATA_PATH.exists():
        return False, (
            "Auto-rebuild skipped (PaySim CSV not found). "
            "Using existing artifacts if available."
        )

    timeout_sec = _auto_rebuild_timeout_seconds()
    try:
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT_PATH)],
            cwd=str(_APP_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        hint = (
            "Increase `AUTO_REBUILD_TIMEOUT_SEC` (seconds) or set it to `0` for no limit. "
            "Or run `python build_artifacts.py` manually if you enabled RF bootstrap. "
            "Default deploy path: Jupyter §12.9c export (no RF auto-rebuild)."
        )
        limit = "no timeout configured" if timeout_sec is None else f"{timeout_sec}s limit"
        raise RuntimeError(
            "Auto-rebuild timed out while running build_artifacts.py "
            f"({limit} on full PaySim can exceed 30 minutes). "
            + hint
        ) from None
    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()[-600:]
        stdout_tail = (result.stdout or "").strip()[-600:]
        detail = stderr_tail or stdout_tail or "Unknown build_artifacts.py error."
        raise RuntimeError(
            "Auto-rebuild failed while running build_artifacts.py. "
            f"Details: {detail}"
        )
    return True, "Artifacts were auto-rebuilt from `build_artifacts.py` (missing/stale detected)."


def _unwrap_sklearn_calibrated_estimator(est: Any) -> Any:
    """Single-level: first fitted base estimator inside CalibratedClassifierCV (if any)."""
    cals = getattr(est, "calibrated_classifiers_", None)
    if not cals:
        return est
    try:
        first = cals[0]
        inner = getattr(first, "estimator", None)
        if inner is None:
            inner = getattr(first, "estimator_", None)
        if inner is None:
            inner = getattr(first, "base_estimator", None)
        if inner is not None:
            return inner
    except Exception:
        pass
    return est


def _unpack_tree_estimator_for_shap(est: Any) -> Any:
    """Peel CalibratedClassifierCV shells and sklearn `Pipeline` final steps to reach the concrete tree booster."""
    from sklearn.pipeline import Pipeline

    cur: Any = est
    for _ in range(12):
        nxt = _unwrap_sklearn_calibrated_estimator(cur)
        if nxt is not cur:
            cur = nxt
            continue
        if isinstance(cur, Pipeline) and getattr(cur, "steps", None):
            last_est = cur.steps[-1][1]
            if last_est is cur:
                break
            cur = last_est
            continue
        break
    return cur


def _shap_explainer_candidates(cal_model: Any, rf_fallback: Any) -> list[Any]:
    """Prefer deploy inner tree → raw calibrator object → bundled RF artifact (SHAP fallback)."""
    out: list[Any] = []
    seen: set[int] = set()

    def push(e: Any) -> None:
        if e is None:
            return
        i = id(e)
        if i in seen:
            return
        seen.add(i)
        out.append(e)

    deep_inner = _unpack_tree_estimator_for_shap(cal_model)
    shallow_inner = _unwrap_sklearn_calibrated_estimator(cal_model)
    push(deep_inner)
    if shallow_inner is not deep_inner:
        push(shallow_inner)
    if cal_model not in out:
        push(cal_model)
    push(rf_fallback)
    return out


def _pick_tree_estimator_for_shap_and_baseline(cal_model: Any, rf_fallback: Any) -> tuple[Any, str]:
    """
    Tree SHAP + `base_probability` use the **deploy model’s inner tree** (e.g. CatBoost via
    `CalibratedClassifierCV`) when `shap.TreeExplainer` accepts it. `rf_plain_base.joblib` is only
    for backup when the deploy object is not tree-SHAP compatible in this environment.
    """
    try:
        import shap  # type: ignore
    except Exception:
        est = rf_fallback
        return est, f"`{type(est).__name__}` (`rf_plain_base.joblib`; SHAP not installed)"

    last_err = ""
    for est in _shap_explainer_candidates(cal_model, rf_fallback):
        try:
            shap.TreeExplainer(est)
            if id(est) == id(rf_fallback):
                return est, "`rf_plain_base.joblib` (**SHAP fallback** — deploy tree not explodable here)"
            return est, f"`{type(est).__name__}` (**deploy inner tree**, same boosted model SHAP traces)"
        except Exception as exc:
            last_err = str(exc)[:180]
            continue
    return rf_fallback, f"`rf_plain_base.joblib` (**SHAP fallback** — deploy inner error: {last_err})"


@st.cache_resource(show_spinner=False)
def load_artifacts(artifact_cache_key: str):
    missing = [p for p in [PREPROCESSOR_PATH, BASE_MODEL_PATH, META_PATH] if not p.exists()]
    if missing:
        missing_text = "\n".join(f"- `{m}`" for m in missing)
        raise FileNotFoundError(
            "Missing artifacts:\n"
            f"{missing_text}\n\n"
            "Export from Jupyter (notebook §12.9c) with project-root cwd, "
            "or run `python build_artifacts.py` for RF fallback only "
            "(unset `AUTO_REBUILD_IF_STALE` or use `AUTO_REBUILD_IF_STALE=0` to disable auto RF rebuild)."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    rf_plain_fallback = joblib.load(BASE_MODEL_PATH)
    metadata = json.loads(META_PATH.read_text())

    # Defaults cover build_artifacts.py / minimal exports; notebook JSON adds CatBoost, XGBoost, etc.
    _default_cal_map = {
        "rf_plain_uncalibrated": "rf_plain_base.joblib",
        "rf_plain_sigmoid": "rf_plain_sigmoid_calibrated.joblib",
        "rf_plain_isotonic": "rf_plain_isotonic_calibrated.joblib",
    }
    _meta_map = metadata.get("calibration_model_file_map") or {}
    model_file_map = {**_default_cal_map, **_meta_map}

    final_model_key = str(metadata.get("final_model_key", "")).strip()

    if final_model_key:
        cal_model_file = model_file_map.get(final_model_key)
        if not cal_model_file:
            raise FileNotFoundError(
                f"`final_model_key`={final_model_key!r} is not listed in `calibration_model_file_map`. "
                "Re-export artifacts from the notebook so RF / XGB / CatBoost rows map to files on disk."
            )
    else:
        # Older/minimal exports: try canonical alias files (notebook + build_artifacts both may write these).
        cal_model_file = None
        if (ARTIFACT_DIR / "rf_selected_calibrated.joblib").exists():
            cal_model_file = "rf_selected_calibrated.joblib"
            final_model_key = "rf_selected_calibrated"
        elif (ARTIFACT_DIR / "rf_plain_sigmoid_calibrated.joblib").exists():
            cal_model_file = "rf_plain_sigmoid_calibrated.joblib"
            final_model_key = "rf_plain_sigmoid"
        else:
            raise FileNotFoundError(
                "artifacts/feature_metadata.json has empty `final_model_key` and no "
                "`rf_selected_calibrated.joblib` / `rf_plain_sigmoid_calibrated.joblib` fallback found."
            )

    cal_model_path = ARTIFACT_DIR / cal_model_file
    if not cal_model_path.exists():
        raise FileNotFoundError(
            f"Calibrated model file missing: `{cal_model_path}` for `final_model_key={final_model_key}`. "
            "Ensure the matching joblib exists next to feature_metadata.json."
        )
    cal_model = joblib.load(cal_model_path)
    base_model, shap_ui_hint = _pick_tree_estimator_for_shap_and_baseline(cal_model, rf_plain_fallback)
    metadata["resolved_calibration_model_file"] = cal_model_file
    metadata["resolved_final_model_key"] = final_model_key
    metadata["tree_shap_baseline_explainer_hint"] = shap_ui_hint
    metadata["tree_shap_backend_class"] = type(base_model).__name__
    return preprocessor, base_model, cal_model, metadata


def build_artifact_cache_key() -> str:
    """
    Build a cache key from artifact mtimes so notebook/build refreshes are picked up on rerun.
    """
    tracked = [META_PATH, PREPROCESSOR_PATH, BASE_MODEL_PATH]
    tracked.extend(sorted(ARTIFACT_DIR.glob("*.joblib")))
    parts = []
    for p in tracked:
        if p.exists():
            parts.append(f"{p.name}:{p.stat().st_mtime_ns}")
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def load_model_card_text(model_card_mtime: float):
    """Read MODEL_CARD.md once per *file version* to avoid stale cached content."""
    if not MODEL_CARD_PATH.exists():
        return None
    return MODEL_CARD_PATH.read_text(encoding="utf-8")




def apply_live_metadata_to_model_card(md: str, metadata: dict) -> str:
    """Patch MODEL_CARD.md text so §§7–§8 numeric lines match loaded `artifacts/feature_metadata.json` (same tab layout)."""
    out = md

    def _gf(x: float) -> str:
        return format(float(x), "g")

    fk = str(metadata.get("resolved_final_model_key") or metadata.get("final_model_key") or "").strip()

    cs = metadata.get("cost_sensitive_policy") or {}
    if "cost_fp" in cs:
        out = re.sub(
            r"- false positive cost = \*\*\d+\*\*",
            f"- false positive cost = **{int(cs['cost_fp'])}**",
            out,
            count=1,
        )
    if "cost_fn" in cs:
        out = re.sub(
            r"- false negative cost = \*\*\d+\*\*",
            f"- false negative cost = **{int(cs['cost_fn'])}**",
            out,
            count=1,
        )

    tn = metadata.get("triage_thresholds") or {}
    if tn:
        rr = float(tn["review_threshold"])
        blk = float(tn["block_threshold"])
        mod = float(tn["moderate_cutoff"])
        op = float(tn.get("operating_threshold", rr))
        rr_s, blk_s, mod_s, op_s = _gf(rr), _gf(blk), _gf(mod), _gf(op)

        out = re.sub(r"- `review_threshold` = \*\*[\d.]+\*\*", f"- `review_threshold` = **{rr_s}**", out, count=1)
        out = re.sub(r"- `block_threshold` = \*\*[\d.]+\*\*", f"- `block_threshold` = **{blk_s}**", out, count=1)
        out = re.sub(r"- `moderate_cutoff` = \*\*[\d.]+\*\*", f"- `moderate_cutoff` = **{mod_s}**", out, count=1)
        out = re.sub(
            r"- `operating_threshold` \(cost-optimal scalar search\) = \*\*[\d.]+\*\*",
            f"- `operating_threshold` (cost-optimal scalar search) = **{op_s}**",
            out,
            count=1,
        )

        cap_i = metadata.get("chain_size_cap")
        if cap_i is not None:
            out = re.sub(
                r"- `chain_size_cap` = \*\*\d+\*\*",
                f"- `chain_size_cap` = **{int(cap_i)}**",
                out,
                count=1,
            )

        out = re.sub(
            r"- If `p < [\d.]+` → \*\*GREEN\*\*",
            f"- If `p < {rr_s}` → **GREEN**",
            out,
            count=1,
        )
        out = re.sub(
            r"- Else if `[\d.]+\s*<= p < [\d.]+` → \*\*YELLOW\*\*",
            f"- Else if `{rr_s} <= p < {blk_s}` → **YELLOW**",
            out,
            count=1,
        )
        out = re.sub(
            r"- Else `p >= [\d.]+` → \*\*RED\*\*",
            f"- Else `p >= {blk_s}` → **RED**",
            out,
            count=1,
        )
        out = re.sub(
            r"`p >= moderate_cutoff` \(\d\.?\d*\)",
            f"`p >= moderate_cutoff` ({escape(mod_s)})",
            out,
            count=1,
        )

    if fk:
        out = re.sub(
            r"\(often \*\*`[^`]+`\*\*\) from the calibrated comparison table",
            f"(often **`{fk}`**) from the calibrated comparison table",
            out,
            count=1,
        )
        out = re.sub(
            r"\(selected scorer: \*\*`[^`]+`\*\*\)",
            f"(selected scorer: **`{fk}`**)",
            out,
            count=1,
        )

    scm = metadata.get("selected_calibration_metrics_test") or {}
    if scm.get("pr_auc") is not None:
        out = re.sub(
            r"(\| PR-AUC \(test, calibrated deploy model\) \| )[^|]*(\|)",
            rf"\g<1>{scm['pr_auc']}",
            out,
            count=1,
        )
    if scm.get("brier") is not None:
        out = re.sub(
            r"(\| Brier Score \| )[^|]*(\|)",
            rf"\g<1>{scm['brier']}",
            out,
            count=1,
        )
    if scm.get("roc_auc") is not None:
        out = re.sub(
            r"(\| ROC-AUC \| )[^|]*(\|)",
            rf"\g<1>{scm['roc_auc']}",
            out,
            count=1,
        )

    ts_snap = metadata.get("triage_snapshot") or {}
    bef = ts_snap.get("fraud_capture_in_red_before_pct")
    aft = ts_snap.get("fraud_capture_in_red_after_pct")
    if bef is not None and aft is not None:
        fraud_cell = f"{float(bef):.4f}% → {float(aft):.4f}%"
        out = re.sub(
            r"(\| Fraud captured in RED \(before → after escalation\) \| )[^|]*(\|)",
            rf"\g<1>{fraud_cell}",
            out,
            count=1,
        )
    if ts_snap.get("legitimate_in_green_pct") is not None:
        pct = float(ts_snap["legitimate_in_green_pct"])
        lc = format(pct, ".6f").rstrip("0").rstrip(".") + "%"
        out = re.sub(
            r"(\| Legitimate allowed \(GREEN\) \| )[^|]*(\|)",
            rf"\g<1>{lc}",
            out,
            count=1,
        )

    boot = metadata.get("bootstrap_prauc_snapshot") or {}
    if boot.get("pr_auc_point") is not None:
        out = re.sub(
            r"(\| PR-AUC point estimate \| )[^|]*(\|)",
            rf"\g<1>{boot['pr_auc_point']}",
            out,
            count=1,
        )
    if boot.get("pr_auc_bootstrap_mean") is not None:
        out = re.sub(
            r"(\| Bootstrap mean PR-AUC \| )[^|]*(\|)",
            rf"\g<1>{boot['pr_auc_bootstrap_mean']}",
            out,
            count=1,
        )
    if boot.get("ci_lower") is not None:
        out = re.sub(
            r"(\| 95% CI lower \| )[^|]*(\|)",
            rf"\g<1>{boot['ci_lower']}",
            out,
            count=1,
        )
    if boot.get("ci_upper") is not None:
        out = re.sub(
            r"(\| 95% CI upper \| )[^|]*(\|)",
            rf"\g<1>{boot['ci_upper']}",
            out,
            count=1,
        )
    vr, nb = boot.get("valid_runs"), boot.get("n_boot_requested")
    if vr is not None and nb is not None:
        out = re.sub(
            r"(\| Valid bootstrap runs \| )[^|]*(\|)",
            rf"\g<1>{int(vr)} / {int(nb)}",
            out,
            count=1,
        )

    return out


def add_engineered_features_for_inference(
    df_raw: pd.DataFrame,
    chain_size_cap: int,
    chain_mode: str,
    fallback_chain_size: int = 1,
    fallback_is_chain_member: int = 0,
) -> pd.DataFrame:
    df = df_raw.copy()
    df["orig_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_residual"] = df["orig_delta"] - df["amount"]
    df["orig_zero_old"] = (df["oldbalanceOrg"] == 0).astype(np.int8)
    df["dest_zero_old"] = (df["oldbalanceDest"] == 0).astype(np.int8)
    df["orig_zero_new"] = (df["newbalanceOrig"] == 0).astype(np.int8)
    df["dest_zero_new"] = (df["newbalanceDest"] == 0).astype(np.int8)
    df["log_amount"] = np.log1p(df["amount"].astype(np.float64))

    if chain_mode == "provided":
        if "chain_size" not in df.columns or "is_chain_member" not in df.columns:
            raise ValueError(
                "Batch mode requires chain-aware inputs: `chain_size` and `is_chain_member`."
            )
        df["chain_size"] = pd.to_numeric(df["chain_size"], errors="coerce").fillna(1).astype(np.int32)
        df["is_chain_member"] = (
            pd.to_numeric(df["is_chain_member"], errors="coerce").fillna(0).astype(np.int8)
        )
    elif chain_mode == "fallback":
        # Demo-safe fallback when no transaction-history lookup exists.
        df["chain_size"] = int(fallback_chain_size)
        df["is_chain_member"] = int(fallback_is_chain_member)
    else:
        raise ValueError("Invalid chain_mode. Use 'provided' or 'fallback'.")

    if chain_size_cap is not None:
        df["is_chain_member"] = (
            (df["is_chain_member"].astype(int) == 1) & (df["chain_size"].astype(int) <= chain_size_cap)
        ).astype(np.int8)

    return df


def apply_triage_rule(prob: float, is_chain_member: int, thresholds: dict) -> str:
    review_t = float(thresholds["review_threshold"])
    block_t = float(thresholds["block_threshold"])
    moderate_t = float(thresholds["moderate_cutoff"])

    if prob < review_t:
        bucket = "GREEN"
    elif prob < block_t:
        bucket = "YELLOW"
    else:
        bucket = "RED"

    # Chain escalation rule.
    if bucket != "RED" and int(is_chain_member) == 1 and prob >= moderate_t:
        bucket = "RED"
    return bucket


def build_reason_block(
    probability: float,
    bucket: str,
    chain_size: int,
    is_chain_member: int,
    explain_method: str,
    risk_up_drivers: list[str],
    risk_down_drivers: list[str],
) -> str:
    chain_text = (
        "Chain-like pattern detected in the provided input." if is_chain_member == 1
        else "No chain-like pattern detected in the provided input."
    )
    up_text = ", ".join(risk_up_drivers) if risk_up_drivers else "No strong fraud-increasing signals."
    down_text = ", ".join(risk_down_drivers) if risk_down_drivers else "No strong fraud-reducing signals."
    return (
        f"Calibrated fraud probability is {probability:.4f}, so final triage bucket is {bucket}. "
        f"{chain_text} chain_size={int(chain_size)}. "
        f"Risk-increasing drivers: {up_text}. "
        f"Risk-reducing drivers: {down_text}. "
        f"Local explanation method: {explain_method}."
    )


def bucket_meta(bucket: str) -> dict:
    b = bucket.upper()
    if b == "GREEN":
        return {"label": "ALLOW", "color": "#2ca02c", "emoji": "✅"}
    if b == "YELLOW":
        return {"label": "REVIEW", "color": "#ffbf00", "emoji": "⚠️"}
    return {"label": "BLOCK", "color": "#d62728", "emoji": "🛑"}


def next_action_text(bucket: str) -> str:
    b = bucket.upper()
    if b == "GREEN":
        return "Allow and log transaction."
    if b == "YELLOW":
        return "Send to manual review queue."
    return "Block immediately and alert fraud team."


def generate_ollama_summary(context: dict, model_name: str = "llama3.1:8b") -> dict:
    """
    Generate an optional analyst-style note from local Ollama.
    This function is explanation-only and never affects model decisions.
    """
    def _fallback_summary(ctx: dict) -> str:
        bucket = str(ctx.get("final_bucket", "UNKNOWN"))
        prob = str(ctx.get("fraud_probability_percent", "N/A"))
        tx_type = str(ctx.get("transaction_type", "transaction"))
        amount = str(ctx.get("amount", "N/A"))
        action = str(ctx.get("final_action", "Follow current policy action"))
        up = ctx.get("top_reasons_risk_up", []) or []
        down = ctx.get("top_reasons_risk_down", []) or []
        up_txt = ", ".join([str(x) for x in up[:2]]) if up else "no strong risk-increasing signals"
        down_txt = ", ".join([str(x) for x in down[:2]]) if down else "no strong risk-reducing signals"
        return (
            f"This {tx_type} transaction ({amount}) is in {bucket} with calibrated fraud probability {prob}. "
            f"Key signals: risk-up -> {up_txt}; risk-down -> {down_txt}. "
            f"Recommended action: {action}"
        )

    prompt = (
        "You are a fraud analyst leaving a quick Slack-style note for a teammate (warm, conversational tone).\n"
        "Write 2-3 short sentences in plain English.\n"
        "Use only the evidence JSON below; do not invent amounts, types, or probabilities.\n"
        "Stay consistent with the implied handling from final_bucket / final_action, but say it like a person would.\n\n"
        "CRITICAL (hard requirements — your whole reply must satisfy all):\n"
        "- The visible text MUST include the exact transaction_type string from JSON (for example PAYMENT).\n"
        "- The visible text MUST include the exact amount string from JSON (keep the same commas/decimals; you may prefix with $).\n"
        "- The visible text MUST include the exact fraud_probability_percent string from JSON exactly once.\n"
        "- Do NOT answer with only vague reassurance; if you skip any of the three strings above, rewrite until they appear.\n\n"
        "Content expectations:\n"
        "- Open with the transaction story: weave transaction_type + amount together in plain words "
        "(example shape: 'This looks like a normal PAYMENT for $1,000.00' — use the actual type/amount strings).\n"
        "- Add one sentence about the risk picture using ONLY top_reasons_risk_up and top_reasons_risk_down from JSON. "
        "If risk-up reasons are empty or clearly weak, say there are no strong risk indicators (or similar). "
        "If a risk-up reason is present, mention the clearest one in everyday language (do not invent new drivers).\n"
        "- Close with a short next step that matches final_bucket from JSON (this is mandatory):\n"
        "  - GREEN: allow + log only; do NOT suggest manual review, 'double-check', or 'review to be sure'.\n"
        "  - YELLOW: manual review / look closer is appropriate.\n"
        "  - RED: block / escalate / stop wording is appropriate.\n"
        "  Do not paste final_action verbatim; paraphrase in plain words.\n"
        "- Avoid empty platitudes that could apply to every row (do not lean only on 'normal one' / 'nothing risky' "
        "without naming type/amount/probability).\n"
        "- If fraud_probability_percent is extremely low and final_bucket is GREEN, treat SHAP-style drivers as mild "
        "context only — do not flip the story into an uncertain review case.\n"
        "- If final_bucket is GREEN, do NOT write 'However' + 'risk indicator' / 'stands out' scare sentences; "
        "those read like a false alarm next to a ~0% score.\n\n"
        "Style guardrails:\n"
        "- Do NOT use stiff corporate phrases such as: 'in line with', 'as per', 'per the decision', "
        "'the system has determined', 'it has been determined', 'in accordance with'.\n"
        "- Do NOT repeat the same word twice in a row across sentences (especially 'decision').\n"
        "- Do NOT echo JSON field names as visible text.\n"
        "- No bullet points or numbering.\n\n"
        "Output rules:\n"
        "- Return only the final note text (no preamble, no 'Here are', no 'In summary').\n"
        "- Never start with meta labels like 'Here\\'s the note', 'Here is the note', or 'Note:' — start directly with the analyst wording.\n\n"
        "Evidence:\n"
        f"{json.dumps(context, indent=2)}\n"
    )

    def _looks_like_runtime_log(text: str) -> bool:
        t = text.lower()
        markers = [
            "stack trace",
            "backtrace",
            "sigabrt",
            "terminating due to uncaught exception",
            "libc++abi",
            "goroutine",
            "pc=",
            "fault",
        ]
        return any(m in t for m in markers)

    def _clean_summary_text(text: str) -> str:
        """Light cleanup only — preserve the model's own phrasing and line breaks."""
        cleaned_lines = []
        for line in str(text).splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("WARNING:"):
                continue
            if _looks_like_runtime_log(s):
                continue
            cleaned_lines.append(s)
        # Keep model line breaks (reads more human than one flattened paragraph).
        cleaned = "\n".join(cleaned_lines).strip()
        cleaned = re.sub(r"[,\s]*,[,\s]*,", ", ", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(
            r"^(here are\s+\d+\s+short sentences[^:]*:\s*|in summary[:,]?\s*|this note[:,]?\s*)",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()
        cleaned = re.sub(
            r"^(here'?s\s+the\s+note:?\s*|here\s+is\s+the\s+note:?\s*|note:?\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()
        if len(cleaned) > 1200:
            cleaned = cleaned[:1200].rstrip() + " ..."
        return cleaned

    def _ensure_analyst_facts_in_note(text: str, ctx: dict) -> str:
        """If the model skips JSON facts, patch minimally to avoid duplicate PAYMENT openers."""
        t = str(text or "").strip()
        if not t:
            return t
        ttype = str(ctx.get("transaction_type", "") or "").strip()
        amt = str(ctx.get("amount", "") or "").strip()
        pct = str(ctx.get("fraud_probability_percent", "") or "").strip()
        if not (ttype and amt and pct):
            return t

        def _has_amount(s: str) -> bool:
            return amt in s or amt.replace(",", "") in s.replace(",", "")

        def _pct_present(s: str) -> bool:
            if pct in s:
                return True
            raw = ctx.get("fraud_probability_value")
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return False
            for alt in (
                f"{v:.0%}",
                f"{v:.1%}",
                f"{v:.2%}",
                f"{v:.4%}",
            ):
                if alt in s:
                    return True
            return False

        has_type = ttype in t
        has_amt = _has_amount(t)
        has_pct = _pct_present(t)
        if has_type and has_amt and has_pct:
            return t

        # Type+amount already written; only probability formatting differed — lead with %, do not trail after action.
        if has_type and has_amt and not has_pct:
            head = f"Calibrated fraud probability is {pct}. "
            if t.lower().startswith(head.lower().strip()):
                return t
            return (head + t).strip()

        prefix = f"This looks like a legitimate {ttype} for ${amt} with fraud probability {pct}. "
        rest = t
        low = rest.lower()
        if low.startswith("this looks like") and ttype in rest[:120] and _has_amount(rest[:220]):
            cut = rest.find(".")
            if cut != -1:
                rest = rest[cut + 1 :].strip()
        return (prefix + rest).strip()

    def _strip_contradictory_green_hedge(text: str, ctx: dict) -> str:
        """Drop scare-hedge sentences that clash with GREEN + near-zero fraud probability."""
        if str(ctx.get("final_bucket", "")).upper() != "GREEN":
            return text
        t = str(text or "").strip()
        if not t:
            return t
        parts = re.split(r"(?<=[.!?])\s+", t)
        kept: list[str] = []
        for seg in parts:
            sl = seg.lower()
            if "however" in sl and ("risk indicator" in sl or "stands out" in sl):
                continue
            kept.append(seg)
        out = " ".join(s for s in kept if s).strip()
        return out or t

    def _polish_analyst_note(text: str, ctx: dict) -> str:
        t = _clean_summary_text(text)
        t = _ensure_analyst_facts_in_note(t, ctx)
        t = _strip_contradictory_green_hedge(t, ctx)
        return t

    # Preferred path: local Ollama HTTP APIs (try multiple compatible endpoints).
    http_attempts = [
        (
            "http://127.0.0.1:11434/api/generate",
            {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.65},
            },
            lambda d: str(d.get("response", "")),
        ),
        (
            "http://127.0.0.1:11434/api/chat",
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.65},
            },
            lambda d: str((d.get("message") or {}).get("content", "")),
        ),
        (
            "http://127.0.0.1:11434/v1/chat/completions",
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.65,
            },
            lambda d: str(((d.get("choices") or [{}])[0].get("message") or {}).get("content", "")),
        ),
    ]
    for url, payload, extractor in http_attempts:
        try:
            req = urlrequest.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=35) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
            text = _polish_analyst_note(extractor(data), context)
            if text and not _looks_like_runtime_log(text):
                return {"text": text, "source": "Ollama"}
        except (urlerror.HTTPError, urlerror.URLError, TimeoutError, json.JSONDecodeError, OSError):
            # Try next HTTP style; if all fail we'll fallback to CLI.
            continue

    # Fallback: CLI invocation.
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
    except FileNotFoundError:
        return {
            "text": "Ollama is not available on this machine. Start/install Ollama, then try again.",
            "source": "Fallback",
        }
    except subprocess.TimeoutExpired:
        return {
            "text": "Ollama response timed out. Please try again or switch to a smaller local model.",
            "source": "Fallback",
        }
    except Exception:
        return {"text": "Could not generate analyst summary right now. Please retry.", "source": "Fallback"}

    if result.returncode != 0:
        return {
            "text": (
                "Could not generate analyst summary right now. "
                "Please confirm Ollama is running and the selected model is available."
            ),
            "source": "Fallback",
        }

    text = _polish_analyst_note(result.stdout or "", context)
    if not text or _looks_like_runtime_log(text):
        return {"text": _fallback_summary(context), "source": "Fallback"}
    return {"text": text, "source": "Ollama"}


def transparent_decision_lines(
    probability: float,
    bucket: str,
    is_chain_member: int,
    chain_size: int,
    thresholds: dict,
    risk_up: list[str],
    risk_down: list[str],
) -> tuple[str, str, str, str, str]:
    review_t = float(thresholds["review_threshold"])
    block_t = float(thresholds["block_threshold"])
    moderate_t = float(thresholds["moderate_cutoff"])
    b = bucket.upper()

    top_up = risk_up[0] if risk_up else "no strong fraud-increasing local drivers"
    top_down = risk_down[0] if risk_down else "no strong fraud-reducing local drivers"
    chain_state = "active" if int(is_chain_member) == 1 else "inactive"

    if b == "GREEN":
        rule_line = (
            f"Decision rule: {probability:.2%} is below review threshold {review_t:.2%}, so transaction is allowed."
        )
        human_line = (
            f"Because calibrated risk is low ({probability:.2%} < {review_t:.2%}), chain signal is {chain_state}, "
            f"and top local evidence shows {top_down}."
        )
    elif b == "YELLOW":
        rule_line = (
            f"Decision rule: {review_t:.2%} <= {probability:.2%} < {block_t:.2%}, so transaction goes to manual review."
        )
        human_line = (
            f"Because calibrated risk is in review band ({probability:.2%}), chain signal is {chain_state}, "
            f"and local driver to check is {top_up}."
        )
    else:
        if probability >= block_t:
            rule_line = (
                f"Decision rule: {probability:.2%} is at/above block threshold {block_t:.2%}, so transaction is blocked."
            )
            human_line = (
                f"Because calibrated risk is high ({probability:.2%} >= {block_t:.2%}), "
                f"chain signal is {chain_state}, and strongest local risk signal is {top_up}."
            )
        else:
            rule_line = (
                f"Decision rule: chain escalation fired (chain member with risk >= {moderate_t:.2%}), so transaction is blocked."
            )
            human_line = (
                f"Because chain escalation fired (risk {probability:.2%} >= {moderate_t:.2%} with active chain signal), "
                f"and strongest local risk signal is {top_up}."
            )

    chain_line = (
        f"Chain signal: {'active' if int(is_chain_member) == 1 else 'inactive'} (chain_size={int(chain_size)}; cap applied in pipeline)."
    )
    up_text = ", ".join(risk_up[:1]) if risk_up else "No strong fraud-increasing local drivers"
    down_text = ", ".join(risk_down[:1]) if risk_down else "No material fraud-increasing local drivers detected"
    return rule_line, chain_line, f"What increased risk: {up_text}.", f"What lowered risk: {down_text}.", human_line


def feature_to_plain_text(feature_name: str) -> str:
    f = str(feature_name)
    mapping = {
        "num__is_chain_member": "Chain signal active",
        "num__chain_size": "Large linked transaction chain",
        "num__orig_residual": "Unusual sender balance movement",
        "num__orig_delta": "Unexpected sender balance delta",
        "num__dest_delta": "Unexpected receiver balance delta",
        "num__log_amount": "Risky transaction amount pattern",
        "num__step": "High-risk timing pattern",
        "num__orig_zero_new": "Sender balance drops to zero",
        "num__dest_zero_old": "Receiver started at zero balance",
    }
    if f in mapping:
        return mapping[f]
    if "cat__type_" in f:
        return f"Risky transaction type ({f.split('cat__type_')[-1]})"
    return "Elevated model risk signal"


def compute_local_contribution_matrix(base_model, x_proc: np.ndarray):
    """Return class-1 local contribution matrix if available, else (None, reason)."""
    try:
        import shap  # type: ignore
    except Exception:
        return None, "SHAP unavailable (package not installed)"

    try:
        # Build TreeExplainer once per loaded model to reduce repeat latency.
        model_id = id(base_model)
        cached_model_id = st.session_state.get("_shap_explainer_model_id")
        explainer = st.session_state.get("_shap_explainer_obj")
        if explainer is None or cached_model_id != model_id:
            explainer = shap.TreeExplainer(base_model)
            st.session_state["_shap_explainer_obj"] = explainer
            st.session_state["_shap_explainer_model_id"] = model_id
        raw = explainer.shap_values(x_proc, check_additivity=False)
    except Exception:
        return None, "SHAP unavailable (explainer failed)"

    if isinstance(raw, list):
        if len(raw) >= 2:
            return np.asarray(raw[1], dtype=np.float64), "Tree SHAP (class-1 local attributions)"
        if len(raw) == 1:
            return np.asarray(raw[0], dtype=np.float64), "Tree SHAP (single-output local attributions)"
        return None, "SHAP unavailable (empty output)"

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 3 and raw.shape[-1] >= 2:
        return raw[:, :, 1], "Tree SHAP (class-1 local attributions)"
    if raw.ndim == 2:
        return raw, "Tree SHAP (single-output local attributions)"
    return None, "SHAP unavailable (unexpected output shape)"


def top_local_driver_lists(
    contribution_row: np.ndarray,
    processed_feature_names: list[str],
    top_k: int = 3,
) -> tuple[list[str], list[str]]:
    # Positive contribution => pushes probability toward fraud class.
    max_abs = float(np.max(np.abs(contribution_row))) if contribution_row.size else 0.0
    min_strength = max(0.02, 0.12 * max_abs)
    pos_idx = np.where(contribution_row > 0)[0]
    neg_idx = np.where(contribution_row < 0)[0]

    pos_sorted = pos_idx[np.argsort(contribution_row[pos_idx])[::-1]] if len(pos_idx) else np.array([], dtype=int)
    neg_sorted = neg_idx[np.argsort(contribution_row[neg_idx])] if len(neg_idx) else np.array([], dtype=int)

    up = [
        feature_to_plain_text(processed_feature_names[i])
        for i in pos_sorted
        if contribution_row[i] >= min_strength
    ][:top_k]
    down = [
        feature_to_plain_text(processed_feature_names[i])
        for i in neg_sorted
        if abs(contribution_row[i]) >= min_strength
    ][:top_k]
    return up, down


def _finite_shap_float(x) -> float:
    """JSON-safe SHAP values (NaN/Inf break json.loads in the UI)."""
    v = float(x)
    return 0.0 if not np.isfinite(v) else v


def top_local_contributor_rows(
    contribution_row: np.ndarray,
    processed_feature_names: list[str],
    top_k: int = 8,
) -> list[dict]:
    top_idx = np.argsort(np.abs(contribution_row))[::-1][:top_k]
    rows = []
    for i in top_idx:
        sv = _finite_shap_float(contribution_row[i])
        rows.append(
            {
                "feature": str(processed_feature_names[i]),
                "plain_label": feature_to_plain_text(processed_feature_names[i]),
                "shap_value": sv,
                "abs_shap": abs(sv),
            }
        )
    return rows


def build_case_consistent_chips(pred_row: dict, tx_type: str) -> list[str]:
    amount = float(pred_row.get("amount", 0.0))
    old_org = float(pred_row.get("oldbalanceOrg", 0.0))
    new_org = float(pred_row.get("newbalanceOrig", 0.0))
    orig_residual = float(pred_row.get("orig_residual", 0.0))
    is_chain = int(pred_row.get("is_chain_member", 0))

    chips: list[str] = []
    chips.append("Chain signal active" if is_chain == 1 else "Chain signal inactive")

    if old_org > 0 and new_org == 0:
        chips.append("Sender balance drops to zero")
    elif abs(orig_residual) > max(500.0, 0.15 * max(amount, 1.0)):
        chips.append("Suspicious balance movement")
    else:
        chips.append("Balance movement looks consistent")

    if tx_type in {"TRANSFER", "CASH_OUT"}:
        chips.append(f"Transaction type: {tx_type} (higher-risk flow)")
    elif tx_type == "PAYMENT":
        chips.append("Transaction type: PAYMENT (typically lower-risk)")
    else:
        chips.append(f"Transaction type: {tx_type}")

    return chips[:3]


def run_inference(
    df_input: pd.DataFrame,
    preprocessor,
    base_model,
    cal_model,
    metadata: dict,
    chain_mode: str,
    fallback_chain_size: int = 1,
    fallback_is_chain_member: int = 0,
    include_local_shap: bool = True,
) -> pd.DataFrame:
    required_raw = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    missing = [c for c in required_raw if c not in df_input.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    chain_size_cap = int(metadata.get("chain_size_cap", 12))
    thresholds = metadata["triage_thresholds"]
    model_cols = metadata["input_feature_columns"]
    proc_names = metadata["processed_feature_names"]

    df_feat = add_engineered_features_for_inference(
        df_input,
        chain_size_cap,
        chain_mode=chain_mode,
        fallback_chain_size=fallback_chain_size,
        fallback_is_chain_member=fallback_is_chain_member,
    )
    X_model = df_feat[model_cols].copy()

    X_proc = preprocessor.transform(X_model)
    X_proc = np.asarray(X_proc, dtype=np.float64)
    X_proc = np.nan_to_num(np.clip(X_proc, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

    calibrated_prob = cal_model.predict_proba(X_proc)[:, 1]
    base_prob = base_model.predict_proba(X_proc)[:, 1]

    out = df_input.copy()
    engineered_debug_cols = [
        "log_amount",
        "orig_delta",
        "dest_delta",
        "orig_residual",
        "orig_zero_old",
        "dest_zero_old",
        "orig_zero_new",
        "dest_zero_new",
        "is_chain_member",
        "chain_size",
    ]
    for c in engineered_debug_cols:
        if c in df_feat.columns:
            out[c] = df_feat[c]
    out["base_probability"] = base_prob
    out["base_rf_probability"] = base_prob  # legacy column name / CSV schemas
    out["calibrated_probability"] = calibrated_prob
    out["chain_size"] = out["chain_size"].astype(int)
    out["is_chain_member"] = out["is_chain_member"].astype(int)
    out["triage_bucket"] = [
        apply_triage_rule(p, cm, thresholds)
        for p, cm in zip(out["calibrated_probability"], out["is_chain_member"])
    ]
    contrib_matrix = None
    explain_method = "Local SHAP disabled for performance"
    if include_local_shap:
        contrib_matrix, explain_method = compute_local_contribution_matrix(base_model, X_proc)
    risk_up_cols: list[str] = []
    risk_down_cols: list[str] = []
    reasons: list[str] = []
    for i in range(len(out)):
        if include_local_shap and contrib_matrix is not None and i < len(contrib_matrix):
            up, down = top_local_driver_lists(contrib_matrix[i], proc_names, top_k=3)
            top_rows = top_local_contributor_rows(contrib_matrix[i], proc_names, top_k=8)
        else:
            up, down = [], []
            top_rows = []

        risk_up_cols.append(" | ".join(up) if up else "None")
        risk_down_cols.append(" | ".join(down) if down else "None")
        out.loc[out.index[i], "local_shap_top"] = json.dumps(top_rows, allow_nan=False)
        reasons.append(
            build_reason_block(
                float(out.iloc[i]["calibrated_probability"]),
                str(out.iloc[i]["triage_bucket"]),
                int(out.iloc[i]["chain_size"]),
                int(out.iloc[i]["is_chain_member"]),
                explain_method,
                up,
                down,
            )
        )
    out["risk_up_drivers"] = risk_up_cols
    out["risk_down_drivers"] = risk_down_cols
    out["reason"] = reasons
    return out


def drift_csv_cache_key() -> str:
    """Invalidate drift loader when path or file contents (mtime) change."""
    p = DATA_PATH
    if not p.exists():
        return f"{p}:missing"
    return f"{p}:{p.stat().st_mtime_ns}"


@st.cache_data(show_spinner=False)
def load_drift_source_df(_csv_cache_key: str) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at `{DATA_PATH}`. "
            "Place the PaySim CSV in the project root or set `PAYSIM_CSV` / `PAYSIM_DATA_PATH`."
        )
    use_cols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]
    return pd.read_csv(DATA_PATH, usecols=use_cols)


def psi_numeric(early: pd.Series, late: pd.Series, bins: int = 10) -> float:
    e = pd.to_numeric(early, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    l = pd.to_numeric(late, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if e.empty or l.empty:
        return 0.0
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(e, q)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    e_counts, _ = np.histogram(e, bins=edges)
    l_counts, _ = np.histogram(l, bins=edges)
    e_pct = e_counts / max(e_counts.sum(), 1)
    l_pct = l_counts / max(l_counts.sum(), 1)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    l_pct = np.clip(l_pct, eps, None)
    return float(np.sum((l_pct - e_pct) * np.log(l_pct / e_pct)))


def psi_binary(early: pd.Series, late: pd.Series) -> float:
    e_counts = early.fillna(0).astype(int).value_counts(normalize=True)
    l_counts = late.fillna(0).astype(int).value_counts(normalize=True)
    levels = sorted(set(e_counts.index).union(set(l_counts.index)))
    eps = 1e-6
    total = 0.0
    for lvl in levels:
        e = max(float(e_counts.get(lvl, 0.0)), eps)
        l = max(float(l_counts.get(lvl, 0.0)), eps)
        total += (l - e) * np.log(l / e)
    return float(total)


def drift_level(psi_value: float) -> str:
    if psi_value < 0.1:
        return "GREEN"
    if psi_value <= 0.2:
        return "YELLOW"
    return "RED"

def render_banner(
    image_path: str | Path,
    height: int = 280,
    fill_remaining: bool = False,
    viewport_offset_px: int = 360,
    fit_mode: str = "cover",
):
    """Render a responsive full-width hero/banner with clean cropping."""
    path = Path(image_path)
    if not path.exists():
        return

    ext = path.suffix.lower()
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    h_desktop = max(260, min(int(height), 320))
    h_tablet = max(220, h_desktop - 30)
    h_mobile = max(190, h_desktop - 60)
    if fill_remaining:
        # Fill the visible area left after title/tabs + top KPI strip.
        # `clamp` prevents extreme sizes on very small/large displays.
        height_css = f"clamp(260px, calc(100vh - {int(viewport_offset_px)}px), 820px)"
        height_css_tablet = f"clamp(220px, calc(100vh - {int(viewport_offset_px) + 20}px), 720px)"
        height_css_mobile = f"clamp(180px, calc(100vh - {int(viewport_offset_px) + 40}px), 600px)"
    else:
        height_css = f"{h_desktop}px"
        height_css_tablet = f"{h_tablet}px"
        height_css_mobile = f"{h_mobile}px"

    fit_mode_norm = str(fit_mode).lower()
    if fit_mode_norm == "contain":
        bg_fit = "contain"
    elif fit_mode_norm == "width":
        bg_fit = "100% auto"
    else:
        bg_fit = "cover"
    banner_key = hashlib.md5(f"{path.resolve()}::{h_desktop}::{fill_remaining}::{bg_fit}".encode("utf-8")).hexdigest()[:10]
    outer_class = f"hero-banner-outer-{banner_key}"
    wrapper_class = f"hero-banner-wrapper-{banner_key}"
    st.markdown(
        f"""
        <style>
          .{outer_class} {{
            width: 100%;
            max-width: 100%;
            margin: 0;
            padding: 0;
          }}
          .{wrapper_class} {{
            width: 100%;
            max-width: 100%;
            height: {height_css};
            overflow: hidden !important;
            border-radius: 18px;
            margin: 0;
            padding: 0;
            display: block;
            background-image: url("data:{mime};base64,{b64}");
            background-size: {bg_fit};
            background-position: center center;
            background-repeat: no-repeat;
            background-color: #060d1a;
          }}
          @media (max-width: 1200px) {{
            .{wrapper_class} {{ height: {height_css_tablet}; }}
          }}
          @media (max-width: 768px) {{
            .{wrapper_class} {{ height: {height_css_mobile}; }}
          }}
        </style>
        <div class="{outer_class}">
          <div class="{wrapper_class}" role="img" aria-label="Hero banner"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_banner_full_width(image_path: str | Path):
    """Render image full width with natural aspect ratio (no crop)."""
    path = Path(image_path)
    if not path.exists():
        return
    ext = path.suffix.lower()
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div style="width:100%;max-width:100%;overflow:hidden;border-radius:18px;margin:0;padding:0;">
          <img
            src="data:{mime};base64,{b64}"
            alt="Command center banner"
            style="width:100%;height:auto;display:block;margin:0;padding:0;border:0;max-width:100%;"
          />
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_banner_html(image_path: str | Path, border_radius_px: int = 18) -> str:
    """Return inline HTML for a full-width banner image."""
    path = Path(image_path)
    if not path.exists():
        return ""
    ext = path.suffix.lower()
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return (
        f"<div class='cc-banner-wrap' style='width:100%;max-width:100%;overflow:hidden;"
        f"border-radius:{int(border_radius_px)}px;margin:0;padding:0;'>"
        f"<img src='data:{mime};base64,{b64}' alt='Command center banner' "
        "style='width:100%;height:auto;display:block;margin:0;padding:0;border:0;max-width:100%;' />"
        "</div>"
    )


def main():
    st.set_page_config(page_title="PaySim Fraud Triage App", page_icon="🛡️", layout="wide")
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp, section.main, [data-testid="stAppViewContainer"] > .main {
            overflow-x: hidden !important;
        }
        .stApp {
            background: #0b1220 !important;
            color: #e9f1ff !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        section.main > div.block-container {
            max-width: 100% !important;
        }
        [data-testid="stImage"] img {
            max-width: 100% !important;
            height: auto !important;
            display: block !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            min-width: 0 !important;
        }
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, label {
            color: #d6e2f7 !important;
        }
        .block-container {
            padding-top: 1.05rem !important;
            padding-bottom: 0.45rem !important;
        }
        h1, h2, h3 {
            letter-spacing: 0.2px;
            margin-bottom: 0.2rem !important;
            line-height: 1.15 !important;
        }
        h3 {
            margin-top: 0.2rem !important;
            font-size: 1.2rem !important;
            color: #eef3ff !important;
        }
        button[data-baseweb="tab"] {
            padding: 0.46rem 0.92rem !important;
            border-radius: 10px !important;
            min-height: 36px !important;
            margin-right: 0.22rem !important;
            border: 1px solid #2b3a52 !important;
            background: #101726 !important;
        }
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.1px !important;
            color: #dbe8ff !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: #172843 !important;
            border: 1px solid #4f9bff !important;
            box-shadow: 0 0 0 2px rgba(79, 155, 255, 0.16) !important;
        }
        button[data-baseweb="tab"][aria-selected="false"] > div[data-testid="stMarkdownContainer"] p {
            color: #c5d5f3 !important;
            opacity: 0.95 !important;
        }
        button[data-baseweb="tab"]:hover {
            border-color: #4a6388 !important;
            background: #132036 !important;
        }
        [data-testid="stButton"] button {
            border-radius: 8px !important;
            border: 1px solid #2f3c52 !important;
            background: #111b2b !important;
            color: #dbe8ff !important;
        }
        [data-testid="stButton"] button:hover {
            border-color: #4a6388 !important;
            background: #15243a !important;
            color: #eef4ff !important;
        }
        [data-testid="stButton"] button[kind="primary"] {
            background: #ff4b4b !important;
            border: 1px solid #ff6a6a !important;
            color: #ffffff !important;
        }
        [data-testid="stButton"] button[kind="primary"]:hover {
            background: #ff5c5c !important;
            border-color: #ff7777 !important;
        }
        /* keep download/uploader action buttons visible on dark theme */
        [data-testid="stDownloadButton"] button,
        [data-testid="stFileUploader"] section button {
            border-radius: 8px !important;
            border: 1px solid #4a6388 !important;
            background: #1c2d48 !important;
            color: #eef4ff !important;
            opacity: 1 !important;
        }
        [data-testid="stDownloadButton"] button:hover,
        [data-testid="stFileUploader"] section button:hover {
            border-color: #6fa8ff !important;
            background: #24395a !important;
            color: #ffffff !important;
        }
        [data-testid="stDownloadButton"] button p,
        [data-testid="stFileUploader"] section button p,
        [data-testid="stDownloadButton"] button span,
        [data-testid="stFileUploader"] section button span {
            color: #eef4ff !important;
            -webkit-text-fill-color: #eef4ff !important;
            opacity: 1 !important;
        }
        /* smoother form controls */
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            border: 1px solid #2f3c52 !important;
            background: #1a2a42 !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
            min-height: 38px !important;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: #4a6388 !important;
            background: #172133 !important;
        }
        div[data-baseweb="select"] > div:focus-within {
            border-color: #4f9bff !important;
            box-shadow: 0 0 0 2px rgba(79, 155, 255, 0.18) !important;
            background: #203453 !important;
        }
        div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
        div[data-baseweb="select"] div[role="combobox"] *,
        div[data-baseweb="select"] span {
            color: #e8f0ff !important;
        }
        div[data-baseweb="select"] div[role="combobox"],
        div[data-baseweb="select"] div[role="combobox"] * {
            color: #e8f0ff !important;
            -webkit-text-fill-color: #e8f0ff !important;
            opacity: 1 !important;
        }
        div[data-baseweb="select"] > div > div,
        div[data-baseweb="select"] > div > div * {
            color: #e8f0ff !important;
            -webkit-text-fill-color: #e8f0ff !important;
            opacity: 1 !important;
        }
        div[data-baseweb="select"] input {
            color: #e7efff !important;
        }
        /* smoother number/text inputs */
        div[data-baseweb="input"] > div {
            border-radius: 10px !important;
            border: 1px solid #2f3c52 !important;
            background: #141b28 !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
            min-height: 38px !important;
        }
        div[data-baseweb="input"] > div:hover {
            border-color: #4a6388 !important;
            background: #172133 !important;
        }
        div[data-baseweb="input"] > div:focus-within {
            border-color: #4f9bff !important;
            box-shadow: 0 0 0 2px rgba(79, 155, 255, 0.18) !important;
            background: #18253a !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="input"] textarea,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: #e8f0ff !important;
        }
        [data-testid="stCaptionContainer"] {
            color: #b8c6df !important;
        }
        /* metric readability (used in Drift tab PR-AUC/Delta cards) */
        [data-testid="stMetricLabel"] {
            color: #d6e2f7 !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #d6e2f7 !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricValue"] {
            color: #f4f8ff !important;
            opacity: 1 !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricValue"] * {
            color: #f4f8ff !important;
            -webkit-text-fill-color: #f4f8ff !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricDelta"] {
            opacity: 1 !important;
        }
        [data-testid="stMetricDelta"] * {
            opacity: 1 !important;
        }
        /* Streamlit 1.4x help/tooltip markdown (Popover body is data-testid=stTooltipContent) */
        div[data-testid="stTooltipContent"],
        div[data-testid="stTooltipErrorContent"] {
            background: #eef3ff !important;
            background-color: #eef3ff !important;
            color: #081018 !important;
            border: 1px solid #5a7fba !important;
            box-shadow: 0 8px 26px rgba(0, 0, 0, 0.45) !important;
            font-size: 0.9rem !important;
            line-height: 1.4 !important;
            z-index: 100055 !important;
        }
        div[data-testid="stTooltipContent"] *,
        div[data-testid="stTooltipErrorContent"] * {
            color: #081018 !important;
            -webkit-text-fill-color: #081018 !important;
            opacity: 1 !important;
        }
        div[data-testid="stTooltipContent"] p,
        div[data-testid="stTooltipErrorContent"] p,
        div[data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stTooltipErrorContent"] [data-testid="stMarkdownContainer"] p {
            color: #081018 !important;
            -webkit-text-fill-color: #081018 !important;
        }
        div[data-testid="stDataFrameTooltipContent"],
        div[data-testid="stDataFrameTooltipContent"] * {
            background-color: #eef3ff !important;
            color: #081018 !important;
            -webkit-text-fill-color: #081018 !important;
        }
        /* global tooltip readability (Streamlit BaseWeb + Radix portals + Vega) */
        [role="tooltip"],
        div[data-baseweb="tooltip"] {
            background: #f0f4ff !important;
            color: #0b1220 !important;
            border: 1px solid #6b8cc4 !important;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.38) !important;
            opacity: 1 !important;
            font-size: 0.9rem !important;
            z-index: 100050 !important;
        }
        [role="tooltip"] *,
        div[data-baseweb="tooltip"] * {
            color: #0b1220 !important;
            -webkit-text-fill-color: #0b1220 !important;
            opacity: 1 !important;
        }
        /* Streamlit help (?): often mounted in a Radix/floating-ui popper wrapper */
        [data-radix-popper-content-wrapper] {
            z-index: 100050 !important;
        }
        [data-radix-popper-content-wrapper] > div {
            background: #f0f4ff !important;
            color: #0b1220 !important;
            border: 1px solid #6b8cc4 !important;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.38) !important;
            font-size: 0.9rem !important;
            line-height: 1.35 !important;
        }
        [data-radix-popper-content-wrapper] p,
        [data-radix-popper-content-wrapper] span,
        [data-radix-popper-content-wrapper] div,
        [data-radix-popper-content-wrapper] li {
            color: #0b1220 !important;
            -webkit-text-fill-color: #0b1220 !important;
            opacity: 1 !important;
        }
        .vg-tooltip {
            background: #f8fbff !important;
            color: #0f172a !important;
            border: 1px solid #9fb6d9 !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
            font-size: 0.86rem !important;
        }
        .vg-tooltip td,
        .vg-tooltip th,
        .vg-tooltip span,
        .vg-tooltip div {
            color: #0f172a !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 112px;
            background: #111b2b !important;
            border: 1px dashed #35507a !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #d5e3fa !important;
        }
        [data-testid="stExpander"] details {
            border: 1px solid #2f3c52 !important;
            border-radius: 8px !important;
            background: #0f1726 !important;
        }
        [data-testid="stExpander"] summary {
            color: #d6e2f7 !important;
        }
        .kpi-card {
            border: 1px solid #2f3c52;
            border-radius: 12px;
            padding: 7px 10px;
            background: #121927;
            min-height: 66px;
        }
        .kpi-label {
            font-size: 0.78rem;
            color: #aab3c2;
            margin: 0 0 4px 0;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0;
            color: #f3f6ff;
        }
        .reason-chip {
            display: inline-block;
            margin: 4px 8px 4px 0;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid #33405a;
            background: #111826;
            color: #d9e4ff;
            font-size: 0.85rem;
        }
        .reason-chip-strong {
            display: inline-block;
            margin: 5px 8px 5px 0;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid #3c4f71;
            background: #132038;
            color: #e6eeff;
            font-size: 0.9rem;
            font-weight: 600;
        }
        label[data-testid="stWidgetLabel"] p {
            color: #c5cede !important;
            font-size: 0.82rem !important;
        }
        .compact-note {
            color: #c8d4ea;
            font-size: 0.95rem;
            margin-top: 4px;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.65rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    artifact_build_loading = (
        _env_flag("AUTO_REBUILD_IF_STALE", AUTO_REBUILD_IF_STALE_DEFAULT)
        and BUILD_SCRIPT_PATH.exists()
        and (_artifacts_missing() or _artifacts_stale())
        and DATA_PATH.exists()
    )

    # Rebuild path: only this message + spinner until build finishes; no tabs (they render below after load_artifacts).
    _artifact_build_placeholder = st.empty()
    rebuilt = False
    rebuild_note = ""
    try:
        if artifact_build_loading:
            with _artifact_build_placeholder.container():
                st.info(
                    "**Loading `build_artifacts.py`…** Please wait. "
                    "When this finishes, this message goes away and the app opens with tabs."
                )
            with st.spinner("Running `build_artifacts.py` — please wait…"):
                rebuilt, rebuild_note = maybe_auto_rebuild_artifacts()
        else:
            rebuilt, rebuild_note = maybe_auto_rebuild_artifacts()
    except Exception as e:
        # Non-fatal: app can still proceed with existing artifacts if present.
        st.warning(str(e))
    finally:
        if artifact_build_loading:
            _artifact_build_placeholder.empty()

    if rebuild_note and not rebuilt:
        st.caption(rebuild_note)

    try:
        preprocessor, base_model, cal_model, metadata = load_artifacts(build_artifact_cache_key())
    except Exception as e:
        st.error(str(e))
        st.stop()

    if TITLE_ICON_PATH.exists():
        title_icon_b64 = base64.b64encode(TITLE_ICON_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <h1 style="margin:0;display:flex;align-items:center;gap:16px;font-weight:900;">
              <span>PaySim Fraud Triage App</span>
              <img src="data:image/png;base64,{title_icon_b64}" alt="Calibration icon"
                   style="height:70px;width:auto;display:inline-block;vertical-align:middle;" />
            </h1>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("PaySim Fraud Triage App")
    st.caption("Fast demo for fraud triage decisions using the final calibrated model.")
    _dyn_bits = []
    if os.getenv("PAYSIM_CSV") or os.getenv("PAYSIM_DATA_PATH") or os.getenv("PAYSIM_CSV_NAME"):
        _dyn_bits.append(f"**Data:** `{DATA_PATH}`")
    if os.getenv("PAYSIM_NOTEBOOK"):
        _dyn_bits.append(f"**Notebook (stale check):** `{NOTEBOOK_PATH}`")
    if _dyn_bits:
        st.caption(" · ".join(_dyn_bits))

    overview_tab, dashboard_tab, batch_tab, drift_tab, model_card_tab = st.tabs(
        ["Command Center", "Dashboard", "Batch upload", "Drift Monitor", "Model Card"]
    )

    with overview_tab:
        final_key = str(metadata.get("resolved_final_model_key", metadata.get("final_model_key", "unknown")))
        t = metadata.get("triage_thresholds", {})
        review_t = float(t.get("review_threshold", 0.0))
        block_t = float(t.get("block_threshold", 0.0))
        moderate_t = float(t.get("moderate_cutoff", t.get("review_threshold", 0.0)))
        artifact_paths = [META_PATH, PREPROCESSOR_PATH]
        resolved_cal_file = str(metadata.get("resolved_calibration_model_file", "")).strip()
        if resolved_cal_file:
            artifact_paths.append(ARTIFACT_DIR / resolved_cal_file)
        existing_artifacts = [p for p in artifact_paths if p.exists()]
        artifact_updated_text = (
            datetime.fromtimestamp(max(p.stat().st_mtime for p in existing_artifacts)).strftime("%Y-%m-%d %H:%M")
            if existing_artifacts else "n/a"
        )

        st.caption("Demo landing page for triage flow, operating policy, and quick navigation.")

        if COMMAND_CENTER_IMAGE_PATH.exists():
            render_banner(COMMAND_CENTER_IMAGE_PATH, height=220, fit_mode="contain")
        else:
            st.warning("Command Center image not found at configured path.")

        st.caption(f"Artifacts on disk (reload picks up Jupyter exports): **{artifact_updated_text}**")

        st.markdown("#### Deployment Snapshot")
        s1, s2, s3, s4 = st.columns(4, gap="small")
        s1.markdown(
            f"<div class='kpi-card'><p class='kpi-label'>Final model</p><p class='kpi-value'>{escape(final_key)}</p></div>",
            unsafe_allow_html=True,
        )
        s2.markdown(
            f"<div class='kpi-card'><p class='kpi-label'>Review threshold</p><p class='kpi-value'>{review_t:.2f}</p></div>",
            unsafe_allow_html=True,
        )
        s3.markdown(
            f"<div class='kpi-card'><p class='kpi-label'>Block threshold</p><p class='kpi-value'>{block_t:.2f}</p></div>",
            unsafe_allow_html=True,
        )
        s4.markdown(
            "<div class='kpi-card'><p class='kpi-label'>Fraud prevalence</p><p class='kpi-value'>~0.13%</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### How To Use This App")
        b1, b2, b3, b4 = st.columns(4, gap="small")
        b1.button("Open Dashboard", use_container_width=True, key="cc_open_dashboard")
        b2.button("Open Batch Upload", use_container_width=True, key="cc_open_batch")
        b3.button("Open Drift Monitor", use_container_width=True, key="cc_open_drift")
        b4.button("Open Model Card", use_container_width=True, key="cc_open_model_card")
        n1, n2, n3, n4 = st.columns(4, gap="small")
        n1.markdown(
            "<div class='kpi-card'><p class='kpi-label'>Dashboard</p><p style='margin:0;color:#d5e4ff;'>"
            "Score one transaction and review decision reasoning."
            "</p></div>",
            unsafe_allow_html=True,
        )
        n2.markdown(
            "<div class='kpi-card'><p class='kpi-label'>Batch Upload</p><p style='margin:0;color:#d5e4ff;'>"
            "Upload CSV rows for fast bulk triage and export."
            "</p></div>",
            unsafe_allow_html=True,
        )
        n3.markdown(
            "<div class='kpi-card'><p class='kpi-label'>Drift Monitor</p><p style='margin:0;color:#d5e4ff;'>"
            "Track PSI and PR-AUC drift before performance degrades."
            "</p></div>",
            unsafe_allow_html=True,
        )
        n4.markdown(
            "<div class='kpi-card'><p class='kpi-label'>Model Card</p><p style='margin:0;color:#d5e4ff;'>"
            "Read assumptions, thresholds, and deployment notes."
            "</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Decision Logic At A Glance")
        d1, d2, d3 = st.columns(3, gap="small")
        d1.markdown(
            f"""
            <div class='kpi-card' style='min-height:108px;height:108px;'>
              <p class='kpi-label'><b>Calibrated Risk</b></p>
              <p style='margin:0;color:#d5e4ff;'>
                Model outputs fraud probability <b>p</b> (0–1) from <code>{escape(final_key)}</code>.<br>
                Bands from <code>feature_metadata.json</code>:<br>
                <b>p &lt; {review_t:.2f}</b> · <b>{review_t:.2f} ≤ p &lt; {block_t:.2f}</b> · <b>p ≥ {block_t:.2f}</b>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        d2.markdown(
            f"""
            <div class='kpi-card' style='min-height:108px;height:108px;'>
              <p class='kpi-label'><b>Chain-Aware Scoring</b></p>
              <p style='margin:0;color:#d5e4ff;'>
                Uses <code>chain_size</code> / <code>is_chain_member</code> (TRANSFER + CASH_OUT logic).<br>
                If bucket is not RED but chain is active and <b>p ≥ {moderate_t:.2f}</b>,
                policy escalates to <b>RED</b> (same rule as <code>apply_triage_rule</code>).
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        d3.markdown(
            f"""
            <div class='kpi-card' style='min-height:108px;height:108px;'>
              <p class='kpi-label'><b>Business Output</b></p>
              <p style='margin:0;color:#d5e4ff;'>
                <b>GREEN</b>: p &lt; {review_t:.2f}<br>
                <b>YELLOW</b>: {review_t:.2f} ≤ p &lt; {block_t:.2f}<br>
                <b>RED</b>: p ≥ {block_t:.2f} or chain-escalated (≥ {moderate_t:.2f} on chain)
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dashboard_tab:
        if HEADER_IMAGE_PATH.exists():
            render_banner_full_width(HEADER_IMAGE_PATH)
        st.markdown("<div style='margin-bottom:0.05rem;'></div>", unsafe_allow_html=True)
        st.caption(
            "Demo flow: choose an example (or enter values), click Score transaction, then review action and top reasons."
        )
        _t = metadata.get("triage_thresholds", {})
        _final_key = str(metadata.get("resolved_final_model_key", metadata.get("final_model_key", "unknown")))
        st.caption(
            "Pipeline state: "
            f"model=`{_final_key}` | "
            f"review={float(_t.get('review_threshold', 0.0)):.2f} | "
            f"block={float(_t.get('block_threshold', 0.0)):.2f} | "
            f"chain_cutoff={float(_t.get('moderate_cutoff', 0.0)):.2f}"
        )

        defaults = {
            "step": 1,
            "type": "PAYMENT",
            "amount": 1000.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4000.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 2000.0,
            "is_chain_member_demo": 0,
            "chain_size_demo": 1,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        if "last_pred_single" not in st.session_state:
            st.session_state["last_pred_single"] = None
        if "advanced_expander_reset" not in st.session_state:
            st.session_state["advanced_expander_reset"] = 0
        saved = st.session_state["last_pred_single"]
        current_prob = None if saved is None else float(saved["pred"]["calibrated_probability"])
        current_bucket = "NO PREDICTION" if saved is None else str(saved["pred"]["triage_bucket"]).upper()
        current_action = "--" if saved is None else bucket_meta(current_bucket)["label"]
        kpi_accent = "#2f3c52" if saved is None else bucket_meta(current_bucket)["color"]

        s1, s2, s3 = st.columns(3)
        _fk_esc = escape(str(_final_key))
        s1.markdown(
            f"""
            <div class="kpi-card" style="border-color:#3d5a86;box-shadow:0 0 0 1px rgba(79,155,255,0.12) inset;">
              <p class="kpi-label">Deploy calibrator</p>
              <p class="kpi-value">{_fk_esc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s2.markdown(
            f"""
            <div class="kpi-card" style="border-color:{kpi_accent};">
              <p class="kpi-label">Fraud score</p>
              <p class="kpi-value">{"Waiting for input" if current_prob is None else f"{current_prob:.2%}"}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s3.markdown(
            f"""
            <div class="kpi-card" style="border-color:{kpi_accent};">
              <p class="kpi-label">Final action</p>
              <p class="kpi-value">{"No decision yet" if saved is None else current_action}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([0.85, 1.35], gap="large")
        with left:
            st.subheader("Transaction details")
            st.caption("Quick examples")
            st.caption(
                "\"Fraud-like\" is tuned for mid–high risk under the current deploy scorer—a TRANSFER "
                "with mismatched balances (not canonical PaySim drain fraud, which scores almost 1 under CatBoost)."
            )
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                if st.button("Low risk", type="secondary", use_container_width=True):
                    st.session_state.update(
                        {
                            "step": 1, "type": "PAYMENT", "amount": 1200.0,
                            "oldbalanceOrg": 8000.0, "newbalanceOrig": 6800.0,
                            "oldbalanceDest": 4000.0, "newbalanceDest": 5200.0,
                            "is_chain_member_demo": 0, "chain_size_demo": 1,
                        }
                    )
            with ex2:
                if st.button("Fraud-like", type="secondary", use_container_width=True):
                    st.session_state.update(
                        {
                            # Calibrated ~65% under catboost_plain_sigmoid (distinct from canonical drain TRANSFER≈100%).
                            "step": 200,
                            "type": "TRANSFER",
                            "amount": 6296.24,
                            "oldbalanceOrg": 7608.66,
                            "newbalanceOrig": 0.0,
                            "oldbalanceDest": 127395.62,
                            "newbalanceDest": 128385.80,
                            "is_chain_member_demo": 0,
                            "chain_size_demo": 1,
                        }
                    )
            with ex3:
                if st.button("Chain-aware", type="secondary", use_container_width=True):
                    st.session_state.update(
                        {
                            "step": 1, "type": "CASH_OUT", "amount": 181.0,
                            "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
                            "oldbalanceDest": 0.0, "newbalanceDest": 181.0,
                            "is_chain_member_demo": 1, "chain_size_demo": 2,
                        }
                    )
            with st.form("single_tx_form"):
                c1, c2 = st.columns(2)
                with c1:
                    step = st.number_input("Step", min_value=1, step=1, key="step")
                    tx_type = st.selectbox("Transaction type", ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT", "CASH_IN"], key="type")
                    amount = st.number_input("Amount", min_value=0.0, step=100.0, key="amount")
                    oldbalanceOrg = st.number_input("Sender balance (before)", min_value=0.0, step=100.0, key="oldbalanceOrg")
                with c2:
                    newbalanceOrig = st.number_input("Sender balance (after)", min_value=0.0, step=100.0, key="newbalanceOrig")
                    oldbalanceDest = st.number_input("Receiver balance (before)", min_value=0.0, step=100.0, key="oldbalanceDest")
                    newbalanceDest = st.number_input("Receiver balance (after)", min_value=0.0, step=100.0, key="newbalanceDest")

                adv_label = f"Advanced settings{' ' * int(st.session_state['advanced_expander_reset'])}"
                with st.expander(adv_label, expanded=False):
                    fallback_is_chain_member = st.selectbox(
                        "Chain member flag", [0, 1], key="is_chain_member_demo"
                    )
                    fallback_chain_size = st.number_input(
                        "Chain size", min_value=1, step=1, key="chain_size_demo"
                    )
                st.caption("Primary action")
                submit = st.form_submit_button("Score transaction", type="primary", use_container_width=True)

            if submit:
                one = pd.DataFrame([{
                    "step": int(step),
                    "type": tx_type,
                    "amount": float(amount),
                    "oldbalanceOrg": float(oldbalanceOrg),
                    "newbalanceOrig": float(newbalanceOrig),
                    "oldbalanceDest": float(oldbalanceDest),
                    "newbalanceDest": float(newbalanceDest),
                }])
                pred = run_inference(
                    one, preprocessor, base_model, cal_model, metadata,
                    chain_mode="fallback",
                    fallback_chain_size=int(fallback_chain_size),
                    fallback_is_chain_member=int(fallback_is_chain_member),
                    include_local_shap=True,
                ).iloc[0]
                st.session_state["last_pred_single"] = {
                    "pred": pred.to_dict(),
                    "tx_type": tx_type,
                }
                # Force a fresh collapsed render of Advanced settings after scoring.
                st.session_state["advanced_expander_reset"] += 1

        with right:
            st.subheader("Decision panel")
            saved = st.session_state["last_pred_single"]
            if saved is None:
                st.markdown(
                    """
                    <div style="padding:12px;border-radius:12px;border:1px solid #3a3a3a;background:#171717;min-height:146px;">
                      <h2 style="margin:0 0 10px 0;color:#d2d7e0;font-size:30px;">No prediction yet</h2>
                      <p style="margin:0;color:#aeb6c7;font-size:14px;">Enter transaction details and click <b>Score transaction</b>.</p>
                      <p style="margin:8px 0 0 0;color:#9a9a9a;">Fraud probability: --</p>
                      <p style="margin:2px 0 0 0;color:#9a9a9a;">Final action: --</p>
                      <p style="margin:2px 0 0 0;color:#9a9a9a;">Quick explanation: --</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(0)
            else:
                pred = saved["pred"]
                tx_type = str(pred.get("type", saved["tx_type"]))
                bucket = str(pred["triage_bucket"]).upper()
                meta = bucket_meta(bucket)
                prob_val = float(pred["calibrated_probability"])
                action_sentence = {
                    "GREEN": "Allow transaction",
                    "YELLOW": "Send for review",
                    "RED": "Block transaction",
                }.get(bucket, "Review transaction")
                badge_text = f"{bucket} / {meta['label'].title()}"
                confidence = "High" if prob_val >= 0.9 or prob_val <= 0.1 else ("Medium" if prob_val >= 0.7 or prob_val <= 0.3 else "Low")
                case_chips = build_case_consistent_chips(pred, tx_type)
                chip_1, chip_2, chip_3 = case_chips[0], case_chips[1], case_chips[2]
                up = [x.strip() for x in str(pred.get("risk_up_drivers", "")).split("|") if x.strip() and x.strip() != "None"]
                down = [x.strip() for x in str(pred.get("risk_down_drivers", "")).split("|") if x.strip() and x.strip() != "None"]
                rule_line, chain_line, up_line, down_line, human_line = transparent_decision_lines(
                    prob_val,
                    bucket,
                    int(pred["is_chain_member"]),
                    int(pred["chain_size"]),
                    metadata["triage_thresholds"],
                    up,
                    down,
                )
                st.markdown(
                    f"""
                    <div style="padding:14px;border-radius:12px;border:1px solid #333;background:#161616;min-height:184px;border-top:6px solid {meta['color']};">
                      <p style="margin:0 0 6px 0;font-size:13px;color:#bfc5d4;">Calibrated fraud probability</p>
                      <p style="margin:0 0 8px 0;font-size:46px;line-height:1.0;"><b>{prob_val:.2%}</b></p>
                      <p style="margin:0 0 8px 0;color:#aeb6c7;font-size:12px;">Calibrated score (from exported pipeline)</p>
                      <div style="display:inline-block;background:{meta['color']};color:#ffffff;padding:8px 16px;border-radius:999px;font-weight:800;margin:0 0 8px 0;">
                        {meta['emoji']} {badge_text}
                      </div>
                      <p style="margin:4px 0 10px 0;font-size:16px;opacity:0.98;"><b>{action_sentence}</b></p>
                      <div>
                        <span class="reason-chip-strong">{chip_1}</span>
                        <span class="reason-chip-strong">{chip_2}</span>
                        <span class="reason-chip-strong">{chip_3}</span>
                      </div>
                      <p style="margin:8px 0 4px 0;">{rule_line}</p>
                      <p style="margin:0 0 4px 0;">{chain_line}</p>
                      <p style="margin:0 0 4px 0;">{up_line}</p>
                      <p style="margin:0 0 4px 0;">{down_line}</p>
                      <p style="margin:0 0 4px 0;"><b>Because:</b> {human_line}</p>
                      <p style="margin:0 0 4px 0;color:#b9c7df;">Trace: input → calibrated score {prob_val:.2%} → threshold policy → chain rule → <b>{bucket}</b></p>
                      <p style="margin:0;">Score confidence: <b>{confidence}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(max(prob_val, 0.0), 1.0))
                st.markdown(
                    f"<div class='compact-note'><b>Next step:</b> {next_action_text(bucket)}</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Decision legend: GREEN = Allow | YELLOW = Review | RED = Block")

            with st.expander("Why this decision", expanded=False):
                if saved is not None:
                    pred = saved["pred"]
                    st.caption("Technical evidence behind this decision.")
                    st.info(
                        "Any analyst summary is for explanation only. "
                        "Fraud score and final action come from the calibrated ML pipeline."
                    )
                    chips = []
                    chips.extend(build_case_consistent_chips(pred, tx_type))
                    up = [x.strip() for x in str(pred.get("risk_up_drivers", "")).split("|") if x.strip() and x.strip() != "None"]
                    down = [x.strip() for x in str(pred.get("risk_down_drivers", "")).split("|") if x.strip() and x.strip() != "None"]
                    chips.extend([f"Risk up: {c}" for c in up[:2]])
                    chips.extend([f"Risk down: {c}" for c in down[:2]])
                    chips.append(f"Chain size: {int(pred['chain_size'])}")
                    chips.append(f"Final bucket: {str(pred['triage_bucket']).upper()}")
                    st.markdown(
                        "".join([f"<span class='reason-chip'>{c}</span>" for c in chips]),
                        unsafe_allow_html=True,
                    )

                    t = metadata["triage_thresholds"]
                    review_t = float(t["review_threshold"])
                    block_t = float(t["block_threshold"])
                    moderate_t = float(t["moderate_cutoff"])
                    p_cal = float(pred["calibrated_probability"])
                    p_base = float(pred.get("base_probability") or pred.get("base_rf_probability") or 0.0)
                    is_chain = int(pred["is_chain_member"]) == 1
                    chain_escalated = (
                        str(pred["triage_bucket"]).upper() == "RED"
                        and p_cal < block_t
                        and is_chain
                        and p_cal >= moderate_t
                    )

                    evidence_df = pd.DataFrame(
                        [
                            {
                                "item": (
                                    f"Uncalibrated tree probability (Tree SHAP booster: "
                                    f"{metadata.get('tree_shap_backend_class', '?')})"
                                ),
                                "value": f"{p_base:.2%}",
                            },
                            {"item": "Calibrated probability", "value": f"{p_cal:.2%}"},
                            {"item": "Review threshold", "value": f"{review_t:.0%}"},
                            {"item": "Block threshold", "value": f"{block_t:.0%}"},
                            {"item": "Moderate cutoff (chain rule)", "value": f"{moderate_t:.0%}"},
                            {"item": "Chain signal active", "value": "Yes" if is_chain else "No"},
                            {"item": "Chain escalation fired", "value": "Yes" if chain_escalated else "No"},
                        ]
                    )
                    st.dataframe(evidence_df, use_container_width=True, hide_index=True)

                    if up:
                        st.markdown("**Local SHAP drivers increasing fraud risk:**")
                        for i, label in enumerate(up[:3], 1):
                            st.markdown(f"{i}. {label}")
                    if down:
                        st.markdown("**Local SHAP drivers reducing fraud risk:**")
                        for i, label in enumerate(down[:3], 1):
                            st.markdown(f"{i}. {label}")

                    show_shap_plot = st.toggle(
                        "Show local SHAP visual (optional)",
                        value=False,
                        key="show_local_shap_plot",
                    )
                    if show_shap_plot:
                        raw = str(pred.get("local_shap_top", "[]"))
                        try:
                            shap_rows = json.loads(raw) if raw else []
                        except Exception:
                            shap_rows = []
                        if shap_rows:
                            st.caption(
                                "SHAP reading guide: red bars increase fraud risk, green bars decrease fraud risk; "
                                "the final bucket comes from the net calibrated score against policy thresholds."
                            )
                            shap_df = pd.DataFrame(shap_rows)
                            shap_df = shap_df.sort_values("shap_value", ascending=True)
                            shap_df["plot_label"] = shap_df["plain_label"].map(
                                lambda s: (s[:34] + "...") if len(str(s)) > 37 else str(s)
                            )
                            try:
                                import matplotlib.pyplot as plt  # type: ignore

                                colors = ["#2ca02c" if v < 0 else "#d62728" for v in shap_df["shap_value"]]
                                fig, ax = plt.subplots(figsize=(6.6, 3.1))
                                fig.patch.set_facecolor("#1a1d24")
                                ax.set_facecolor("#1a1d24")
                                ax.tick_params(colors="#e9f1ff")
                                ax.xaxis.label.set_color("#e9f1ff")
                                ax.title.set_color("#e9f1ff")
                                for spine in ax.spines.values():
                                    spine.set_color("#555555")
                                ax.barh(shap_df["plot_label"], shap_df["shap_value"], color=colors)
                                ax.axvline(0.0, color="#777777", linewidth=1)
                                ax.set_xlabel("Local SHAP contribution")
                                ax.set_title("Why this score moved (current transaction)")
                                fig.tight_layout()
                                st.pyplot(fig, clear_figure=True, use_container_width=True)
                                plt.close(fig)
                            except Exception as e:
                                st.caption(f"Could not render SHAP chart: {e}")
                            show_shap_table = st.toggle(
                                "Show SHAP details table",
                                value=False,
                                key="show_local_shap_table",
                            )
                            if show_shap_table:
                                st.dataframe(
                                    shap_df[["feature", "plain_label", "shap_value", "abs_shap"]],
                                    use_container_width=True,
                                    hide_index=True,
                                    height=320,
                                )
                        else:
                            st.caption("No local SHAP contribution values available for this prediction.")

                    show_debug = st.toggle("Show developer/debug details (optional)", value=False, key="show_debug_values")
                    if show_debug:
                        debug_cols = [
                            "log_amount", "orig_delta", "dest_delta", "orig_residual",
                            "orig_zero_old", "dest_zero_old", "orig_zero_new", "dest_zero_new",
                            "is_chain_member", "chain_size",
                        ]
                        debug_df = pd.DataFrame(
                            [{"feature": c, "value": pred[c]} for c in debug_cols]
                        )
                        st.dataframe(debug_df, use_container_width=True)

                    st.markdown("### LLM Analyst Summary")
                    st.caption(
                        "Optional local explanation layer. "
                        "Fraud score and final action come from the calibrated ML pipeline."
                    )
                    llm_model = st.selectbox(
                        "Local Ollama model",
                        options=["llama3.1:8b", "smollm:1.7b"],
                        index=0,
                        key="ollama_model_single",
                    )
                    pred_key = "|".join(
                        [
                            str(pred.get("step", "")),
                            str(pred.get("type", tx_type)),
                            f"{float(pred.get('amount', 0.0)):.2f}",
                            f"{float(pred.get('calibrated_probability', 0.0)):.6f}",
                            str(pred.get("triage_bucket", "")),
                            str(llm_model),
                            f"v{OLLAMA_ANALYST_PROMPT_VERSION}",
                        ]
                    )
                    if "ollama_summary_cache" not in st.session_state:
                        st.session_state["ollama_summary_cache"] = {}

                    if st.button("Generate analyst summary", key="generate_analyst_summary_single"):
                        prob = float(pred.get("calibrated_probability", 0.0))
                        context = {
                            "fraud_probability_percent": f"{prob:.2%}",
                            "fraud_probability_value": round(prob, 6),
                            "final_bucket": str(pred.get("triage_bucket", "")).upper(),
                            "final_action": next_action_text(str(pred.get("triage_bucket", ""))),
                            "transaction_type": str(pred.get("type", tx_type)),
                            "amount": f"{float(pred.get('amount', 0.0)):,.2f}",
                            "chain_signal": "active" if int(pred.get("is_chain_member", 0)) == 1 else "inactive",
                            "chain_size": int(pred.get("chain_size", 1)),
                            "thresholds": {
                                "review_threshold_percent": f"{review_t:.0%}",
                                "block_threshold_percent": f"{block_t:.0%}",
                                "moderate_cutoff_percent": f"{moderate_t:.0%}",
                            },
                            "top_reasons_risk_up": up[:3],
                            "top_reasons_risk_down": down[:3],
                        }
                        with st.spinner("Generating local analyst summary..."):
                            summary_payload = generate_ollama_summary(context=context, model_name=llm_model)
                        st.session_state["ollama_summary_cache"][pred_key] = {
                            "model": llm_model,
                            "summary": summary_payload.get("text", ""),
                            "source": summary_payload.get("source", "Unknown"),
                        }

                    cached_summary = st.session_state["ollama_summary_cache"].get(pred_key)
                    if cached_summary:
                        st.markdown(
                            f"<div class='compact-note'><b>Model:</b> {cached_summary['model']} | "
                            f"<b>Source:</b> {cached_summary.get('source', 'Unknown')}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "Analyst summary is generated by a local LLM for explanation only. "
                            "Fraud score and final action come from the calibrated ML pipeline."
                        )
                        _sum_txt = str(cached_summary.get("summary", ""))
                        st.markdown(
                            "<div style=\"white-space:pre-wrap;border-left:4px solid #4f9bff;"
                            "padding:12px 14px;border-radius:8px;background:rgba(79,155,255,0.12);"
                            f"color:#e9f1ff;\">{escape(_sum_txt)}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Score once to view explanation chips.")

        if submit:
            # Run after the decision panel renders so SHAP + metrics update on the same score action.
            # (Calling rerun earlier skipped `with right:` entirely.)
            st.rerun()

        with st.expander("Policy details (cost-sensitive thresholds)", expanded=False):
            t = metadata["triage_thresholds"]
            operating_t = float(t.get("operating_threshold", t.get("moderate_cutoff", t["block_threshold"])))
            st.write(
                f"Cost policy used for thresholding: false positive cost = {COST_FP}, "
                f"false negative cost = {COST_FN}."
            )
            st.write(
                f"Operating threshold = {operating_t:.2f}, "
                f"Review threshold = {float(t['review_threshold']):.2f}, "
                f"Block threshold = {float(t['block_threshold']):.2f}"
            )
            st.caption(
                "Fraud misses are costlier than false alerts, so thresholds are tuned to improve fraud capture."
            )

        with st.expander("Technical notes", expanded=False):
            final_key = str(metadata.get("resolved_final_model_key", metadata.get("final_model_key", "unknown")))
            final_reason = str(metadata.get("final_model_reason", "n/a"))
            _shap_cls_tn = str(metadata.get("tree_shap_backend_class", "?"))
            _shap_ex = str(metadata.get("tree_shap_baseline_explainer_hint", ""))
            st.write(
                f"Final deployed calibrator: `{final_key}` (from `feature_metadata.json`). "
                f"Tree SHAP runs on **`{_shap_cls_tn}`** when supported; see `tree_shap_baseline_explainer_hint` for details. "
                f"Current hint: {_shap_ex}"
            )
            st.write(f"Selection reason: {final_reason}")
            st.json(metadata["triage_thresholds"])
            st.write(
                "Manual mode uses fallback chain values because transaction-history state lookup "
                "is not connected in this first app version."
            )
            st.write(
                "Uses saved artifacts only (preprocessor + deploy calibrator; Tree SHAP uses the matched "
                "baseline when supported). No retraining in app."
            )

    with batch_tab:
        st.markdown(
            """
            <div style="padding:10px 12px;border-radius:10px;border:1px solid #2f3c52;background:#111826;margin:0 0 8px 0;">
              <div style="font-size:1.05rem;font-weight:700;color:#eaf1ff;margin-bottom:8px;">Batch scoring</div>
              <div style="display:flex;flex-wrap:wrap;gap:6px;">
                <span class="reason-chip">step</span>
                <span class="reason-chip">type</span>
                <span class="reason-chip">amount</span>
                <span class="reason-chip">oldbalanceOrg</span>
                <span class="reason-chip">newbalanceOrig</span>
                <span class="reason-chip">oldbalanceDest</span>
                <span class="reason-chip">newbalanceDest</span>
                <span class="reason-chip">chain_size</span>
                <span class="reason-chip">is_chain_member</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Batch mode expects precomputed chain-aware inputs (`chain_size`, `is_chain_member`) "
            "from your upstream feature pipeline/state store."
        )
        sample_df = pd.DataFrame(
            [
                {
                    "step": 1,
                    "type": "PAYMENT",
                    "amount": 1000.0,
                    "oldbalanceOrg": 5000.0,
                    "newbalanceOrig": 4000.0,
                    "oldbalanceDest": 1000.0,
                    "newbalanceDest": 2000.0,
                    "chain_size": 1,
                    "is_chain_member": 0,
                },
                {
                    "step": 1,
                    "type": "CASH_OUT",
                    "amount": 181.0,
                    "oldbalanceOrg": 181.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 181.0,
                    "chain_size": 2,
                    "is_chain_member": 1,
                },
            ]
        )
        st.download_button(
            "Download sample CSV",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_batch_input.csv",
            mime="text/csv",
        )
        file = st.file_uploader("Upload transactions CSV", type=["csv"])
        st.caption("Upload a CSV to score multiple transactions at once.")

        if file is not None:
            df = pd.read_csv(file)
            preds = run_inference(
                df,
                preprocessor,
                base_model,
                cal_model,
                metadata,
                chain_mode="provided",
                include_local_shap=False,
            )
            batch_summary = (
                preds.groupby("triage_bucket", as_index=False)
                .agg(count=("triage_bucket", "size"))
            )
            batch_summary["triage_bucket"] = pd.Categorical(
                batch_summary["triage_bucket"], categories=["GREEN", "YELLOW", "RED"], ordered=True
            )
            batch_summary = batch_summary.sort_values("triage_bucket")
            total_n = int(batch_summary["count"].sum()) if len(batch_summary) else 0
            batch_summary["pct"] = (
                (batch_summary["count"] / total_n * 100).round(2) if total_n > 0 else 0.0
            )
            batch_summary["pct"] = batch_summary["pct"].map(lambda x: f"{x:.2f}%")
            dot_map = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
            batch_summary["bucket"] = batch_summary["triage_bucket"].astype(str).map(
                lambda b: f"{dot_map.get(b, '⚪')} {b}"
            )
            batch_summary = batch_summary[["bucket", "count", "pct"]]
            st.subheader("Batch bucket summary")
            st.dataframe(batch_summary, use_container_width=True)

            show_cols = [
                "step",
                "type",
                "amount",
                "calibrated_probability",
                "triage_bucket",
                "is_chain_member",
                "chain_size",
                "reason",
            ]
            st.subheader("Prediction result")
            st.dataframe(preds[show_cols], use_container_width=True)
            st.download_button(
                "Download scored CSV",
                data=preds.to_csv(index=False).encode("utf-8"),
                file_name="scored_triage_output.csv",
                mime="text/csv",
            )
            st.subheader("Explanation section")
            st.caption(
                "Batch mode is optimized for fast scoring/export. "
                "Detailed SHAP + LLM explanation is available in the Dashboard (single transaction view)."
            )

        with st.expander("Technical notes / requirements", expanded=False):
            st.write(
                "Batch mode expects precomputed chain-aware inputs (`chain_size`, `is_chain_member`) "
                "from an upstream transaction-history/state pipeline."
            )
            st.write(
                "Uses saved artifacts only (preprocessor + calibrator). No retraining in app."
            )

    with drift_tab:
        st.subheader("Drift Monitor")
        st.caption("Early window = step <= 400, late window = step > 400. This is a monitoring-only view (no retraining).")
        if "drift_enabled" not in st.session_state:
            st.session_state["drift_enabled"] = False
        if "drift_cache_params" not in st.session_state:
            st.session_state["drift_cache_params"] = None
        if "drift_cache_payload" not in st.session_state:
            st.session_state["drift_cache_payload"] = None

        c_run, c_hide = st.columns([0.45, 0.55])
        with c_run:
            if st.button(
                "Run drift calculations now (loads drift dataset)",
                use_container_width=True,
                help="Runs heavy drift computation only when clicked (keeps other tabs fast).",
            ):
                st.session_state["drift_enabled"] = True
        with c_hide:
            if st.button("Hide/clear drift view", use_container_width=True):
                st.session_state["drift_enabled"] = False
                st.session_state["drift_cache_params"] = None
                st.session_state["drift_cache_payload"] = None

        if not st.session_state["drift_enabled"]:
            st.caption("Click 'Run drift calculations now' to compute PSI + PR-AUC delta (monitoring-only).")
        else:
            try:
                raw_df = load_drift_source_df(drift_csv_cache_key())
            except Exception as e:
                st.error(str(e))
                st.stop()

            early_all = raw_df[raw_df["step"] <= 400].copy()
            late_all = raw_df[raw_df["step"] > 400].copy()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Early fraud rate", f"{early_all['isFraud'].mean():.3%}", f"rows: {len(early_all):,}")
            with c2:
                st.metric("Late fraud rate", f"{late_all['isFraud'].mean():.3%}", f"rows: {len(late_all):,}")
            st.caption(
                "These rates use every row in each `step` window; the slider only changes the sample size for PSI and model drift below."
            )

            if "drift_seed" not in st.session_state:
                st.session_state["drift_seed"] = 42
            sample_n = st.select_slider(
                "Rows per window for drift scoring",
                options=[20000, 40000, 60000, 80000, 100000, 140000, 180000, 200000],
                value=100000,
            )
            cseed1, cseed2 = st.columns([0.35, 0.65])
            with cseed1:
                if st.button("Randomize sample", use_container_width=True):
                    st.session_state["drift_seed"] = int(np.random.randint(1, 1_000_000))
            with cseed2:
                st.caption(
                    f"Sampling seed: `{st.session_state['drift_seed']}` (change seed or rows to force visible recompute)"
                )
            simulate_drift = st.toggle(
                "Simulate drift scenario (demo only)",
                value=False,
                help="Applies synthetic shift to late-window sample so you can demonstrate drift alerts.",
            )
            current_params = (
                int(sample_n),
                int(st.session_state["drift_seed"]),
                bool(simulate_drift),
            )
            recompute = (
                st.session_state["drift_cache_payload"] is None
                or st.session_state["drift_cache_params"] != current_params
            )

            if recompute:
                with st.spinner("Computing drift metrics..."):
                    early = early_all.sample(
                        n=min(sample_n, len(early_all)),
                        random_state=int(st.session_state["drift_seed"]),
                    ).copy()
                    late = late_all.sample(
                        n=min(sample_n, len(late_all)),
                        random_state=int(st.session_state["drift_seed"]),
                    ).copy()
                    if simulate_drift:
                        # Stronger synthetic drift for reliable classroom visualization.
                        late["amount"] = late["amount"] * 8.0
                        late["oldbalanceOrg"] = np.maximum(0.0, late["oldbalanceOrg"] * 0.2)
                        late["newbalanceOrig"] = np.maximum(0.0, late["newbalanceOrig"] * 0.05)
                        late.loc[late.sample(frac=0.55, random_state=7).index, "type"] = "CASH_OUT"
                        late.loc[late.sample(frac=0.30, random_state=11).index, "type"] = "TRANSFER"

                    drift_df = pd.concat([early, late], axis=0).reset_index(drop=True)
                    drift_df["orig_delta"] = drift_df["oldbalanceOrg"] - drift_df["newbalanceOrig"]
                    drift_df["orig_residual"] = drift_df["orig_delta"] - drift_df["amount"]

                    chain_group = (
                        drift_df.groupby(["step", "amount"], as_index=False)
                        .agg(
                            chain_size=("type", "size"),
                            has_transfer=("type", lambda s: (s == "TRANSFER").any()),
                            has_cash_out=("type", lambda s: (s == "CASH_OUT").any()),
                        )
                    )
                    chain_group["is_chain_member"] = (
                        chain_group["has_transfer"]
                        & chain_group["has_cash_out"]
                        & (chain_group["chain_size"] <= int(metadata.get("chain_size_cap", 12)))
                    ).astype(np.int8)
                    drift_df = drift_df.merge(
                        chain_group[["step", "amount", "chain_size", "is_chain_member"]],
                        on=["step", "amount"],
                        how="left",
                    )
                    drift_df["chain_size"] = drift_df["chain_size"].fillna(1).astype(np.int32)
                    drift_df["is_chain_member"] = drift_df["is_chain_member"].fillna(0).astype(np.int8)

                    if simulate_drift:
                        late_mask = drift_df["step"] > 400
                        rng = np.random.default_rng(123)
                        force_chain = rng.random(late_mask.sum()) < 0.65
                        drift_df.loc[late_mask, "is_chain_member"] = np.where(
                            force_chain, 1, drift_df.loc[late_mask, "is_chain_member"]
                        ).astype(np.int8)
                        drift_df.loc[late_mask, "chain_size"] = np.where(
                            drift_df.loc[late_mask, "is_chain_member"] == 1,
                            np.maximum(drift_df.loc[late_mask, "chain_size"], 4),
                            drift_df.loc[late_mask, "chain_size"],
                        ).astype(np.int32)

                    early_s = drift_df[drift_df["step"] <= 400].copy()
                    late_s = drift_df[drift_df["step"] > 400].copy()
                    psi_rows = []
                    for feat in ["amount", "orig_delta", "is_chain_member", "orig_residual"]:
                        if feat == "is_chain_member":
                            psi_v = psi_binary(early_s[feat], late_s[feat])
                        else:
                            psi_v = psi_numeric(early_s[feat], late_s[feat], bins=10)
                        lvl = drift_level(psi_v)
                        psi_rows.append({"feature": feat, "psi": psi_v, "status": lvl})
                    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)

                    pr_early = np.nan
                    pr_late = np.nan
                    if early_s["isFraud"].nunique() > 1 and late_s["isFraud"].nunique() > 1:
                        preds_early = run_inference(
                            early_s[["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "chain_size", "is_chain_member"]],
                            preprocessor,
                            base_model,
                            cal_model,
                            metadata,
                            chain_mode="provided",
                            include_local_shap=False,
                        )
                        preds_late = run_inference(
                            late_s[["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "chain_size", "is_chain_member"]],
                            preprocessor,
                            base_model,
                            cal_model,
                            metadata,
                            chain_mode="provided",
                            include_local_shap=False,
                        )
                        pr_early = float(average_precision_score(early_s["isFraud"], preds_early["calibrated_probability"]))
                        pr_late = float(average_precision_score(late_s["isFraud"], preds_late["calibrated_probability"]))

                    st.session_state["drift_cache_params"] = current_params
                    st.session_state["drift_cache_payload"] = {
                        "psi_df": psi_df,
                        "pr_early": pr_early,
                        "pr_late": pr_late,
                        "rows_early_sampled": int(len(early_s)),
                        "rows_late_sampled": int(len(late_s)),
                        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "seed": int(st.session_state["drift_seed"]),
                        "sample_n_requested": int(sample_n),
                        "simulate_drift": bool(simulate_drift),
                    }

            if simulate_drift:
                st.warning("Demo mode active: synthetic drift injected into late window.")

            payload = st.session_state["drift_cache_payload"]
            psi_df = payload["psi_df"]
            pr_early = payload["pr_early"]
            pr_late = payload["pr_late"]
            def _fmt_int(v):
                try:
                    return f"{int(v):,}"
                except Exception:
                    return "n/a"
            st.caption(
                "Last computed: "
                f"{payload.get('computed_at', 'n/a')} | "
                f"sample early={_fmt_int(payload.get('rows_early_sampled'))} | "
                f"sample late={_fmt_int(payload.get('rows_late_sampled'))} | "
                f"requested rows/window={_fmt_int(payload.get('sample_n_requested'))} | "
                f"seed={payload.get('seed', 'n/a')} | "
                f"simulate={payload.get('simulate_drift', False)}"
            )

            def _status_icon(s: str) -> str:
                return {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(s, "⚪")

            n_green = int((psi_df["status"] == "GREEN").sum())
            n_yellow = int((psi_df["status"] == "YELLOW").sum())
            n_red = int((psi_df["status"] == "RED").sum())
            if n_red > 0:
                psi_overall = "RED"
            elif n_yellow > 0:
                psi_overall = "YELLOW"
            else:
                psi_overall = "GREEN"
            max_psi = float(psi_df["psi"].max())
            top_psi_row = psi_df.sort_values("psi", ascending=False).iloc[0]
            st.markdown("**PSI drift — labels + numbers**")
            st.markdown(
                "| Label | What it means (this app) |\n"
                "|-------|--------------------------|\n"
                "| 🟢 **GREEN** | PSI **&lt; 0.10** — stable |\n"
                "| 🟡 **YELLOW** | **0.10** ≤ PSI ≤ **0.20** — watch |\n"
                "| 🔴 **RED** | PSI **&gt; 0.20** — strong shift |\n"
            )
            st.caption(
                "**PSI rollup (pessimistic / max-severity):** assign each monitored covariate a PSI and GREEN/YELLOW/RED band; "
                "**overall PSI status** = max-severity class across covariates (any 🔴 RED ⇒ rollup RED)."
            )
            o1, o2, o3, o4, o5 = st.columns(5)
            o1.metric("Overall", f"{_status_icon(psi_overall)} {psi_overall}")
            o2.metric("Peak PSI (argmax over features)", f"{max_psi:.4f}")
            o3.metric("🟢 GREEN count", n_green)
            o4.metric("🟡 YELLOW count", n_yellow)
            o5.metric("🔴 RED count", n_red)
            st.caption(
                f"**PSI hotspot (`argmax` PSI vs early/late cohorts):** `{top_psi_row['feature']}` → "
                f"**{float(top_psi_row['psi']):.4f}** ({top_psi_row['status']}). "
                "Covariate-level PSI breakdown is in the table below."
            )

            psi_show = psi_df.copy()
            psi_show["psi"] = psi_show["psi"].map(lambda x: f"{x:.4f}")
            psi_show["status"] = psi_show["status"].map(lambda s: f"{_status_icon(s)} {s}")
            st.markdown("**PSI score table (feature drift):**")
            st.dataframe(psi_show, use_container_width=True, hide_index=True)
            if (psi_df["psi"] > 0.2).any():
                st.error("Significant feature drift detected (PSI > 0.20) in at least one feature.")
            elif (psi_df["psi"] >= 0.1).any():
                st.warning("Mild feature drift detected (0.10 - 0.20). Monitor closely.")
            else:
                st.success("No meaningful feature drift detected (all PSI < 0.10).")

            bar_df = psi_df.copy()
            bar_df["color"] = bar_df["status"].map({"GREEN": "#2ca02c", "YELLOW": "#ffbf00", "RED": "#d62728"})
            st.markdown("**Feature Drift Summary (PSI Scores):**")
            st.vega_lite_chart(
                bar_df,
                {
                    "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
                    "encoding": {
                        "x": {
                            "field": "feature",
                            "type": "nominal",
                            "title": "Feature",
                            "axis": {"labelAngle": -35, "labelFontSize": 12, "titleFontSize": 13},
                        },
                        "y": {"field": "psi", "type": "quantitative", "title": "PSI score"},
                        "color": {"field": "color", "type": "nominal", "scale": None, "legend": None},
                        "tooltip": [{"field": "feature"}, {"field": "psi"}, {"field": "status"}],
                    },
                    "height": 280,
                },
                use_container_width=True,
            )

            st.markdown("**Model performance drift (PR-AUC):**")
            if np.isnan(pr_early) or np.isnan(pr_late):
                st.warning("PR-AUC comparison unavailable (one window has a single class in this sample).")
            else:
                delta = pr_early - pr_late
                m1, m2, m3 = st.columns(3)
                m1.metric("PR-AUC (early)", f"{pr_early:.4f}")
                m2.metric("PR-AUC (late)", f"{pr_late:.4f}")
                m3.metric("Delta (early - late)", f"{delta:.4f}")
                if abs(delta) > 0.05:
                    st.error("Retraining Recommended ⚠️  Drift detected from PR-AUC gap > 0.05.")
                else:
                    st.success("Model Stable ✅  PR-AUC gap is within tolerance.")

    with model_card_tab:
        st.subheader("Model Card")
        mtime = MODEL_CARD_PATH.stat().st_mtime if MODEL_CARD_PATH.exists() else 0.0
        model_card_text = load_model_card_text(mtime)
        if model_card_text:
            st.markdown(
                apply_live_metadata_to_model_card(model_card_text, metadata),
                unsafe_allow_html=False,
            )
        else:
            st.info(
                "MODEL_CARD.md is missing. Add it to the project root to show the Model Card tab."
            )


if __name__ == "__main__":
    main()
