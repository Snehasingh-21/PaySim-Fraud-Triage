import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2

# Match notebook's final RF setup.
RF_N_ESTIMATORS = 40
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 5

# Dynamic RF-family calibration selection.
CALIBRATION_CV = 2
CALIBRATION_METHODS = ("sigmoid", "isotonic")
BRIER_CLOSE_TOL = 1e-7

# Match final triage rule used in notebook.
OPERATING_THRESHOLD = 0.50
REVIEW_THRESHOLD = 0.40
BLOCK_THRESHOLD = 0.60
MODERATE_CUTOFF = REVIEW_THRESHOLD
CHAIN_SIZE_CAP = 12


def add_row_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row engineered features (no chain / no cross-row stats)."""
    out = df.copy()
    out["orig_delta"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["dest_delta"] = out["newbalanceDest"] - out["oldbalanceDest"]
    out["orig_residual"] = out["orig_delta"] - out["amount"]
    out["orig_zero_old"] = (out["oldbalanceOrg"] == 0).astype(np.int8)
    out["dest_zero_old"] = (out["oldbalanceDest"] == 0).astype(np.int8)
    out["orig_zero_new"] = (out["newbalanceOrig"] == 0).astype(np.int8)
    out["dest_zero_new"] = (out["newbalanceDest"] == 0).astype(np.int8)
    out["log_amount"] = np.log1p(out["amount"].astype(np.float64))
    return out


def add_chain_features(
    frame: pd.DataFrame,
    chain_size_cap: Optional[int] = CHAIN_SIZE_CAP,
) -> pd.DataFrame:
    """Chain features from step/amount/type only; computed within given frame (split-safe)."""
    out = frame.copy()
    chain_group = (
        out.groupby(["step", "amount"], as_index=False)
        .agg(
            chain_size=("type", "size"),
            has_transfer=("type", lambda s: (s == "TRANSFER").any()),
            has_cash_out=("type", lambda s: (s == "CASH_OUT").any()),
        )
    )
    chain_group["is_chain_member"] = chain_group["has_transfer"] & chain_group["has_cash_out"]
    if chain_size_cap is not None:
        chain_group["is_chain_member"] = chain_group["is_chain_member"] & (
            chain_group["chain_size"] <= chain_size_cap
        )
    chain_group["is_chain_member"] = chain_group["is_chain_member"].astype(np.int8)

    out = out.merge(
        chain_group[["step", "amount", "chain_size", "is_chain_member"]],
        on=["step", "amount"],
        how="left",
    )
    out["chain_size"] = out["chain_size"].astype(np.int32)
    out["is_chain_member"] = out["is_chain_member"].astype(np.int8)
    return out


def select_final_rf_calibrated_model(
    calibration_table: pd.DataFrame,
    brier_close_tol: float = BRIER_CLOSE_TOL,
) -> tuple[str, str]:
    """
    Pick final RF-family probability source.
    Rule: lowest Brier, then higher PR-AUC, then higher ROC-AUC.
    """
    rf_candidates = ["rf_plain_uncalibrated", "rf_plain_sigmoid", "rf_plain_isotonic"]
    rf_tbl = calibration_table[calibration_table["model"].isin(rf_candidates)].copy()
    if rf_tbl.empty:
        raise ValueError("No RF calibration rows found for final model selection.")

    min_brier = float(rf_tbl["brier"].min())
    close_tbl = rf_tbl[rf_tbl["brier"] <= min_brier + float(brier_close_tol)].copy()
    close_tbl = close_tbl.sort_values(["pr_auc", "roc_auc"], ascending=[False, False]).reset_index(drop=True)

    final_row = close_tbl.iloc[0]
    final_key = str(final_row["model"])
    final_reason = (
        f"Selected {final_key} dynamically within RF family using lowest Brier "
        f"(tol={brier_close_tol:g}), then highest PR-AUC, then highest ROC-AUC. "
        f"Values => Brier={final_row['brier']:.8f}, PR-AUC={final_row['pr_auc']:.6f}, "
        f"ROC-AUC={final_row['roc_auc']:.6f}."
    )
    return final_key, final_reason


def _resolve_data_path(root: Path) -> Path:
    """Match app.py: PAYSIM_CSV / PAYSIM_DATA_PATH / PAYSIM_CSV_NAME or default filename."""
    for key in ("PAYSIM_CSV", "PAYSIM_DATA_PATH"):
        raw = os.getenv(key)
        if raw:
            p = Path(raw).expanduser()
            return p.resolve() if p.is_absolute() else (root / p).resolve()
    name = os.getenv("PAYSIM_CSV_NAME", "PS_20174392719_1491204439457_log.csv")
    return (root / name).resolve()


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = _resolve_data_path(root)
    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    target = "isFraud"
    drop_from_x = ["nameOrig", "nameDest", "isFlaggedFraud"]

    df_model = add_row_engineered_features(df)
    y = df_model[target].astype(np.int8)
    X_base = df_model.drop(columns=[target] + drop_from_x)

    X_train, X_test, y_train, y_test = train_test_split(
        X_base, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train = add_chain_features(X_train, chain_size_cap=CHAIN_SIZE_CAP)
    X_test = add_chain_features(X_test, chain_size_cap=CHAIN_SIZE_CAP)
    X_train = X_train.drop(columns=["amount"])
    X_test = X_test.drop(columns=["amount"])

    cat_features = ["type"]
    num_features = [c for c in X_train.columns if c not in cat_features]
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    X_train_proc = np.asarray(X_train_proc, dtype=np.float64)
    X_test_proc = np.asarray(X_test_proc, dtype=np.float64)
    X_train_proc = np.nan_to_num(np.clip(X_train_proc, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    X_test_proc = np.nan_to_num(np.clip(X_test_proc, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

    rf_plain = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
    )
    rf_plain.fit(X_train_proc, y_train)

    # Build RF-family probability sources: uncalibrated + sigmoid + isotonic
    probs_by_model: dict[str, np.ndarray] = {
        "rf_plain_uncalibrated": rf_plain.predict_proba(X_test_proc)[:, 1],
    }
    calibrators: dict[str, CalibratedClassifierCV] = {}
    for method in CALIBRATION_METHODS:
        calibrator = CalibratedClassifierCV(
            estimator=rf_plain,
            method=method,
            cv=CALIBRATION_CV,
            n_jobs=1,
            ensemble=False,
        )
        calibrator.fit(X_train_proc, y_train)
        key = f"rf_plain_{method}"
        calibrators[key] = calibrator
        probs_by_model[key] = calibrator.predict_proba(X_test_proc)[:, 1]

    # Compare RF-family calibration variants on same untouched test split.
    y_test_arr = np.asarray(y_test).astype(int)
    rows = []
    for key in ["rf_plain_uncalibrated", "rf_plain_sigmoid", "rf_plain_isotonic"]:
        if key not in probs_by_model:
            continue
        p = np.asarray(probs_by_model[key], dtype=np.float64)
        rows.append(
            {
                "model": key,
                "brier": float(brier_score_loss(y_test_arr, p)),
                "pr_auc": float(average_precision_score(y_test_arr, p)),
                "roc_auc": float(roc_auc_score(y_test_arr, p)),
            }
        )
    calibration_comparison_table = pd.DataFrame(rows).sort_values(
        ["brier", "pr_auc", "roc_auc"], ascending=[True, False, False]
    ).reset_index(drop=True)

    final_model_key, final_model_reason = select_final_rf_calibrated_model(calibration_comparison_table)
    if final_model_key == "rf_plain_uncalibrated":
        final_cal_model = rf_plain
    else:
        final_cal_model = calibrators[final_model_key]

    joblib.dump(preprocessor, out_dir / "preprocessor_paysim.joblib")
    joblib.dump(rf_plain, out_dir / "rf_plain_base.joblib")
    # Save individual calibrated RF variants and selected final calibrated model.
    if "rf_plain_sigmoid" in calibrators:
        joblib.dump(calibrators["rf_plain_sigmoid"], out_dir / "rf_plain_sigmoid_calibrated.joblib")
    if "rf_plain_isotonic" in calibrators:
        joblib.dump(calibrators["rf_plain_isotonic"], out_dir / "rf_plain_isotonic_calibrated.joblib")
    joblib.dump(final_cal_model, out_dir / "rf_selected_calibrated.joblib")

    metadata = {
        "input_feature_columns": list(X_train.columns),
        "processed_feature_names": list(preprocessor.get_feature_names_out()),
        "triage_thresholds": {
            "operating_threshold": OPERATING_THRESHOLD,
            "review_threshold": REVIEW_THRESHOLD,
            "block_threshold": BLOCK_THRESHOLD,
            "moderate_cutoff": MODERATE_CUTOFF,
        },
        "chain_size_cap": CHAIN_SIZE_CAP,
        "final_model_key": final_model_key,
        "final_model_reason": final_model_reason,
        "calibration_selection_rule": "lowest_brier_then_higher_pr_auc_then_higher_roc_auc_within_rf_family",
        "calibration_comparison_table": calibration_comparison_table.to_dict(orient="records"),
        "calibration_model_file_map": {
            "rf_plain_uncalibrated": "rf_plain_base.joblib",
            "rf_plain_sigmoid": "rf_plain_sigmoid_calibrated.joblib",
            "rf_plain_isotonic": "rf_plain_isotonic_calibrated.joblib",
        },
        "notes": (
            "Artifacts: RF base + dynamic RF-family calibrated deployment path; "
            "chain features computed per split (train/test) to match notebook leakage-safe evaluation."
        ),
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Saved artifacts:")
    print(f"- {out_dir / 'preprocessor_paysim.joblib'}")
    print(f"- {out_dir / 'rf_plain_base.joblib'}")
    if (out_dir / "rf_plain_sigmoid_calibrated.joblib").exists():
        print(f"- {out_dir / 'rf_plain_sigmoid_calibrated.joblib'}")
    if (out_dir / "rf_plain_isotonic_calibrated.joblib").exists():
        print(f"- {out_dir / 'rf_plain_isotonic_calibrated.joblib'}")
    print(f"- {out_dir / 'rf_selected_calibrated.joblib'}")
    print(f"Selected final model key: {final_model_key}")
    print(f"Selection reason: {final_model_reason}")
    print(f"- {out_dir / 'feature_metadata.json'}")


if __name__ == "__main__":
    main()
