"""
XGBoost SMB health score model.

Loads mart_company_health from DuckDB, trains an XGBoost pipeline and a
Logistic Regression baseline, evaluates both, saves the XGBoost pipeline
and label encoders, and generates a feature importance chart.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import json

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent
DB_PATH    = ROOT / "data" / "smb_health.duckdb"
MODEL_DIR  = Path(__file__).resolve().parent
MODEL_PATH    = MODEL_DIR / "smb_health_model.pkl"
ENC_PATH      = MODEL_DIR / "label_encoders.pkl"
CHART_PATH    = MODEL_DIR / "feature_importance.png"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ── feature definitions ───────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "txn_failure_rate",
    "missed_payment_rate",
    "late_payment_rate",
    "avg_days_late",
    "max_days_late",
    "avg_retry_count",
    "spend_trend",
    "credit_utilization",
    "total_spend",
    "avg_txn_amount",
    "unique_merchants",
    "unique_mcc_categories",
    "spend_last_30d",
    "spend_last_90d",
    "spend_last_180d",
]

CATEGORICAL_FEATURES = [
    "industry",
    "company_size",
    "region",
    "tenure_bucket",
    "dominant_payment_method",
    "most_common_failure_reason",
]

TARGET      = "health_segment_encoded"
CLASS_NAMES = ["healthy", "watch", "at_risk"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


def _build_preprocessor() -> ColumnTransformer:
    """Creates a fresh, unfitted ColumnTransformer."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,  NUMERIC_FEATURES),
            ("cat", cat_enc,       CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ── 1. data loading ───────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    _section("1. DATA LOADING")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute("SELECT * FROM mart_company_health").df()
    con.close()

    print(f"  mart_company_health  →  {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"\n  Class balance (health_segment):")
    for seg, cnt in df["health_segment"].value_counts().items():
        bar = "▓" * cnt
        print(f"    {seg:<12}  {cnt:>3}  ({cnt/len(df)*100:.0f}%)  {bar}")
    return df


# ── 2. train/test split ───────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    _section("2. TRAIN / TEST SPLIT  (80/20, stratified)")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    for label, ys in [("Train", y_train), ("Test", y_test)]:
        vc = ys.value_counts().sort_index()
        dist = "  ".join(f"{CLASS_NAMES[i]}={vc.get(i, 0)}" for i in range(3))
        print(f"  {label} ({len(ys)} samples)   {dist}")

    return X_train, X_test, y_train, y_test


# ── 3. model training ─────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train) -> Pipeline:
    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_logreg(X_train, y_train) -> Pipeline:
    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ── 4/5. evaluation ───────────────────────────────────────────────────────────
def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    _section(f"EVALUATION — {name.upper()}")

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    cm  = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    ))
    print(f"  Accuracy :  {acc:.4f}")
    print(f"  Macro F1 :  {f1:.4f}")
    print(f"  AUC-ROC  :  {auc:.4f}  (macro, one-vs-rest)")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    col_h = "  ".join(f"{n:>11}" for n in CLASS_NAMES)
    print(f"  {'':12}  {col_h}")
    for i, row in enumerate(cm):
        row_s = "  ".join(f"{v:>11}" for v in row)
        print(f"  {CLASS_NAMES[i]:<12}  {row_s}")

    return {"model": name, "accuracy": acc, "macro_f1": f1, "auc_roc": auc}


# ── 6. feature importance ─────────────────────────────────────────────────────
def feature_importance(xgb_pipeline: Pipeline) -> None:
    _section("6. FEATURE IMPORTANCE  (weight = split count)")

    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    xgb_clf      = xgb_pipeline.named_steps["classifier"]

    # Resolve clean feature names from ColumnTransformer output
    raw_names   = preprocessor.get_feature_names_out()
    clean_names = [n.split("__", 1)[1] for n in raw_names]

    # Attach names so booster.get_score() uses them (not f0, f1…)
    booster = xgb_clf.get_booster()
    booster.feature_names = clean_names

    importance_raw  = booster.get_score(importance_type="weight")
    importance_full = {f: importance_raw.get(f, 0) for f in clean_names}

    imp_df = (
        pd.DataFrame.from_dict(importance_full, orient="index", columns=["importance"])
        .sort_values("importance", ascending=False)
    )

    max_val = imp_df["importance"].max()
    for feat, row in imp_df.iterrows():
        bar_len = int(round(row["importance"] / max_val * 30)) if max_val else 0
        bar     = "█" * bar_len
        marker  = " ◀ top" if feat == imp_df.index[0] else ""
        print(f"  {feat:<35}  {row['importance']:>5.0f}  {bar}{marker}")

    # ── horizontal bar chart ─────────────────────────────────────────────────
    plot_df = imp_df[imp_df["importance"] > 0].sort_values("importance")
    n       = len(plot_df)
    colors  = ["#1565C0" if i >= n - 5 else "#64B5F6" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.38)))
    ax.barh(plot_df.index, plot_df["importance"], color=colors,
            edgecolor="white", linewidth=0.5, height=0.72)

    ax.set_xlabel("Feature Importance  (Weight — number of splits)", fontsize=11)
    ax.set_title("SMB Health Score — Feature Importance",
                 fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--", color="#888")
    ax.tick_params(axis="y", labelsize=9)

    # Value annotations
    for bar, val in zip(ax.patches, plot_df["importance"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=8, color="#333")

    plt.tight_layout()
    fig.savefig(str(CHART_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart saved → {CHART_PATH}")


# ── 7. comparison table ───────────────────────────────────────────────────────
def comparison_table(results: list) -> None:
    _section("7. MODEL COMPARISON")
    print(f"  {'Model':<26}  {'Accuracy':>9}  {'Macro F1':>9}  {'AUC-ROC':>9}")
    print("  " + "─" * 58)
    for r in results:
        print(f"  {r['model']:<26}  {r['accuracy']:>9.4f}  {r['macro_f1']:>9.4f}  {r['auc_roc']:>9.4f}")


# ── 8. serialize ─────────────────────────────────────────────────────────────
def serialize(xgb_pipeline: Pipeline, xgb_results: dict, test_size: int) -> None:
    _section("8. SERIALIZATION")
    joblib.dump(xgb_pipeline, MODEL_PATH)

    enc = xgb_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    joblib.dump(
        {"ordinal_encoder": enc, "categorical_features": CATEGORICAL_FEATURES},
        ENC_PATH,
    )

    metadata = {
        "model":          "XGBoost",
        "test_set_size":  test_size,
        "accuracy":       round(xgb_results["accuracy"],  4),
        "macro_f1":       round(xgb_results["macro_f1"],  4),
        "auc_roc":        round(xgb_results["auc_roc"],   4),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"  Pipeline saved  → {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  Encoders saved  → {ENC_PATH}")
    print(f"  Metadata saved  → {METADATA_PATH}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    _section("3. TRAINING")
    print("  Training XGBoost (n_estimators=200, max_depth=4)...")
    xgb_pipeline = train_xgboost(X_train, y_train)
    print("  Training Logistic Regression baseline (max_iter=1000)...")
    lr_pipeline  = train_logreg(X_train, y_train)
    print("  Done.")

    results = [
        evaluate("XGBoost",             xgb_pipeline, X_test, y_test),
        evaluate("Logistic Regression", lr_pipeline,  X_test, y_test),
    ]

    feature_importance(xgb_pipeline)
    comparison_table(results)
    serialize(xgb_pipeline, xgb_results=results[0], test_size=len(y_test))
    print()


if __name__ == "__main__":
    main()
