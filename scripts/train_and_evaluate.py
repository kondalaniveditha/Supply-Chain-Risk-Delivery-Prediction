"""
scripts/train_and_evaluate.py
==============================
End-to-end training, evaluation, and artefact generation script.
Run this from the project root:
    venv/Scripts/python scripts/train_and_evaluate.py
"""

import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import run_full_pipeline, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.feature_engineering import add_all_features, get_final_feature_columns
from src.regression_models import train_regression_models, save_regression_artifacts
from src.classification_models import train_classification_models, save_classification_artifacts
from src.evaluation import (
    compare_regression_models,
    compare_classification_models,
    select_best_regression_model,
    select_best_classification_model,
    get_confusion_matrix,
)
from src.explainability import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    get_top_shap_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
MODELS_DIR  = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# 1. PREPROCESSING
# ============================================================
logger.info("=" * 60)
logger.info("STEP 1: PREPROCESSING")
logger.info("=" * 60)

pipeline_result = run_full_pipeline(
    raw_path="data/raw/DataCoSupplyChainDataset.csv",
    processed_path="data/processed/supply_chain_processed.csv",
)

df_full      = pipeline_result["df_full"]
df_full_enc  = pipeline_result["df_full_enc"]
df_train     = pipeline_result["df_train"]
df_test      = pipeline_result["df_test"]
encoders     = pipeline_result["encoders"]
quality_rep  = pipeline_result["quality_report"]

# ============================================================
# 2. FEATURE ENGINEERING (applied after preprocessing)
# ============================================================
logger.info("=" * 60)
logger.info("STEP 2: FEATURE ENGINEERING")
logger.info("=" * 60)

# We apply FE to the un-encoded df, then re-encode/rebuild matrices
# For simplicity in this script we work with the already-processed splits
# and add engineered features directly

def prepare_splits(df_train, df_test):
    """Add engineered features and build X/y splits."""
    from src.data_preprocessing import build_feature_matrix

    df_tr = add_all_features(df_train)
    df_te = add_all_features(df_test)

    X_tr, y_reg_tr, y_clf_tr = build_feature_matrix(df_tr)
    X_te, y_reg_te, y_clf_te = build_feature_matrix(df_te)

    # Keep only numeric columns for modelling
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    X_tr = X_tr[num_cols]
    X_te = X_te[num_cols]

    # Align test to train columns
    X_te = X_te.reindex(columns=num_cols, fill_value=0)

    return X_tr, X_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te, num_cols

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, feature_cols = \
    prepare_splits(df_train, df_test)

logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:8])
logger.info("X_train shape: %s", X_train.shape)
logger.info("X_test shape:  %s", X_test.shape)
logger.info("Late rate train: %.2f%%", y_clf_train.mean() * 100)
logger.info("Late rate test:  %.2f%%", y_clf_test.mean() * 100)

# ============================================================
# 3. EDA VISUALIZATIONS
# ============================================================
logger.info("=" * 60)
logger.info("STEP 3: EDA VISUALIZATIONS")
logger.info("=" * 60)

sns.set_theme(style="whitegrid", palette="muted")

# --- 3a. Delivery time distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
df_full["Days for shipping (real)"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color="#3b82d4", edgecolor="white"
)
ax.set_title("Distribution of Actual Delivery Days", fontsize=14, fontweight="bold")
ax.set_xlabel("Days for Shipping (Actual)", fontsize=12)
ax.set_ylabel("Number of Shipments", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/delivery_time_distribution.png", dpi=150)
plt.close()
logger.info("Saved: delivery_time_distribution.png")

# --- 3b. Late delivery distribution ---
fig, ax = plt.subplots(figsize=(6, 5))
late_counts = df_full["Late_delivery_risk"].value_counts().sort_index()
labels = ["On Time / Advance", "Late"]
colors = ["#22c55e", "#ef4444"]
wedges, texts, autotexts = ax.pie(
    late_counts.values, labels=labels, autopct="%1.1f%%",
    colors=colors, startangle=90, textprops={"fontsize": 12}
)
ax.set_title("Late vs On-Time Delivery Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/late_delivery_distribution.png", dpi=150)
plt.close()
logger.info("Saved: late_delivery_distribution.png")

# --- 3c. Delivery performance by shipping mode ---
ship_perf = df_full.groupby("Shipping Mode").agg(
    avg_days=("Days for shipping (real)", "mean"),
    late_rate=("Late_delivery_risk", "mean"),
).reset_index().sort_values("avg_days")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(ship_perf["Shipping Mode"], ship_perf["avg_days"], color="#3b82d4")
axes[0].set_xlabel("Average Delivery Days", fontsize=11)
axes[0].set_title("Avg Delivery Days by Shipping Mode", fontsize=12, fontweight="bold")
axes[0].axvline(df_full["Days for shipping (real)"].mean(), color="red",
                linestyle="--", label=f"Overall avg: {df_full['Days for shipping (real)'].mean():.1f}d")
axes[0].legend(fontsize=9)

axes[1].barh(ship_perf["Shipping Mode"], ship_perf["late_rate"] * 100, color="#f59e0b")
axes[1].set_xlabel("Late Delivery Rate (%)", fontsize=11)
axes[1].set_title("Late Rate by Shipping Mode", fontsize=12, fontweight="bold")
for i, v in enumerate(ship_perf["late_rate"] * 100):
    axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/performance_by_shipping_mode.png", dpi=150)
plt.close()
logger.info("Saved: performance_by_shipping_mode.png")

# --- 3d. Delivery performance by market/region ---
region_perf = df_full.groupby("Market").agg(
    avg_days=("Days for shipping (real)", "mean"),
    late_rate=("Late_delivery_risk", "mean"),
    count=("Days for shipping (real)", "count"),
).reset_index().sort_values("late_rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors_bar = ["#ef4444" if r > region_perf["late_rate"].mean() else "#3b82d4"
              for r in region_perf["late_rate"]]
axes[0].barh(region_perf["Market"], region_perf["late_rate"] * 100, color=colors_bar)
axes[0].set_xlabel("Late Delivery Rate (%)", fontsize=11)
axes[0].set_title("Late Rate by Market", fontsize=12, fontweight="bold")
for i, v in enumerate(region_perf["late_rate"] * 100):
    axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)

axes[1].barh(region_perf["Market"], region_perf["avg_days"], color="#7c5cd8")
axes[1].set_xlabel("Average Delivery Days", fontsize=11)
axes[1].set_title("Avg Delivery Days by Market", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/performance_by_region.png", dpi=150)
plt.close()
logger.info("Saved: performance_by_region.png")

# --- 3e. Delivery performance by top categories ---
cat_perf = df_full.groupby("Category Name").agg(
    avg_days=("Days for shipping (real)", "mean"),
    late_rate=("Late_delivery_risk", "mean"),
    count=("Days for shipping (real)", "count"),
).reset_index().sort_values("count", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(cat_perf.sort_values("late_rate")["Category Name"],
        cat_perf.sort_values("late_rate")["late_rate"] * 100,
        color="#f59e0b")
ax.set_xlabel("Late Delivery Rate (%)", fontsize=11)
ax.set_title("Late Delivery Rate – Top 15 Product Categories", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/performance_by_category.png", dpi=150)
plt.close()
logger.info("Saved: performance_by_category.png")

# --- 3f. Delivery trend over time ---
df_full["order_date"] = pd.to_datetime(
    pipeline_result["df_full"]["order_date"], errors="coerce"
)
monthly = df_full.set_index("order_date").resample("ME").agg(
    total=("Days for shipping (real)", "count"),
    avg_days=("Days for shipping (real)", "mean"),
    late_rate=("Late_delivery_risk", "mean"),
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(monthly.index, monthly["avg_days"], color="#3b82d4", linewidth=2)
axes[0].set_ylabel("Avg Delivery Days", fontsize=11)
axes[0].set_title("Monthly Delivery Performance Trend", fontsize=13, fontweight="bold")
axes[0].grid(linestyle="--", alpha=0.4)

axes[1].plot(monthly.index, monthly["late_rate"] * 100, color="#ef4444", linewidth=2)
axes[1].set_ylabel("Late Rate (%)", fontsize=11)
axes[1].set_xlabel("Order Month", fontsize=11)
axes[1].grid(linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/delivery_trend_over_time.png", dpi=150)
plt.close()
logger.info("Saved: delivery_trend_over_time.png")

# --- 3g. Correlation heatmap of numerical features ---
num_df = df_full[NUMERICAL_FEATURES + ["Days for shipping (real)", "Late_delivery_risk"]].copy()
corr = num_df.corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, annot_kws={"size": 8}, linewidths=0.3)
ax.set_title("Pearson Correlation Heatmap – Numerical Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
logger.info("Saved: correlation_heatmap.png")

# --- 3h. Scheduled vs actual days scatter ---
fig, ax = plt.subplots(figsize=(7, 5))
sample = df_full.sample(min(5000, len(df_full)), random_state=42)
scatter_colors = sample["Late_delivery_risk"].map({0: "#22c55e", 1: "#ef4444"})
ax.scatter(sample["Days for shipment (scheduled)"],
           sample["Days for shipping (real)"],
           c=scatter_colors, alpha=0.3, s=15)
ax.set_xlabel("Days Scheduled", fontsize=11)
ax.set_ylabel("Days Actual", fontsize=11)
ax.set_title("Scheduled vs Actual Delivery Days\n(red=late, green=on-time)",
             fontsize=12, fontweight="bold")
from matplotlib.patches import Patch
legend_elements = [Patch(fc="#22c55e", label="On Time"),
                   Patch(fc="#ef4444", label="Late")]
ax.legend(handles=legend_elements, fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/scheduled_vs_actual_days.png", dpi=150)
plt.close()
logger.info("Saved: scheduled_vs_actual_days.png")

# ============================================================
# 4. REGRESSION MODELLING
# ============================================================
logger.info("=" * 60)
logger.info("STEP 4: REGRESSION MODELLING")
logger.info("=" * 60)

reg_models, reg_scaler = train_regression_models(X_train, y_reg_train)
X_test_reg_scaled = reg_scaler.transform(X_test)

reg_comparison = compare_regression_models(reg_models, X_test_reg_scaled, y_reg_test)
logger.info("Regression comparison:\n%s", reg_comparison.to_string(index=False))

best_reg_name, best_reg_model = select_best_regression_model(reg_comparison, reg_models)
logger.info("Best regression model: %s", best_reg_name)

save_regression_artifacts(best_reg_model, reg_scaler, feature_cols)

# Save comparison table
reg_comparison.to_csv("models/regression_comparison.csv", index=False)

# ============================================================
# 5. CLASSIFICATION MODELLING
# ============================================================
logger.info("=" * 60)
logger.info("STEP 5: CLASSIFICATION MODELLING")
logger.info("=" * 60)

clf_models, clf_scaler = train_classification_models(X_train, y_clf_train)
X_test_clf_scaled = clf_scaler.transform(X_test)

clf_comparison = compare_classification_models(clf_models, X_test_clf_scaled, y_clf_test)
logger.info("Classification comparison:\n%s", clf_comparison.to_string(index=False))

best_clf_name, best_clf_model = select_best_classification_model(clf_comparison, clf_models)
logger.info("Best classification model: %s", best_clf_name)

save_classification_artifacts(best_clf_model, clf_scaler, feature_cols)

# Save comparison table
clf_comparison.to_csv("models/classification_comparison.csv", index=False)

# ============================================================
# 6. MODEL COMPARISON CHARTS
# ============================================================
logger.info("=" * 60)
logger.info("STEP 6: MODEL COMPARISON CHARTS")
logger.info("=" * 60)

# Regression comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_reg = ["MAE", "RMSE", "R²"]
colors_reg = ["#3b82d4", "#7c5cd8", "#22c55e"]
for i, metric in enumerate(metrics_reg):
    col = "R²" if metric == "R²" else metric
    axes[i].barh(reg_comparison["Model"], reg_comparison[col], color=colors_reg[i])
    axes[i].set_title(f"{metric}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel(metric, fontsize=10)
    if metric == "R²":
        axes[i].set_xlim(0, 1)
plt.suptitle("Regression Model Comparison", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/regression_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved: regression_model_comparison.png")

# Classification comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_clf = ["Accuracy", "F1", "ROC-AUC"]
colors_clf = ["#3b82d4", "#f59e0b", "#22c55e"]
for i, metric in enumerate(metrics_clf):
    if metric in clf_comparison.columns:
        axes[i].barh(clf_comparison["Model"], clf_comparison[metric], color=colors_clf[i])
        axes[i].set_title(metric, fontsize=12, fontweight="bold")
        axes[i].set_xlim(0, 1)
        for j, v in enumerate(clf_comparison[metric]):
            axes[i].text(v + 0.005, j, f"{v:.3f}", va="center", fontsize=9)
plt.suptitle("Classification Model Comparison", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/classification_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved: classification_model_comparison.png")

# Confusion matrix for best classifier
from sklearn.metrics import ConfusionMatrixDisplay
y_pred_best_clf = best_clf_model.predict(X_test_clf_scaled)
cm = get_confusion_matrix(y_clf_test, y_pred_best_clf)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On Time", "Late"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix – {best_clf_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/confusion_matrix.png", dpi=150)
plt.close()
logger.info("Saved: confusion_matrix.png")

# ROC Curve
from sklearn.metrics import roc_curve, auc as sk_auc
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in clf_models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_clf_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_clf_test, y_prob)
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – Classification Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/roc_curves.png", dpi=150)
plt.close()
logger.info("Saved: roc_curves.png")

# ============================================================
# 7. FEATURE IMPORTANCE (tree-based best model)
# ============================================================
logger.info("=" * 60)
logger.info("STEP 7: FEATURE IMPORTANCE")
logger.info("=" * 60)

def plot_feature_importance(model, feature_names, title, save_path, top_n=15):
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model %s has no feature_importances_ attribute.", type(model).__name__)
        return
    imp = model.feature_importances_
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df_imp["feature"], df_imp["importance"], color="#3b82d4")
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", save_path)

plot_feature_importance(
    best_reg_model, feature_cols,
    f"Feature Importance – {best_reg_name} (Regression)",
    f"{FIGURES_DIR}/regression_feature_importance.png"
)
plot_feature_importance(
    best_clf_model, feature_cols,
    f"Feature Importance – {best_clf_name} (Classification)",
    f"{FIGURES_DIR}/classification_feature_importance.png"
)

# ============================================================
# 8. SHAP EXPLANATIONS
# ============================================================
logger.info("=" * 60)
logger.info("STEP 8: SHAP EXPLANATIONS")
logger.info("=" * 60)

X_train_scaled = reg_scaler.transform(X_train)
X_test_scaled  = reg_scaler.transform(X_test)

# SHAP for best regression model
try:
    reg_shap_vals, reg_X_sample = compute_shap_values(
        best_reg_model, X_test_scaled, X_train_scaled, sample_size=1500
    )
    plot_shap_summary(
        reg_shap_vals, reg_X_sample, feature_cols,
        title=f"SHAP Summary – {best_reg_name} (Delivery Days)",
        save_path=f"{FIGURES_DIR}/shap_regression_summary.png"
    )
    reg_shap_imp = plot_shap_bar(
        reg_shap_vals, feature_cols,
        title=f"SHAP Global Importance – {best_reg_name} (Regression)",
        save_path=f"{FIGURES_DIR}/shap_regression_bar.png"
    )
    logger.info("Regression SHAP top features:\n%s", reg_shap_imp.head(10).to_string(index=False))
except Exception as e:
    logger.warning("Regression SHAP failed: %s", e)
    reg_shap_imp = pd.DataFrame({"feature": feature_cols[:5], "importance": [0]*5})

# SHAP for best classification model
X_train_clf_scaled = clf_scaler.transform(X_train)
X_test_clf_scaled2 = clf_scaler.transform(X_test)
try:
    clf_shap_vals, clf_X_sample = compute_shap_values(
        best_clf_model, X_test_clf_scaled2, X_train_clf_scaled, sample_size=1500
    )
    plot_shap_summary(
        clf_shap_vals, clf_X_sample, feature_cols,
        title=f"SHAP Summary – {best_clf_name} (Late Delivery Risk)",
        save_path=f"{FIGURES_DIR}/shap_classification_summary.png"
    )
    clf_shap_imp = plot_shap_bar(
        clf_shap_vals, feature_cols,
        title=f"SHAP Global Importance – {best_clf_name} (Classification)",
        save_path=f"{FIGURES_DIR}/shap_classification_bar.png"
    )
    logger.info("Classification SHAP top features:\n%s", clf_shap_imp.head(10).to_string(index=False))
except Exception as e:
    logger.warning("Classification SHAP failed: %s", e)
    clf_shap_imp = pd.DataFrame({"feature": feature_cols[:5], "importance": [0]*5})

# ============================================================
# 9. SAVE METRICS SUMMARY FOR NOTEBOOK / REPORT
# ============================================================
logger.info("=" * 60)
logger.info("STEP 9: SAVING METRICS SUMMARY")
logger.info("=" * 60)

# Best regression metrics on test set
best_reg_pred  = best_reg_model.predict(X_test_reg_scaled)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
best_reg_metrics = {
    "model": best_reg_name,
    "MAE":  round(mean_absolute_error(y_reg_test, best_reg_pred), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_reg_test, best_reg_pred)), 4),
    "R2":   round(r2_score(y_reg_test, best_reg_pred), 4),
}

# Best classification metrics on test set
best_clf_pred  = best_clf_model.predict(X_test_clf_scaled)
best_clf_prob  = best_clf_model.predict_proba(X_test_clf_scaled)[:, 1]
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
best_clf_metrics = {
    "model":     best_clf_name,
    "Accuracy":  round(accuracy_score(y_clf_test, best_clf_pred), 4),
    "Precision": round(precision_score(y_clf_test, best_clf_pred, zero_division=0), 4),
    "Recall":    round(recall_score(y_clf_test, best_clf_pred, zero_division=0), 4),
    "F1":        round(f1_score(y_clf_test, best_clf_pred, zero_division=0), 4),
    "ROC_AUC":   round(roc_auc_score(y_clf_test, best_clf_prob), 4),
}

summary = {
    "dataset": {
        "total_rows": len(df_full),
        "total_cols": 53,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "late_rate_pct": round(float(y_clf_train.mean() * 100), 2),
    },
    "regression": {
        "best_model": best_reg_metrics,
        "all_models": reg_comparison.to_dict("records"),
    },
    "classification": {
        "best_model": best_clf_metrics,
        "all_models": clf_comparison.to_dict("records"),
    },
}

with open("models/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
logger.info("Training summary saved: models/training_summary.json")

# ============================================================
# 10. PRINT FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 60)
print(f"\nDataset: {len(df_full):,} rows × 53 cols")
print(f"Features used: {len(feature_cols)}")
print(f"Train / Test split: {len(X_train):,} / {len(X_test):,} (chronological)")
print(f"Late delivery rate: {y_clf_train.mean()*100:.1f}% (train)")

print("\n--- REGRESSION RESULTS (Target: Days for shipping) ---")
print(reg_comparison.to_string(index=False))
print(f"\nBest model: {best_reg_name}")
print(f"  MAE  = {best_reg_metrics['MAE']}")
print(f"  RMSE = {best_reg_metrics['RMSE']}")
print(f"  R²   = {best_reg_metrics['R2']}")

print("\n--- CLASSIFICATION RESULTS (Target: Late_delivery_risk) ---")
print(clf_comparison.to_string(index=False))
print(f"\nBest model: {best_clf_name}")
print(f"  Accuracy  = {best_clf_metrics['Accuracy']}")
print(f"  Precision = {best_clf_metrics['Precision']}")
print(f"  Recall    = {best_clf_metrics['Recall']}")
print(f"  F1        = {best_clf_metrics['F1']}")
print(f"  ROC-AUC   = {best_clf_metrics['ROC_AUC']}")

print("\nAll figures saved to:  reports/figures/")
print("Models saved to:       models/")
print("=" * 60)
