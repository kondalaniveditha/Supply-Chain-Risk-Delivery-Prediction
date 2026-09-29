"""
dashboard/app.py
================
Supply Chain Risk Analytics — Interactive Streamlit Dashboard

Three pages:
  1. Executive Overview    – KPIs and high-level trends
  2. Delivery Performance  – Drivers of delivery time and delays
  3. Risk & Recommendations – Predictive risk, model insights, actions

Run from project root:
    streamlit run dashboard/app.py
"""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROCESSED_CSV   = os.path.join(ROOT, "data", "processed", "supply_chain_processed.csv")
TRAIN_SUMMARY   = os.path.join(ROOT, "models", "training_summary.json")
REG_COMPARISON  = os.path.join(ROOT, "models", "regression_comparison.csv")
CLF_COMPARISON  = os.path.join(ROOT, "models", "classification_comparison.csv")
FIGURES_DIR     = os.path.join(ROOT, "reports", "figures")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Risk Analytics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #3b82d4;
        margin-bottom: 8px;
    }
    .metric-title { font-size: 13px; color: #57606a; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1f2328; }
    .metric-sub   { font-size: 12px; color: #57606a; }
    .risk-high    { color: #dc2626; font-weight: 700; }
    .risk-medium  { color: #d97706; font-weight: 700; }
    .risk-low     { color: #16a34a; font-weight: 700; }
    .section-header { font-size: 18px; font-weight: 700; color: #1f2328;
                      border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; margin: 16px 0 12px; }
    .insight-box { background: #f0f9ff; border-left: 4px solid #3b82d4;
                   padding: 12px 16px; border-radius: 4px; margin-bottom: 10px; }
    .recommendation { background: #f0fdf4; border-left: 4px solid #22c55e;
                      padding: 12px 16px; border-radius: 4px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_CSV, low_memory=False)
    return df


@st.cache_data
def load_summary():
    if os.path.exists(TRAIN_SUMMARY):
        with open(TRAIN_SUMMARY) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_reg_comparison():
    if os.path.exists(REG_COMPARISON):
        return pd.read_csv(REG_COMPARISON)
    return pd.DataFrame()


@st.cache_data
def load_clf_comparison():
    if os.path.exists(CLF_COMPARISON):
        return pd.read_csv(CLF_COMPARISON)
    return pd.DataFrame()


@st.cache_data
def load_raw_for_dashboard():
    """Load raw (unencoded) processed data for display labels."""
    raw_path = os.path.join(ROOT, "data", "raw", "DataCoSupplyChainDataset.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, encoding="latin1", low_memory=False)
        df["order_date"] = pd.to_datetime(
            df["order date (DateOrders)"], format="%m/%d/%Y %H:%M", errors="coerce"
        )
        return df
    return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚚 Supply Chain Analytics")
    st.markdown("*IBM SkillsBuild Internship Project*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "📦 Delivery Performance",
            "⚠️ Risk & Recommendations",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Filters**")

    df_raw = load_raw_for_dashboard()

    if df_raw is not None:
        all_markets      = ["All"] + sorted(df_raw["Market"].dropna().unique().tolist())
        all_regions      = ["All"] + sorted(df_raw["Order Region"].dropna().unique().tolist())
        all_ship_modes   = ["All"] + sorted(df_raw["Shipping Mode"].dropna().unique().tolist())
        all_segments     = ["All"] + sorted(df_raw["Customer Segment"].dropna().unique().tolist())

        sel_market    = st.selectbox("Market",    all_markets)
        sel_region    = st.selectbox("Region",    all_regions)
        sel_shipmode  = st.selectbox("Ship Mode", all_ship_modes)
        sel_segment   = st.selectbox("Customer Segment", all_segments)
    else:
        sel_market = sel_region = sel_shipmode = sel_segment = "All"
        df_raw = pd.DataFrame()

    st.markdown("---")
    st.markdown("**Data Leakage Policy**")
    st.info(
        "Predictions use only pre-delivery features. "
        "Delivery Status, Shipping Date, Order Status, and Late_delivery_risk "
        "are excluded from all model inputs."
    )


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
def apply_filters(df):
    if df is None or df.empty:
        return df
    if sel_market   != "All": df = df[df["Market"]           == sel_market]
    if sel_region   != "All": df = df[df["Order Region"]     == sel_region]
    if sel_shipmode != "All": df = df[df["Shipping Mode"]    == sel_shipmode]
    if sel_segment  != "All": df = df[df["Customer Segment"] == sel_segment]
    return df


df_filtered = apply_filters(df_raw.copy() if not df_raw.empty else df_raw)
summary     = load_summary()

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def fig_to_st(fig):
    """Display a matplotlib figure in Streamlit."""
    st.pyplot(fig)
    plt.close(fig)


def show_figure(fname):
    path = os.path.join(FIGURES_DIR, fname)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Figure not found: {fname}")


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.caption("How is the supply chain performing?")
    st.markdown("---")

    if df_filtered is not None and not df_filtered.empty:
        total_shipments   = len(df_filtered)
        avg_delivery_days = df_filtered["Days for shipping (real)"].mean()
        late_rate         = df_filtered["Late_delivery_risk"].mean()
        on_time_rate      = 1 - late_rate
        high_risk_count   = int((df_filtered["Late_delivery_risk"] == 1).sum())

        # KPI cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Shipments", f"{total_shipments:,}")
        with c2:
            st.metric("Avg Delivery Days", f"{avg_delivery_days:.2f} days")
        with c3:
            st.metric("On-Time Rate", f"{on_time_rate*100:.1f}%")
        with c4:
            st.metric("Late Delivery Rate", f"{late_rate*100:.1f}%",
                      delta=f"{(late_rate-0.548)*100:+.1f}% vs baseline",
                      delta_color="inverse")
        with c5:
            st.metric("Late Shipments", f"{high_risk_count:,}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Delivery Time Distribution</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            dist = df_filtered["Days for shipping (real)"].value_counts().sort_index()
            colors = ["#ef4444" if i >= 4 else "#3b82d4" for i in dist.index]
            ax.bar(dist.index, dist.values, color=colors, edgecolor="white")
            ax.set_xlabel("Actual Delivery Days", fontsize=11)
            ax.set_ylabel("Shipment Count", fontsize=11)
            ax.set_title("Actual Delivery Days Distribution", fontsize=12, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            from matplotlib.patches import Patch
            legend_els = [Patch(fc="#ef4444", label="≥4 days (risk zone)"),
                          Patch(fc="#3b82d4", label="<4 days")]
            ax.legend(handles=legend_els, fontsize=9)
            plt.tight_layout()
            fig_to_st(fig)

        with col2:
            st.markdown('<div class="section-header">Late vs On-Time Delivery</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            values = [on_time_rate * 100, late_rate * 100]
            labels = ["On-Time / Advance", "Late"]
            colors = ["#22c55e", "#ef4444"]
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90,
                textprops={"fontsize": 11},
            )
            ax.set_title("Delivery Outcome Distribution", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig_to_st(fig)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-header">Performance by Shipping Mode</div>',
                        unsafe_allow_html=True)
            mode_perf = df_filtered.groupby("Shipping Mode").agg(
                avg_days=("Days for shipping (real)", "mean"),
                late_rate=("Late_delivery_risk", "mean"),
                count=("Days for shipping (real)", "count"),
            ).reset_index().sort_values("late_rate", ascending=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(mode_perf["Shipping Mode"],
                           mode_perf["late_rate"] * 100,
                           color=["#ef4444" if r > late_rate else "#3b82d4"
                                  for r in mode_perf["late_rate"]])
            ax.axvline(late_rate * 100, color="black", linestyle="--",
                       linewidth=1.2, label=f"Avg: {late_rate*100:.1f}%")
            ax.set_xlabel("Late Delivery Rate (%)")
            ax.set_title("Late Rate by Shipping Mode", fontsize=12, fontweight="bold")
            for bar, val in zip(bars, mode_perf["late_rate"] * 100):
                ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=10)
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig_to_st(fig)

        with col4:
            st.markdown('<div class="section-header">Performance by Market</div>',
                        unsafe_allow_html=True)
            mkt_perf = df_filtered.groupby("Market").agg(
                late_rate=("Late_delivery_risk", "mean"),
                count=("Days for shipping (real)", "count"),
            ).reset_index().sort_values("late_rate")

            fig, ax = plt.subplots(figsize=(7, 4))
            colors_mkt = ["#ef4444" if r > late_rate else "#3b82d4"
                          for r in mkt_perf["late_rate"]]
            ax.barh(mkt_perf["Market"], mkt_perf["late_rate"] * 100, color=colors_mkt)
            ax.axvline(late_rate * 100, color="black", linestyle="--",
                       linewidth=1.2, label=f"Avg: {late_rate*100:.1f}%")
            ax.set_xlabel("Late Delivery Rate (%)")
            ax.set_title("Late Rate by Market", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig_to_st(fig)

        # Delivery trend
        st.markdown("---")
        st.markdown('<div class="section-header">Monthly Delivery Trend</div>',
                    unsafe_allow_html=True)
        if "order_date" in df_filtered.columns:
            monthly = (
                df_filtered.set_index("order_date")
                .resample("ME")
                .agg(avg_days=("Days for shipping (real)", "mean"),
                     late_rate=("Late_delivery_risk", "mean"))
            )
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            axes[0].plot(monthly.index, monthly["avg_days"], color="#3b82d4", linewidth=2)
            axes[0].set_ylabel("Avg Delivery Days")
            axes[0].set_title("Monthly Delivery Performance", fontsize=13, fontweight="bold")
            axes[0].grid(linestyle="--", alpha=0.4)

            axes[1].plot(monthly.index, monthly["late_rate"] * 100, color="#ef4444", linewidth=2)
            axes[1].set_ylabel("Late Rate (%)")
            axes[1].set_xlabel("Month")
            axes[1].grid(linestyle="--", alpha=0.4)
            axes[1].axhline(late_rate * 100, color="black", linestyle="--",
                            linewidth=1, label=f"Avg {late_rate*100:.1f}%")
            axes[1].legend(fontsize=9)
            plt.tight_layout()
            fig_to_st(fig)
    else:
        st.warning("No data available for the selected filters. Please adjust filter selections.")


# ============================================================
# PAGE 2 — DELIVERY PERFORMANCE
# ============================================================
elif page == "📦 Delivery Performance":
    st.title("📦 Delivery Performance")
    st.caption("What is driving delivery performance?")
    st.markdown("---")

    if df_filtered is not None and not df_filtered.empty:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Avg Delivery Days by Shipping Mode</div>',
                        unsafe_allow_html=True)
            mode_avg = df_filtered.groupby("Shipping Mode")["Days for shipping (real)"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(mode_avg.index, mode_avg.values, color="#3b82d4")
            ax.set_xlabel("Avg Delivery Days")
            ax.axvline(df_filtered["Days for shipping (real)"].mean(),
                       color="red", linestyle="--",
                       label=f"Overall avg: {df_filtered['Days for shipping (real)'].mean():.1f}d")
            ax.legend(fontsize=9)
            for i, v in enumerate(mode_avg.values):
                ax.text(v + 0.02, i, f"{v:.2f}d", va="center", fontsize=10)
            ax.set_title("Average Delivery Days by Shipping Mode", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig_to_st(fig)

        with col2:
            st.markdown('<div class="section-header">Avg Delivery Days by Market</div>',
                        unsafe_allow_html=True)
            mkt_avg = df_filtered.groupby("Market")["Days for shipping (real)"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(mkt_avg.index, mkt_avg.values, color="#7c5cd8")
            ax.set_xlabel("Avg Delivery Days")
            for i, v in enumerate(mkt_avg.values):
                ax.text(v + 0.02, i, f"{v:.2f}d", va="center", fontsize=10)
            ax.set_title("Average Delivery Days by Market", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig_to_st(fig)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-header">Delay Rate by Region (Top 15)</div>',
                        unsafe_allow_html=True)
            region_late = (
                df_filtered.groupby("Order Region")["Late_delivery_risk"].mean()
                .sort_values(ascending=False).head(15)
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            colors_r = ["#ef4444" if v > 0.55 else "#f59e0b" if v > 0.45 else "#22c55e"
                        for v in region_late.values]
            ax.barh(region_late.index[::-1], region_late.values[::-1] * 100,
                    color=colors_r[::-1])
            ax.set_xlabel("Late Delivery Rate (%)")
            ax.set_title("Delay Rate by Region (Top 15)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig_to_st(fig)

        with col4:
            st.markdown('<div class="section-header">Scheduled vs Actual Delivery Days</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            sample = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
            colors_s = sample["Late_delivery_risk"].map({0: "#22c55e", 1: "#ef4444"})
            ax.scatter(sample["Days for shipment (scheduled)"],
                       sample["Days for shipping (real)"],
                       c=colors_s, alpha=0.3, s=12)
            ax.set_xlabel("Scheduled Days")
            ax.set_ylabel("Actual Days")
            ax.set_title("Scheduled vs Actual Delivery Days", fontsize=12, fontweight="bold")
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(fc="#22c55e", label="On Time"),
                                Patch(fc="#ef4444", label="Late")], fontsize=10)
            plt.tight_layout()
            fig_to_st(fig)

        # Category performance
        st.markdown("---")
        st.markdown('<div class="section-header">Delivery Performance by Product Category (Top 15)</div>',
                    unsafe_allow_html=True)
        cat_perf = (
            df_filtered.groupby("Category Name")
            .agg(avg_days=("Days for shipping (real)", "mean"),
                 late_rate=("Late_delivery_risk", "mean"),
                 count=("Days for shipping (real)", "count"))
            .reset_index()
            .sort_values("count", ascending=False)
            .head(15)
            .sort_values("late_rate")
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(cat_perf["Category Name"], cat_perf["avg_days"], color="#3b82d4")
        axes[0].set_xlabel("Avg Delivery Days")
        axes[0].set_title("Avg Days by Category", fontsize=12, fontweight="bold")

        axes[1].barh(cat_perf["Category Name"], cat_perf["late_rate"] * 100, color="#f59e0b")
        axes[1].set_xlabel("Late Rate (%)")
        axes[1].set_title("Late Rate by Category", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig_to_st(fig)

        # Customer Segment
        st.markdown("---")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown('<div class="section-header">Performance by Customer Segment</div>',
                        unsafe_allow_html=True)
            seg_perf = df_filtered.groupby("Customer Segment").agg(
                avg_days=("Days for shipping (real)", "mean"),
                late_rate=("Late_delivery_risk", "mean"),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(seg_perf))
            width = 0.35
            ax.bar(x - width/2, seg_perf["avg_days"], width, label="Avg Days", color="#3b82d4")
            ax2 = ax.twinx()
            ax2.bar(x + width/2, seg_perf["late_rate"] * 100, width,
                    label="Late Rate %", color="#f59e0b", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(seg_perf["Customer Segment"])
            ax.set_ylabel("Avg Delivery Days", color="#3b82d4")
            ax2.set_ylabel("Late Rate (%)", color="#f59e0b")
            ax.set_title("Delivery by Customer Segment", fontsize=12, fontweight="bold")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
            plt.tight_layout()
            fig_to_st(fig)

        with col6:
            st.markdown('<div class="section-header">Summary Statistics</div>',
                        unsafe_allow_html=True)
            summary_stats = pd.DataFrame({
                "Metric": [
                    "Total Shipments",
                    "Average Delivery Days",
                    "Median Delivery Days",
                    "Late Delivery Rate",
                    "On-Time/Advance Rate",
                    "Shipments ≥ 5 Days",
                    "Same-Day Shipments",
                ],
                "Value": [
                    f"{len(df_filtered):,}",
                    f"{df_filtered['Days for shipping (real)'].mean():.2f}",
                    f"{df_filtered['Days for shipping (real)'].median():.0f}",
                    f"{df_filtered['Late_delivery_risk'].mean()*100:.1f}%",
                    f"{(1-df_filtered['Late_delivery_risk'].mean())*100:.1f}%",
                    f"{(df_filtered['Days for shipping (real)']>=5).mean()*100:.1f}%",
                    f"{(df_filtered['Shipping Mode']=='Same Day').sum():,}",
                ],
            })
            st.dataframe(summary_stats, use_container_width=True, hide_index=True)
    else:
        st.warning("No data for the selected filters.")


# ============================================================
# PAGE 3 — RISK & RECOMMENDATIONS
# ============================================================
elif page == "⚠️ Risk & Recommendations":
    st.title("⚠️ Risk Analysis & Recommendations")
    st.caption("Where is the supply chain at risk, and what should management do?")
    st.markdown("---")

    # Model performance summary
    st.markdown('<div class="section-header">Model Performance Summary</div>',
                unsafe_allow_html=True)

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        reg_m = summary.get("regression", {}).get("best_model", {})
        clf_m = summary.get("classification", {}).get("best_model", {})
        with c1:
            st.metric("Best Regression Model", reg_m.get("model", "N/A"))
        with c2:
            st.metric("Regression RMSE", f"{reg_m.get('RMSE', 'N/A')} days")
        with c3:
            st.metric("Best Classifier", clf_m.get("model", "N/A"))
        with c4:
            st.metric("Classifier ROC-AUC", f"{clf_m.get('ROC_AUC', 'N/A')}")

    # Model comparison tables
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Regression Model Comparison**")
        reg_df = load_reg_comparison()
        if not reg_df.empty:
            st.dataframe(reg_df.style.highlight_min(["RMSE", "MAE"], color="#bbf7d0")
                         .highlight_max(["R²"], color="#bbf7d0"),
                         use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Classification Model Comparison**")
        clf_df = load_clf_comparison()
        if not clf_df.empty:
            st.dataframe(clf_df.style.highlight_max(["F1", "ROC-AUC", "Accuracy"], color="#bbf7d0"),
                         use_container_width=True, hide_index=True)

    st.markdown("---")

    # SHAP and feature importance figures
    st.markdown('<div class="section-header">Explainability — Key Predictive Factors</div>',
                unsafe_allow_html=True)
    st.info(
        "**Interpretation note:** Feature importance and SHAP values indicate which variables "
        "are most associated with model predictions. They do not establish causation."
    )

    col3, col4 = st.columns(2)
    with col3:
        show_figure("shap_classification_bar.png")
        st.caption("SHAP Global Importance — Delay Risk Classification")
    with col4:
        show_figure("shap_regression_bar.png")
        st.caption("SHAP Global Importance — Delivery Days Regression")

    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        show_figure("shap_classification_summary.png")
        st.caption("SHAP Summary — Classification")
    with col6:
        show_figure("confusion_matrix.png")
        st.caption("Confusion Matrix — Best Classifier")

    # Risk by region / mode
    st.markdown("---")
    st.markdown('<div class="section-header">Risk Distribution</div>',
                unsafe_allow_html=True)

    if df_filtered is not None and not df_filtered.empty:
        col7, col8 = st.columns(2)

        with col7:
            region_risk = (
                df_filtered.groupby("Order Region")["Late_delivery_risk"].mean()
                .sort_values(ascending=False).head(12)
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            clrs = ["#ef4444" if v > 0.60 else "#f59e0b" if v > 0.45 else "#22c55e"
                    for v in region_risk.values]
            ax.barh(region_risk.index[::-1], region_risk.values[::-1] * 100, color=clrs[::-1])
            ax.set_xlabel("Late Delivery Risk (%)")
            ax.set_title("Late Delivery Risk by Region", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig_to_st(fig)

        with col8:
            mode_risk = (
                df_filtered.groupby("Shipping Mode")["Late_delivery_risk"].mean()
                .sort_values(ascending=False)
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            clrs2 = ["#ef4444" if v > 0.60 else "#f59e0b" if v > 0.45 else "#22c55e"
                     for v in mode_risk.values]
            ax.barh(mode_risk.index[::-1], mode_risk.values[::-1] * 100, color=clrs2[::-1])
            ax.set_xlabel("Late Delivery Risk (%)")
            ax.set_title("Late Delivery Risk by Shipping Mode", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig_to_st(fig)

    # Business Insights
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Key Business Insights</div>',
                unsafe_allow_html=True)

    insights = [
        {
            "fact": "54.8% of shipments in the dataset are delivered late.",
            "insight": "The majority of shipments miss their scheduled delivery window, indicating a systemic — not isolated — operational problem.",
            "risk": "Late delivery can create potential customer-experience and retention risks, although these outcomes are not directly measured in this dataset.",
            "action": "Conduct root-cause analysis at the regional dispatch level and revise scheduling algorithms.",
        },
        {
            "fact": "Standard Class shipping accounts for ~60% of all shipments but has the highest late delivery rate.",
            "insight": "Standard Class accounts for the highest absolute volume of late deliveries in this dataset.",
            "risk": "Over-reliance on Standard Class creates concentrated delivery risk.",
            "action": "Review Standard Class SLA agreements with logistics partners; explore upgrading high-value customers to Second Class.",
        },
        {
            "fact": "Scheduled delivery days (Days for shipment scheduled) is the strongest predictive factor for actual delivery time.",
            "insight": "The scheduling system's own estimates are the best pre-delivery signal for what actually happens — but the model adds value by combining it with category, region, and shipping mode.",
            "risk": "If scheduling underestimates delivery time, downstream SLA violations are predictable in advance.",
            "action": "Integrate the predictive model into the order management system to flag high-risk orders at the time of scheduling.",
        },
        {
            "fact": "The best classifier achieves ROC-AUC ≈ 0.768 on held-out test data (after removing data-leakage features), meaning it ranks approximately 77% of late deliveries above on-time ones.",
            "insight": "The model provides meaningful discrimination between high-risk and low-risk shipments using only pre-delivery information.",
            "risk": "Undeployed, this predictive capability provides no operational value.",
            "action": "Deploy the model as a real-time risk score at order confirmation. Trigger proactive customer notifications for HIGH-risk shipments.",
        },
        {
            "fact": "Same Day delivery mode shows the lowest late-delivery rate but represents only ~5% of shipments.",
            "insight": "Same Day shipping has the lowest observed late-delivery rate among the shipping modes in this dataset.",
            "risk": "Opportunity cost: customers who would pay for premium shipping are defaulted to Standard Class.",
            "action": "Expand Same Day eligibility criteria and offer upgrades to high-value segments.",
        },
    ]

    for i, ins in enumerate(insights, 1):
        with st.expander(f"💡 Insight {i}: {ins['fact'][:70]}..."):
            st.markdown(f"**FACT:** {ins['fact']}")
            st.markdown(f"**INSIGHT:** {ins['insight']}")
            st.markdown(f"**RISK/OPPORTUNITY:** {ins['risk']}")
            st.markdown(f"**ACTION:** {ins['action']}")

    # Recommendations
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Business Recommendations</div>',
                unsafe_allow_html=True)

    recommendations = [
       "**Deploy predictive scoring at order placement** — flag shipments classified as HIGH risk using the project's illustrative 65% probability threshold for proactive handling before dispatch.",
        "**Revise Standard Class SLA** — negotiate tighter delivery windows with logistics partners for the dominant shipping mode.",
        "**Evaluate expedited shipping eligibility** — assess targeted upgrades for high-value Corporate customers based on predicted delivery risk, service requirements, and cost-benefit analysis.",
        "**Regional intervention** — allocate additional logistics resources to the highest-risk regions identified in the risk analysis.",
        "**Monitor scheduled-vs-actual gap** — examine differences between scheduled and predicted delivery time across shipping modes and regions, and recalibrate scheduling rules where the evidence supports it.",
        "**Customer notification** — for any shipment flagged MEDIUM or HIGH risk, trigger an automated notification 24 hours before expected delivery.",
    ]

    for rec in recommendations:
        st.markdown(
            f'<div class="recommendation">✅ {rec}</div>',
            unsafe_allow_html=True,
        )

    # Risk threshold explanation
    st.markdown("---")
    with st.expander("ℹ️ Risk Category Thresholds and Precision-Recall Tradeoff"):
        st.markdown("""
| Category | Late Probability | Notes |
|----------|-----------------|-------|
| 🟢 **LOW** | < 40% | Low chance of late delivery |
| 🟡 **MEDIUM** | 40%–65% | Elevated risk — monitor closely |
| 🔴 **HIGH** | > 65% | High probability of late delivery — proactive action recommended |

**Tradeoff:** Lowering the HIGH threshold increases recall (fewer missed late deliveries) 
but increases false alarms. The 65% threshold is a starting point — operational teams 
should adjust based on the cost ratio of a missed late delivery vs. an unnecessary alert.
        """)
