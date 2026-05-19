"""
Folio — SMB Financial Health Dashboard
Portfolio Overview · Company Explorer · Signal Analysis
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import json

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report as sk_classification_report

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Folio | SMB Health",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DB_PATH       = ROOT / "data" / "smb_health.duckdb"
CHART_PATH    = ROOT / "models" / "risk_model" / "feature_importance.png"
METADATA_PATH = ROOT / "models" / "risk_model" / "model_metadata.json"

SEG_COLORS = {
    "healthy": "#10B981",
    "watch":   "#F59E0B",
    "at_risk": "#EF4444",
}
SEG_ORDER  = ["healthy", "watch", "at_risk"]
TEMPLATE   = "plotly_dark"

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 18px 22px;
}
[data-testid="stSidebar"] { padding-top: 1.5rem; }
.stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    ch = con.execute("SELECT * FROM mart_company_health").df()

    dv = con.execute("""
        SELECT txn_date, health_segment,
               SUM(daily_volume)       AS daily_volume,
               SUM(daily_txn_count)    AS daily_txn_count,
               SUM(daily_failed_count) AS daily_failed_count
        FROM mart_daily_volume
        GROUP BY txn_date, health_segment
        ORDER BY txn_date
    """).df()

    pf = con.execute("""
        SELECT failure_reason,
               SUM(failure_count)      AS failure_count,
               SUM(total_amount_failed) AS total_amount_failed
        FROM mart_payment_failures
        GROUP BY failure_reason
    """).df()

    ms = con.execute("SELECT * FROM model_scores").df()
    con.close()

    dv["txn_date"] = pd.to_datetime(dv["txn_date"])

    # Merge model scores into the company table
    df = ch.merge(
        ms[["company_id", "predicted_segment", "correct",
            "score_proba_healthy", "score_proba_watch", "score_proba_at_risk"]],
        on="company_id",
        how="left",
    )
    df["correct"] = df["correct"].astype(bool)
    return df, dv, pf, ms


# ── helpers ───────────────────────────────────────────────────────────────────
def _style_seg(df: pd.DataFrame, col: str) -> "pd.io.formats.style.Styler":
    css = {
        "at_risk": "background-color:rgba(239,68,68,.18);color:#fca5a5;font-weight:600",
        "watch":   "background-color:rgba(245,158,11,.18);color:#fcd34d;font-weight:600",
        "healthy": "background-color:rgba(16,185,129,.18);color:#6ee7b7;font-weight:600",
    }
    return df.style.map(lambda v: css.get(v, ""), subset=[col])


def _pct(v) -> str:
    return "—" if pd.isna(v) else f"{v:.1%}"

def _days(v) -> str:
    return "—" if pd.isna(v) else f"{v:+.1f}d"

def _x(v) -> str:
    return "—" if pd.isna(v) else f"{v:.2f}×"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview(df: pd.DataFrame) -> None:
    st.title("Portfolio Overview")

    total       = len(df)
    at_risk_n   = (df["health_segment"] == "at_risk").sum()
    avg_score   = df["score_proba_healthy"].mean()
    avg_miss    = df["missed_payment_rate"].mean()

    # ── metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies Monitored", f"{total:,}")
    c2.metric("Portfolio At-Risk",  f"{at_risk_n / total:.1%}",
              f"{at_risk_n} companies flagged", delta_color="inverse")
    c3.metric("Avg Health Score",   f"{avg_score:.1%}")
    c4.metric("Missed Payment Rate", f"{avg_miss:.1%}", delta_color="off")

    st.divider()

    # ── donut + histogram ──────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Risk Tier Distribution")
        vc = (
            df["health_segment"]
            .value_counts()
            .reindex(SEG_ORDER)
            .reset_index()
        )
        vc.columns = ["segment", "count"]
        fig_donut = go.Figure(go.Pie(
            labels=vc["segment"],
            values=vc["count"],
            hole=0.56,
            marker_colors=[SEG_COLORS[s] for s in vc["segment"]],
            textinfo="label+percent",
            textfont_size=13,
            direction="clockwise",
            sort=False,
        ))
        fig_donut.update_layout(
            template=TEMPLATE, height=320,
            margin=dict(t=10, b=30, l=10, r=10),
            legend=dict(orientation="h", y=-0.08),
            annotations=[dict(text=f"{total:,}<br>companies",
                              x=0.5, y=0.5, font_size=14,
                              showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        st.subheader("Health Score Distribution")
        plot_df = df.dropna(subset=["score_proba_healthy"])
        fig_hist = px.histogram(
            plot_df,
            x="score_proba_healthy",
            color="health_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"health_segment": SEG_ORDER},
            nbins=40,
            opacity=0.82,
            barmode="overlay",
            labels={"score_proba_healthy": "Health Score  P(healthy)",
                    "health_segment": "Segment"},
            template=TEMPLATE,
        )
        fig_hist.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.18),
            yaxis_title="Companies",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── needs attention table ─────────────────────────────────────────────────
    st.subheader("⚠  Needs Attention — Bottom 15 by Health Score")

    attn = (
        df.dropna(subset=["score_proba_healthy"])
        .nsmallest(15, "score_proba_healthy")
        [["company_name", "industry", "health_segment",
          "missed_payment_rate", "avg_days_late", "spend_trend"]]
        .copy()
        .rename(columns={
            "company_name":        "Company",
            "industry":            "Industry",
            "health_segment":      "Segment",
            "missed_payment_rate": "Missed Pay%",
            "avg_days_late":       "Avg Days Late",
            "spend_trend":         "Spend Trend",
        })
    )
    attn["Missed Pay%"]   = attn["Missed Pay%"].apply(_pct)
    attn["Avg Days Late"] = attn["Avg Days Late"].apply(_days)
    attn["Spend Trend"]   = attn["Spend Trend"].apply(_pct)

    st.dataframe(
        _style_seg(attn, "Segment"),
        use_container_width=True,
        hide_index=True,
        height=530,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPANY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def _company_detail(row: pd.Series, df: pd.DataFrame) -> None:
    """Renders the detail panel for one selected company."""
    with st.expander(f"📋  {row['company_name']}  —  Detail View", expanded=True):
        left, right = st.columns([1, 2])

        with left:
            score = float(row.get("score_proba_healthy") or 0)
            gauge_color = (
                SEG_COLORS["at_risk"] if score < 0.33 else
                SEG_COLORS["watch"]   if score < 0.66 else
                SEG_COLORS["healthy"]
            )
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(score * 100, 1),
                number={"suffix": "%", "font": {"size": 30, "color": gauge_color}},
                title={"text": "Health Score", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#666",
                             "tickfont": {"size": 10}},
                    "bar":  {"color": gauge_color, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  33], "color": "rgba(239,68,68,.12)"},
                        {"range": [33, 66], "color": "rgba(245,158,11,.12)"},
                        {"range": [66,100], "color": "rgba(16,185,129,.12)"},
                    ],
                    "threshold": {
                        "value": score * 100,
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                    },
                },
            ))
            fig_gauge.update_layout(
                template=TEMPLATE, height=220,
                margin=dict(t=20, b=0, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            actual    = row.get("health_segment", "—")
            predicted = row.get("predicted_segment", "—")
            correct   = row.get("correct", None)
            icon      = "✅" if correct else "⚠️"
            st.markdown(
                f"**Actual:** `{actual}`  \n"
                f"**Model prediction:** `{predicted}` {icon}"
            )

        with right:
            # probability bar
            ph = float(row.get("score_proba_healthy") or 0)
            pw = float(row.get("score_proba_watch")   or 0)
            pr = float(row.get("score_proba_at_risk") or 0)

            fig_proba = go.Figure()
            for label, val, color in [
                ("P(healthy)", ph, SEG_COLORS["healthy"]),
                ("P(watch)",   pw, SEG_COLORS["watch"]),
                ("P(at_risk)", pr, SEG_COLORS["at_risk"]),
            ]:
                fig_proba.add_trace(go.Bar(
                    name=label,
                    x=[val],
                    y=["Segment probability"],
                    orientation="h",
                    marker_color=color,
                    text=f"{val:.1%}",
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=12, color="white"),
                ))
            fig_proba.update_layout(
                barmode="stack",
                template=TEMPLATE,
                height=90,
                margin=dict(t=4, b=4, l=0, r=0),
                showlegend=True,
                legend=dict(orientation="h", y=-1.2, font_size=11),
                xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig_proba, use_container_width=True)

            st.markdown("##### Signals vs Portfolio Average")
            signals = [
                ("Missed Payment Rate", "missed_payment_rate", _pct),
                ("Txn Failure Rate",    "txn_failure_rate",    _pct),
                ("Avg Days Late",       "avg_days_late",       _days),
                ("Spend Trend",         "spend_trend",         _pct),
                ("Credit Utilization",  "credit_utilization",  _x),
            ]
            sig_rows = []
            for label, col, fmt in signals:
                cv = row.get(col, np.nan)
                pa = df[col].mean()
                sig_rows.append({
                    "Signal":        label,
                    "This Company":  fmt(cv),
                    "Portfolio Avg": fmt(pa),
                })
            st.dataframe(
                pd.DataFrame(sig_rows),
                use_container_width=True,
                hide_index=True,
                height=215,
            )


def page_explorer(df: pd.DataFrame) -> None:
    # ── sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**Filters**")
        industries = st.multiselect(
            "Industry",
            sorted(df["industry"].unique()),
            default=sorted(df["industry"].unique()),
        )
        tiers = st.multiselect(
            "Risk Tier",
            SEG_ORDER,
            default=SEG_ORDER,
        )
        sizes = st.multiselect(
            "Company Size",
            sorted(df["company_size"].unique()),
            default=sorted(df["company_size"].unique()),
        )
        regions = st.multiselect(
            "Region",
            sorted(df["region"].unique()),
            default=sorted(df["region"].unique()),
        )

    mask = (
        df["industry"].isin(industries) &
        df["health_segment"].isin(tiers) &
        df["company_size"].isin(sizes) &
        df["region"].isin(regions)
    )
    filtered = df[mask].reset_index(drop=True)

    # ── main table ────────────────────────────────────────────────────────────
    st.title("Company Explorer")
    st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** companies")

    disp = (
        filtered[[
            "company_name", "industry", "company_size", "region",
            "health_segment", "score_proba_healthy",
            "missed_payment_rate", "txn_failure_rate",
            "avg_days_late", "spend_trend", "credit_utilization",
        ]]
        .rename(columns={
            "company_name":        "Company",
            "industry":            "Industry",
            "company_size":        "Size",
            "region":              "Region",
            "health_segment":      "Segment",
            "score_proba_healthy": "Health Score",
            "missed_payment_rate": "Missed Pay%",
            "txn_failure_rate":    "Failure Rate",
            "avg_days_late":       "Avg Days Late",
            "spend_trend":         "Spend Trend",
            "credit_utilization":  "Credit Util",
        })
        .copy()
    )
    disp["Health Score"]  = disp["Health Score"].apply(_pct)
    disp["Missed Pay%"]   = disp["Missed Pay%"].apply(_pct)
    disp["Failure Rate"]  = disp["Failure Rate"].apply(_pct)
    disp["Avg Days Late"] = disp["Avg Days Late"].apply(_days)
    disp["Spend Trend"]   = disp["Spend Trend"].apply(_pct)
    disp["Credit Util"]   = disp["Credit Util"].apply(_x)

    event = st.dataframe(
        _style_seg(disp, "Segment"),
        use_container_width=True,
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key="explorer_table",
    )

    # ── company detail panel ──────────────────────────────────────────────────
    if event.selection.rows:
        idx = event.selection.rows[0]
        _company_detail(filtered.iloc[idx], df)
    else:
        st.caption("↑ Click any row to open the company detail panel")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIGNAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def page_signals(df: pd.DataFrame, dv: pd.DataFrame,
                 pf: pd.DataFrame, ms: pd.DataFrame) -> None:
    st.title("Signal Analysis")

    # ── row 1: feature importance ─────────────────────────────────────────────
    st.subheader("Feature Importance")
    if CHART_PATH.exists():
        st.image(str(CHART_PATH), use_container_width=True)
        st.caption(
            "Weight = number of times a feature is used to split data across all XGBoost trees. "
            "Spend-volume signals dominate: **total_spend** and **avg_txn_amount** rank first and "
            "second, with **txn_failure_rate** close behind. "
            "Categorical features (industry, region, tenure) carry minimal weight — "
            "the model cares about *behavior*, not *identity*."
        )
    else:
        st.warning("Feature importance chart not found. Run `python3 models/risk_model/train.py` first.")

    st.divider()

    # ── row 2: daily volume + failure breakdown ───────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Daily Transaction Volume")
        min_d = dv["txn_date"].min().date()
        max_d = dv["txn_date"].max().date()
        date_range = st.slider(
            "Date range",
            min_value=min_d,
            max_value=max_d,
            value=(min_d, max_d),
            key="vol_slider",
            label_visibility="collapsed",
            format="YYYY-MM-DD",
        )
        dv_filt = dv[
            (dv["txn_date"].dt.date >= date_range[0]) &
            (dv["txn_date"].dt.date <= date_range[1])
        ]
        fig_vol = px.line(
            dv_filt,
            x="txn_date",
            y="daily_volume",
            color="health_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"health_segment": SEG_ORDER},
            labels={"txn_date": "Date", "daily_volume": "Daily Volume ($)",
                    "health_segment": "Segment"},
            template=TEMPLATE,
        )
        fig_vol.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.18),
        )
        fig_vol.update_traces(line_width=1.5)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_r:
        st.subheader("Failure Reason Breakdown")
        pf_sorted = pf.sort_values("failure_count", ascending=True)
        fig_fail  = px.bar(
            pf_sorted,
            x="failure_count",
            y="failure_reason",
            orientation="h",
            color="failure_reason",
            color_discrete_sequence=["#60A5FA", "#34D399", "#FBBF24", "#F87171"],
            labels={"failure_count": "Failed Transactions", "failure_reason": "Reason"},
            template=TEMPLATE,
            text_auto=True,
        )
        fig_fail.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        fig_fail.update_traces(textposition="outside")
        st.plotly_chart(fig_fail, use_container_width=True)

    st.divider()

    # ── row 3: cohort box plots ───────────────────────────────────────────────
    st.subheader("Cohort Comparison — Behavioral Signals by Risk Tier")
    bc1, bc2, bc3 = st.columns(3)
    box_metrics = [
        (bc1, "missed_payment_rate", "Missed Payment Rate"),
        (bc2, "txn_failure_rate",    "Txn Failure Rate"),
        (bc3, "avg_days_late",       "Avg Days Late  (days)"),
    ]
    for col_widget, col_name, col_label in box_metrics:
        fig_box = px.box(
            df,
            x="health_segment",
            y=col_name,
            color="health_segment",
            color_discrete_map=SEG_COLORS,
            category_orders={"health_segment": SEG_ORDER},
            labels={"health_segment": "", col_name: col_label},
            template=TEMPLATE,
            points=False,
        )
        fig_box.update_layout(
            height=310,
            margin=dict(t=30, b=10, l=10, r=10),
            showlegend=False,
            title=dict(text=col_label, font=dict(size=12), x=0.5),
        )
        col_widget.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ── row 4: model performance summary ─────────────────────────────────────
    st.subheader("Model Performance Summary")
    perf_l, perf_r = st.columns([1, 1])

    with perf_l:
        # Held-out test-set metrics written by train.py
        if METADATA_PATH.exists():
            meta = json.loads(METADATA_PATH.read_text())
        else:
            meta = {"accuracy": None, "macro_f1": None, "auc_roc": None, "test_set_size": "?"}

        n   = meta.get("test_set_size", "?")
        acc = meta.get("accuracy")
        f1  = meta.get("macro_f1")
        auc = meta.get("auc_roc")

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Accuracy  (test n={n})",  f"{acc:.1%}"  if acc is not None else "—")
        m2.metric(f"Macro F1  (test n={n})",  f"{f1:.3f}"   if f1  is not None else "—")
        m3.metric(f"AUC-ROC   (test n={n})",  f"{auc:.3f}"  if auc is not None else "—")
        st.markdown("")

        report_dict = sk_classification_report(
            ms["health_segment"], ms["predicted_segment"],
            output_dict=True, zero_division=0,
        )
        perf_rows = []
        for seg in SEG_ORDER:
            m = report_dict.get(seg, {})
            perf_rows.append({
                "Class":     seg,
                "Precision": f"{m.get('precision', 0):.3f}",
                "Recall":    f"{m.get('recall',    0):.3f}",
                "F1":        f"{m.get('f1-score',  0):.3f}",
                "Support":   int(m.get("support", 0)),
            })
        st.caption("Per-class breakdown on full cohort (n=1,000)")
        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(
            _style_seg(perf_df, "Class"),
            use_container_width=True,
            hide_index=True,
        )

    with perf_r:
        st.markdown("##### Context")
        st.info(
            "XGBoost achieves **89.5% accuracy** (macro F1: 0.849, AUC-ROC: 0.979) on a "
            "held-out test set of 200 companies. The **watch** segment is hardest to classify — "
            "boundary cases between healthy and at-risk create natural ambiguity, which mirrors "
            "real-world label noise in SMB risk scoring.\n\n"
            "In production, Folio would train on **lagged signals** (Q1 behavior → Q2 label) "
            "and tune probability thresholds per risk appetite."
        )


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    df, dv, pf, ms = load_all()

    with st.sidebar:
        st.markdown("## 📊 Folio")
        st.markdown("*SMB Financial Health Platform*")
        st.divider()
        page = st.radio(
            "Navigate to",
            ["Portfolio Overview", "Company Explorer", "Signal Analysis"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"**{len(df):,}** companies · **{len(df.columns)}** signals")

    if page == "Portfolio Overview":
        page_overview(df)
    elif page == "Company Explorer":
        page_explorer(df)
    else:
        page_signals(df, dv, pf, ms)


if __name__ == "__main__":
    main()
