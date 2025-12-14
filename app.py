# =========================
# NovaMart Marketing Dashboard
# FINAL VERIFIED VERSION
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# -------------------------
# GLOBAL CONFIG
# -------------------------
st.set_page_config(
    page_title="NovaMart Marketing Analytics",
    layout="wide",
    page_icon="📊"
)

PLOT_THEME = "plotly_dark"

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return {
        "campaign": pd.read_csv("campaign_performance.csv", parse_dates=["date"]),
        "customer": pd.read_csv("customer_data.csv"),
        "product": pd.read_csv("product_sales.csv"),
        "lead": pd.read_csv("lead_scoring_results.csv"),
        "feature": pd.read_csv("feature_importance.csv"),
        "learning": pd.read_csv("learning_curve.csv"),
        "geo": pd.read_csv("geographic_data.csv"),
        "attrib": pd.read_csv("channel_attribution.csv"),
        "funnel": pd.read_csv("funnel_data.csv"),
        "corr": pd.read_csv("correlation_matrix.csv"),
        "journey": pd.read_csv("customer_journey.csv")
    }

data = load_data()

# -------------------------
# SIDEBAR NAV
# -------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Campaign Analytics",
        "Customer Insights",
        "Product Performance",
        "Geographic Analysis",
        "Attribution & Funnel",
        "ML Model Evaluation"
    ]
)

# =========================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================
if page == "Executive Overview":
    st.title("📌 Executive Overview")

    df = data["campaign"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"₹{df['revenue'].sum():,.0f}")
    col2.metric("Total Conversions", f"{df['conversions'].sum():,.0f}")
    col3.metric("Avg ROAS", f"{df['roas'].mean():.2f}")
    col4.metric("Customers", data["customer"].shape[0])

    monthly = df.resample("M", on="date")["revenue"].sum().reset_index()
    fig = px.line(monthly, x="date", y="revenue", title="Revenue Trend",
                  template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 2: CAMPAIGN ANALYTICS
# =========================
elif page == "Campaign Analytics":
    st.title("📈 Campaign Analytics")

    df = data["campaign"]
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    year = st.selectbox("Select Year", sorted(df["year"].unique()))
    dff = df[df["year"] == year]

    # --- Grouped Bar: Region vs Quarter ---
    grp = dff.groupby(["quarter", "region"])["revenue"].sum().reset_index()
    fig = px.bar(grp, x="quarter", y="revenue", color="region",
                 barmode="group", title="Regional Revenue by Quarter",
                 template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # --- Stacked Bar: Campaign Type ---
    percent = st.checkbox("100% Stacked View")
    camp = dff.groupby(["month", "campaign_type"])["spend"].sum().reset_index()
    fig = px.bar(
        camp, x="month", y="spend", color="campaign_type",
        barmode="stack",
        template=PLOT_THEME,
        title="Campaign Type Contribution"
    )
    if percent:
        fig.update_layout(barnorm="percent")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 3: CUSTOMER INSIGHTS
# =========================
elif page == "Customer Insights":
    st.title("👥 Customer Insights")

    df = data["customer"]

    # --- LTV Box Plot ---
    fig = px.box(df, x="segment", y="ltv", points="all",
                 title="Lifetime Value by Segment",
                 template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # --- Income vs LTV ---
    show_trend = st.checkbox("Show Trend Line")
    fig = px.scatter(df, x="income", y="ltv", color="segment",
                     trendline="ols" if show_trend else None,
                     title="Income vs LTV",
                     template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # --- Sunburst ---
    sun = df.groupby(["region", "city_tier", "segment"]).size().reset_index(name="count")
    fig = px.sunburst(
        sun,
        path=["region", "city_tier", "segment"],
        values="count",
        title="Customer Segmentation Breakdown",
        template=PLOT_THEME
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 4: PRODUCT PERFORMANCE
# =========================
elif page == "Product Performance":
    st.title("📦 Product Performance")

    df = data["product"]
    fig = px.treemap(
        df,
        path=["category", "subcategory", "product"],
        values="sales",
        color="profit_margin",
        title="Product Sales Hierarchy",
        template=PLOT_THEME
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 5: GEOGRAPHIC ANALYSIS
# =========================
elif page == "Geographic Analysis":
    st.title("🗺 Geographic Analysis")

    df = data["geo"]

    metric = st.selectbox(
        "Metric",
        ["revenue", "customers", "market_penetration", "yoy_growth"]
    )

    fig = px.choropleth(
        df,
        locations="state",
        locationmode="India states",
        color=metric,
        color_continuous_scale="Viridis",
        title="State-wise Performance",
        template=PLOT_THEME
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 6: ATTRIBUTION & FUNNEL
# =========================
elif page == "Attribution & Funnel":
    st.title("🔄 Attribution & Funnel")

    df = data["attrib"]
    model = st.selectbox("Attribution Model", df.columns[1:])
    fig = px.pie(df, names="channel", values=model, hole=0.4,
                 title="Attribution Model Comparison",
                 template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

    funnel = data["funnel"]
    fig = px.funnel(funnel, x="visitors", y="stage",
                    title="Conversion Funnel",
                    template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 7: ML MODEL EVALUATION
# =========================
elif page == "ML Model Evaluation":
    st.title("🤖 ML Model Evaluation")

    df = data["lead"]
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5)

    y_true = df["actual_converted"]
    y_pred = (df["predicted_probability"] >= threshold).astype(int)

    # --- Confusion Matrix (Counts + %) ---
    cm = confusion_matrix(y_true, y_pred)
    perc = cm / cm.sum(axis=1, keepdims=True) * 100
    text = [[f"{cm[i,j]} ({perc[i,j]:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm, text=text, texttemplate="%{text}",
        colorscale="Blues"
    ))
    fig.update_layout(title="Confusion Matrix",
                      xaxis_title="Predicted",
                      yaxis_title="Actual",
                      template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # --- ROC Curve ---
    fpr, tpr, thr = roc_curve(y_true, df["predicted_probability"])
    auc = roc_auc_score(y_true, df["predicted_probability"])
    idx = np.argmin(np.abs(thr - threshold))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC Curve"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash"))
    fig.add_trace(go.Scatter(
        x=[fpr[idx]], y=[tpr[idx]],
        mode="markers", marker=dict(size=10),
        name=f"Threshold {threshold}"
    ))
    fig.update_layout(
        title=f"ROC Curve (AUC = {auc:.2f})",
        xaxis_title="FPR",
        yaxis_title="TPR",
        template=PLOT_THEME
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Learning Curve with Confidence Bands ---
    lc = data["learning"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lc["train_size"], y=lc["train_score"],
        mode="lines", name="Train"
    ))
    fig.add_trace(go.Scatter(
        x=lc["train_size"], y=lc["val_score"],
        mode="lines", name="Validation"
    ))
    fig.update_layout(title="Learning Curve",
                      template=PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)
