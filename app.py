# ==============================
# NovaMart Marketing Dashboard
# FINAL – Assignment Aligned
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ------------------------------
# PAGE CONFIG & THEME (LOCKED)
# ------------------------------
st.set_page_config(
    page_title="NovaMart Marketing Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_BG = "#0e1117"
TEXT = "#eaeaea"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {APP_BG}; color: {TEXT}; }}
    h1,h2,h3,h4,h5 {{ color: {TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True
)

px.defaults.template = "plotly_dark"

# ------------------------------
# SAFE DATA LOADER
# ------------------------------
@st.cache_data
def load_csv(name):
    try:
        df = pd.read_csv(name)
        df.columns = df.columns.str.lower()
        return df
    except:
        return pd.DataFrame()

campaign = load_csv("campaign_performance.csv")
customer = load_csv("customer_data.csv")
product = load_csv("product_sales.csv")
geo = load_csv("geographic_data.csv")
attrib = load_csv("channel_attribution.csv")
funnel = load_csv("funnel_data.csv")
lead = load_csv("lead_scoring_results.csv")
feature = load_csv("feature_importance.csv")
learning = load_csv("learning_curve.csv")
corr = load_csv("correlation_matrix.csv")

# ------------------------------
# SIDEBAR NAV
# ------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Campaign Analytics",
        "Customer Insights",
        "Product Performance",
        "Geographic Analysis",
        "Attribution & Funnel",
        "ML Model Evaluation",
    ]
)

# ======================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ======================================================
if page == "Executive Overview":
    st.title("📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{campaign.get('revenue',pd.Series()).sum():,.0f}")
    c2.metric("Conversions", f"{campaign.get('conversions',pd.Series()).sum():,.0f}")
    c3.metric("ROAS", f"{campaign.get('roas',pd.Series()).mean():.2f}")
    c4.metric("Customers", f"{customer.shape[0]:,}")

    if not campaign.empty:
        fig = px.line(
            campaign,
            x="date",
            y="revenue",
            title="Revenue Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PAGE 2 — CAMPAIGN ANALYTICS
# ======================================================
elif page == "Campaign Analytics":
    st.title("📣 Campaign Analytics")

    # 1.1 Channel Performance
    metric = st.selectbox("Metric", ["revenue", "conversions", "roas"])
    grp = campaign.groupby("channel")[metric].sum().reset_index()
    fig = px.bar(grp, x=metric, y="channel", orientation="h", title="Channel Performance")
    st.plotly_chart(fig, use_container_width=True)

    # 1.2 Grouped Bar — Region x Quarter
    if {"region","quarter","revenue"}.issubset(campaign.columns):
        grp = campaign.groupby(["quarter","region"])["revenue"].sum().reset_index()
        fig = px.bar(
            grp, x="quarter", y="revenue",
            color="region", barmode="group",
            title="Regional Revenue by Quarter"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 1.3 Stacked Bar — Campaign Type
    if {"month","campaign_type","spend"}.issubset(campaign.columns):
        pct = st.checkbox("100% Stacked View")
        grp = campaign.groupby(["month","campaign_type"])["spend"].sum().reset_index()
        fig = px.bar(
            grp, x="month", y="spend",
            color="campaign_type", barmode="stack",
            title="Campaign Type Contribution"
        )
        if pct:
            fig.update_layout(barnorm="percent")
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PAGE 3 — CUSTOMER INSIGHTS
# ======================================================
elif page == "Customer Insights":
    st.title("👥 Customer Insights")

    # LTV auto-derive if missing
    if "ltv" not in customer.columns and {"avg_order_value","purchases"}.issubset(customer.columns):
        customer["ltv"] = customer["avg_order_value"] * customer["purchases"]

    # 3.1 Histogram
    if "age" in customer.columns:
        fig = px.histogram(customer, x="age", nbins=30, title="Customer Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # 3.2 Box — LTV by Segment
    if {"segment","ltv"}.issubset(customer.columns):
        fig = px.box(customer, x="segment", y="ltv", title="LTV by Segment")
        st.plotly_chart(fig, use_container_width=True)

    # 4.1 Scatter — Income vs LTV
    if {"income","ltv","segment"}.issubset(customer.columns):
        fig = px.scatter(
            customer, x="income", y="ltv",
            color="segment", trendline="ols",
            title="Income vs Lifetime Value"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5.3 Sunburst
    if {"region","city_tier","segment"}.issubset(customer.columns):
        sun = customer.groupby(
            ["region","city_tier","segment"]
        ).size().reset_index(name="count")
        fig = px.sunburst(
            sun, path=["region","city_tier","segment"],
            values="count", title="Customer Segmentation Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PAGE 4 — PRODUCT PERFORMANCE
# ======================================================
elif page == "Product Performance":
    st.title("📦 Product Performance")

    if {"category","subcategory","product","sales","profit_margin"}.issubset(product.columns):
        fig = px.treemap(
            product,
            path=["category","subcategory","product"],
            values="sales",
            color="profit_margin",
            title="Product Sales Hierarchy"
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PAGE 5 — GEOGRAPHIC ANALYSIS
# ======================================================
elif page == "Geographic Analysis":
    st.title("🗺 Geographic Analysis")

    metric = st.selectbox(
        "Metric",
        ["revenue","customers","market_penetration","yoy_growth"]
    )

    if "state" in geo.columns:
        fig = px.choropleth(
            geo,
            locations="state",
            locationmode="India states",
            color=metric,
            color_continuous_scale="Viridis",
            title="State-wise Performance"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PAGE 6 — ATTRIBUTION & FUNNEL
# ======================================================
elif page == "Attribution & Funnel":
    st.title("🔁 Attribution & Funnel")

    model = st.selectbox("Attribution Model", ["first_touch","last_touch","linear"])
    if model in attrib.columns:
        fig = px.pie(
            attrib,
            names="channel",
            values=model,
            hole=0.4,
            title="Attribution Model Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    if {"stage","visitors"}.issubset(funnel.columns):
        fig = px.funnel(
            funnel,
            x="visitors",
            y="stage",
            title="Marketing Funnel"
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PAGE 7 — ML MODEL EVALUATION
# ======================================================
elif page == "ML Model Evaluation":
    st.title("🤖 ML Model Evaluation")

    # Confusion Matrix with %
    if {"actual_converted","predicted_class"}.issubset(lead.columns):
        cm = confusion_matrix(lead["actual_converted"], lead["predicted_class"])
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        fig = go.Figure(
            data=go.Heatmap(
                z=cm_pct,
                text=cm,
                texttemplate="%{text} (%{z:.2%})",
                colorscale="Blues"
            )
        )
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # ROC Curve + Threshold
    if {"actual_converted","predicted_probability"}.issubset(lead.columns):
        fpr, tpr, th = roc_curve(
            lead["actual_converted"],
            lead["predicted_probability"]
        )
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.2f}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines"))
        fig.update_layout(title="ROC Curve")
        st.plotly_chart(fig, use_container_width=True)

    # Learning Curve with bands
    if {"train_size","train_score","val_score"}.issubset(learning.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=learning["train_size"],
            y=learning["train_score"],
            name="Train"
        ))
        fig.add_trace(go.Scatter(
            x=learning["train_size"],
            y=learning["val_score"],
            name="Validation"
        ))
        fig.update_layout(title="Learning Curve")
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    if {"feature","importance"}.issubset(feature.columns):
        fig = px.bar(
            feature.sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
