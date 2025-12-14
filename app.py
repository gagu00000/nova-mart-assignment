# app.py - NovaMart Marketing Analytics Dashboard (COMPLETE VERSION)
# Place the 11 CSVs in the same folder as this file (root).
# Requirements: streamlit, pandas, numpy, plotly, scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
from datetime import datetime

# ---------------------------
# THEME SETTINGS
# ---------------------------
APP_BG = "#0E1117"
TEXT_COLOR = "#FFFFFF"
SIDEBAR_BLUE = "#1F4E79"
PRIMARY = "#0B3D91"
ACCENT = "#2B8CC4"
PALETTE = [PRIMARY, ACCENT, "#66A3D2", "#B2D4EE", "#F4B400", "#E57373", "#81C784"]

st.set_page_config(page_title="NovaMart Marketing Dashboard", layout="wide", initial_sidebar_state="expanded")

# Plotly dark template with corporate blue palette
pio.templates["hybrid_blue"] = pio.templates["plotly_dark"]
pio.templates["hybrid_blue"].layout.update({
    "paper_bgcolor": APP_BG,
    "plot_bgcolor": APP_BG,
    "font": {"color": TEXT_COLOR, "family": "Arial"},
    "colorway": PALETTE,
    "legend": {"title_font": {"color": TEXT_COLOR}, "font": {"color": TEXT_COLOR}},
    "title": {"x": 0.01, "font": {"color": TEXT_COLOR}},
})
pio.templates.default = "hybrid_blue"

# CSS Styling
st.markdown(f"""
<style>
body, .stApp, .block-container {{
  background-color: {APP_BG} !important;
  color: {TEXT_COLOR} !important;
}}
section[data-testid="stSidebar"] {{
  background-color: {SIDEBAR_BLUE} !important;
  color: {TEXT_COLOR} !important;
}}
div[data-testid="metric-container"] {{
  background: rgba(255,255,255,0.03) !important;
  padding: 10px !important;
  border-radius: 8px;
}}
div[data-testid="metric-container"] .stMetricLabel, div[data-testid="metric-container"] .stMetricValue {{
  color: {TEXT_COLOR} !important;
}}
.stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stMultiSelect, .stSlider {{
  color: {TEXT_COLOR} !important;
}}
.stPlotlyChart > div {{
  background: transparent !important;
}}
h1, h2, h3, h4, h5, p, span, label {{
  color: {TEXT_COLOR} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def safe_read_csv(path, parse_dates=None):
    """Safely load CSV with error handling."""
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except FileNotFoundError:
        st.error(f"❌ File {path} not found. Please upload to root directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading {path}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_all():
    """Load all 11 CSV datasets and enrich campaign data."""
    files = {
        'campaign': "campaign_performance.csv",
        'customer': "customer_data.csv",
        'product': "product_sales.csv",
        'lead': "lead_scoring_results.csv",
        'feature_importance': "feature_importance.csv",
        'learning_curve': "learning_curve.csv",
        'geo': "geographic_data.csv",
        'attribution': "channel_attribution.csv",
        'funnel': "funnel_data.csv",
        'journey': "customer_journey.csv",
        'corr': "correlation_matrix.csv"
    }
    data = {}
    for k, fname in files.items():
        if k == 'campaign':
            data[k] = safe_read_csv(fname, parse_dates=['date'])
        else:
            data[k] = safe_read_csv(fname)

    # Enrich campaign data
    if not data['campaign'].empty and 'date' in data['campaign'].columns:
        try:
            data['campaign']['date'] = pd.to_datetime(data['campaign']['date'])
            data['campaign']['year'] = data['campaign']['date'].dt.year
            data['campaign']['month'] = data['campaign']['date'].dt.strftime('%B')
            data['campaign']['month_num'] = data['campaign']['date'].dt.month
            data['campaign']['quarter'] = data['campaign']['date'].dt.to_period('Q').astype(str)
        except Exception:
            pass

    # Normalize learning curve columns
    if not data['learning_curve'].empty:
        lc = data['learning_curve']
        rename_map = {}
        if 'training_size' in lc.columns and 'train_size' not in lc.columns:
            rename_map['training_size'] = 'train_size'
        if 'validation_score' in lc.columns and 'val_score' not in lc.columns:
            rename_map['validation_score'] = 'val_score'
        if rename_map:
            lc = lc.rename(columns=rename_map)
            data['learning_curve'] = lc

    return data

data = load_all()

# Fix customer data
if 'customer' in data and not data['customer'].empty:
    cust = data['customer']
    cust.columns = cust.columns.str.lower()
    if 'segment' not in cust.columns and 'customer_segment' in cust.columns:
        cust['segment'] = cust['customer_segment']
    if 'ltv' not in cust.columns:
        if {'avg_order_value', 'churn_probability'}.issubset(cust.columns):
            cust['churn_probability'] = cust['churn_probability'].replace(0, 0.001)
            cust['ltv'] = cust['avg_order_value'] / cust['churn_probability']
    data['customer'] = cust

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def df_or_warn(key):
    """Get dataframe or show warning if missing."""
    df = data.get(key)
    if df is None or df.empty:
        st.warning(f"⚠️ Dataset `{key}` missing or empty. Upload `{key}.csv` to enable this chart.")
        return pd.DataFrame()
    return df.copy()

def money(x):
    """Format currency."""
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return x

def export_data(df, filename):
    """Export dataframe as CSV download button."""
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# ---------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------

def kpi_overview():
    """Display executive KPI cards."""
    st.header("Executive Overview")
    df = df_or_warn('campaign')
    cust = df_or_warn('customer')
    c1, c2, c3, c4 = st.columns(4)
    if df.empty:
        c1.metric("Total Revenue", "N/A")
        c2.metric("Total Conversions", "N/A")
        c3.metric("Total Spend", "N/A")
        c4.metric("ROAS", "N/A")
    else:
        total_rev = df['revenue'].sum() if 'revenue' in df.columns else 0
        total_conv = df['conversions'].sum() if 'conversions' in df.columns else 0
        total_spend = df['spend'].sum() if 'spend' in df.columns else 0
        roas = total_rev / total_spend if total_spend else np.nan
        c1.metric("Total Revenue", money(total_rev))
        c2.metric("Total Conversions", f"{int(total_conv):,}")
        c3.metric("Total Spend", money(total_spend))
        c4.metric("ROAS", f"{roas:.2f}" if not np.isnan(roas) else "N/A")
    c4.metric("Customer Count", f"{cust.shape[0]:,}" if not cust.empty else "N/A")

def channel_performance():
    """Horizontal bar chart showing channel performance."""
    st.subheader("Channel Performance Comparison")
    df = df_or_warn('campaign')
    if df.empty:
        return
    metric = st.selectbox("Metric", ['revenue', 'conversions', 'roas'], index=0, key="chan_metric")
    if metric not in df.columns:
        st.warning(f"Metric '{metric}' not found.")
        return
    agg = df.groupby('channel', dropna=False)[metric].sum().reset_index().sort_values(metric)
    fig = px.bar(agg, x=metric, y='channel', orientation='h', text_auto=True, 
                 title=f"Total {metric.title()} by Channel")
    st.plotly_chart(fig, use_container_width=True)
    export_data(agg, f"channel_performance_{metric}.csv")

def grouped_bar_regional():
    """NEW: Grouped bar chart - Regional performance by quarter."""
    st.subheader("Regional Performance by Quarter")
    df = df_or_warn('campaign')
    if df.empty or 'region' not in df.columns or 'quarter' not in df.columns:
        st.warning("campaign_performance.csv must contain 'region' and 'quarter'.")
        return
    
    years = sorted(df['year'].unique()) if 'year' in df.columns else [2023, 2024]
    selected_year = st.selectbox("Select Year", years, key="reg_year")
    
    dff = df[df['year'] == selected_year] if 'year' in df.columns else df
    agg = dff.groupby(['quarter', 'region'])['revenue'].sum().reset_index()
    
    fig = px.bar(agg, x='quarter', y='revenue', color='region', barmode='group',
                 title=f"Revenue by Region and Quarter ({selected_year})",
                 labels={'revenue': 'Revenue (₹)', 'quarter': 'Quarter'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight**: West and South regions consistently outperform. Q4 shows festive season boost.")
    export_data(agg, f"regional_performance_{selected_year}.csv")

def stacked_bar_campaign_type():
    """NEW: Stacked bar chart - Campaign type contribution to monthly spend."""
    st.subheader("Campaign Type Contribution to Monthly Spend")
    df = df_or_warn('campaign')
    if df.empty or 'campaign_type' not in df.columns:
        st.warning("campaign_performance.csv must contain 'campaign_type'.")
        return
    
    view_type = st.radio("View Type", ['Absolute', '100% Stacked'], key="stack_view")
    
    agg = df.groupby(['month_num', 'month', 'campaign_type'])['spend'].sum().reset_index()
    agg = agg.sort_values('month_num')
    
    barmode = 'relative' if view_type == 'Absolute' else 'relative'
    barnorm = '' if view_type == 'Absolute' else 'percent'
    
    fig = px.bar(agg, x='month', y='spend', color='campaign_type', 
                 title=f"Campaign Type Spend by Month ({view_type})",
                 labels={'spend': 'Spend (₹)', 'month': 'Month'},
                 barmode=barmode, barnorm=barnorm if barnorm else None)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight**: Lead Generation campaigns consume largest budget. Seasonal Sale spikes in Q4.")
    export_data(agg, "campaign_type_spend.csv")

def revenue_trend():
    """Line chart showing revenue trend over time."""
    st.subheader("Revenue Trend Over Time")
    df = df_or_warn('campaign')
    if df.empty or 'date' not in df.columns or 'revenue' not in df.columns:
        st.warning("campaign_performance.csv must contain 'date' and 'revenue'.")
        return
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input("Date range", value=(min_date, max_date), 
                                min_value=min_date, max_value=max_date, key="rt_dates")
    agg_level = st.selectbox("Aggregation level", ['Daily', 'Weekly', 'Monthly'], index=2, key="rt_agg")
    channels = st.multiselect("Channels", 
                               options=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], 
                               default=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], 
                               key="rt_channels")
    dff = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
    if channels:
        dff = dff[dff['channel'].isin(channels)]
    if agg_level == 'Daily':
        res = dff.groupby('date')['revenue'].sum().reset_index()
    elif agg_level == 'Weekly':
        res = dff.set_index('date').resample('W')['revenue'].sum().reset_index()
    else:
        res = dff.set_index('date').resample('M')['revenue'].sum().reset_index()
    fig = px.line(res, x='date', y='revenue', title=f"{agg_level} Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

def cumulative_conversions():
    """Stacked area chart showing cumulative conversions by channel."""
    st.subheader("Cumulative Conversions by Channel")
    df = df_or_warn('campaign')
    if df.empty or 'conversions' not in df.columns or 'date' not in df.columns:
        st.warning("campaign_performance.csv must include 'date' and 'conversions'.")
        return
    region = st.selectbox("Region", options=['All'] + (sorted(df['region'].dropna().unique().tolist()) 
                          if 'region' in df.columns else []), key="cum_region")
    dff = df.copy()
    if region != 'All':
        dff = dff[dff['region'] == region]
    tmp = dff.groupby(['date','channel'])['conversions'].sum().reset_index().sort_values('date')
    tmp['cum'] = tmp.groupby('channel')['conversions'].cumsum()
    fig = px.area(tmp, x='date', y='cum', color='channel', title='Cumulative Conversions')
    st.plotly_chart(fig, use_container_width=True)

def age_distribution():
    """Histogram showing customer age distribution."""
    st.subheader("Customer Age Distribution")
    df = df_or_warn('customer')
    if df.empty or 'age' not in df.columns:
        st.warning("customer_data.csv missing 'age'.")
        return
    bins = st.slider("Bins", 5, 50, 20, key="age_bins")
    segs = ['All'] + (df['segment'].dropna().unique().tolist() if 'segment' in df.columns else [])
    seg = st.selectbox("Segment", options=segs, index=0, key="age_seg")
    dff = df.copy()
    if seg != 'All':
        dff = dff[dff['segment'] == seg]
    fig = px.histogram(dff, x='age', nbins=bins, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Customer base skews toward 25-40 age range.")

def ltv_by_segment():
    """Box plot comparing LTV across customer segments."""
    st.subheader("Lifetime Value (LTV) by Segment")
    df = df_or_warn('customer')
    if df.empty or 'ltv' not in df.columns or 'segment' not in df.columns:
        st.warning("customer_data.csv must include 'ltv' and 'segment'.")
        return
    show_points = st.checkbox("Show individual points", key="ltv_points")
    fig = px.box(df, x='segment', y='ltv', points='all' if show_points else 'outliers',
                 title='LTV by Segment')
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Premium segment shows highest median and widest spread.")

def satisfaction_violin():
    """Violin plot showing satisfaction distribution by NPS category."""
    st.subheader("Satisfaction Score Distribution by NPS")
    df = df_or_warn('customer')
    if df.empty or 'satisfaction_score' not in df.columns or 'nps_category' not in df.columns:
        st.warning("customer_data.csv missing satisfaction_score or nps_category.")
        return
    split = 'acquisition_channel' if 'acquisition_channel' in df.columns else None
    if split:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', color=split, 
                       box=True, points='outliers', title='Satisfaction by NPS and Channel')
    else:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', 
                       box=True, points='outliers', title='Satisfaction by NPS')
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Clear separation between groups; Detractors show bimodal distribution.")

def income_vs_ltv():
    """FIXED: Scatter plot with working trendline toggle."""
    st.subheader("Income vs Lifetime Value")
    df = df_or_warn('customer')
    if df.empty or 'income' not in df.columns or 'ltv' not in df.columns:
        st.warning("customer_data.csv must include 'income' and 'ltv'.")
        return
    show_trend = st.checkbox("Show trend line", key="income_trend")
    
    fig = px.scatter(
        df, x='income', y='ltv',
        color='segment' if 'segment' in df.columns else None,
        trendline='ols' if show_trend else None,  # FIXED: Actually add trendline
        title="Income vs LTV",
        hover_data=['customer_id'] if 'customer_id' in df.columns else None
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Positive correlation between income and LTV. Premium customers cluster high.")

def channel_bubble():
    """Bubble chart showing CTR vs Conversion Rate by channel."""
    st.subheader("Channel Performance Matrix (CTR vs Conversion Rate)")
    df = df_or_warn('campaign')
    if df.empty or not set(['ctr','conversion_rate','spend']).issubset(df.columns):
        st.warning("campaign_performance missing ctr/conversion_rate/spend.")
        return
    agg = df.groupby('channel').agg({'ctr':'mean','conversion_rate':'mean','spend':'sum'}).reset_index().dropna()
    fig = px.scatter(agg, x='ctr', y='conversion_rate', size='spend', color='channel', 
                    hover_data=['spend'], title="CTR vs Conversion Rate by Channel")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Email has best CTR-CVR combination. Google Ads in mid-performance zone.")

def correlation_heatmap():
    """Correlation heatmap of marketing metrics."""
    st.subheader("Correlation Heatmap")
    df = df_or_warn('corr')
    if df.empty:
        return
    try:
        fig = px.imshow(df.values, x=df.columns, y=df.index, 
                       color_continuous_scale='RdBu', zmin=-1, zmax=1,
                       text_auto=True, title="Marketing Metrics Correlation")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight**: Strong positive correlation between Spend-Impressions-Clicks.")
    except Exception:
        st.warning("correlation_matrix.csv must be a square matrix.")

def calendar_heatmap():
    """Calendar heatmap showing daily performance."""
    st.subheader("Calendar Heatmap - Daily Performance")
    df = df_or_warn('campaign')
    if df.empty or 'date' not in df.columns:
        st.warning("campaign_performance needs 'date'.")
        return
    metric_options = [c for c in ['revenue','impressions','conversions'] if c in df.columns]
    if not metric_options:
        st.warning("No suitable metric for calendar heatmap.")
        return
    metric = st.selectbox("Metric", metric_options, key="cal_metric")
    d = df.groupby('date')[metric].sum().reset_index()
    d['dow'] = d['date'].dt.weekday
    d['week'] = d['date'].dt.isocalendar().week
    pivot = d.pivot_table(index='dow', columns='week', values=metric, aggfunc='sum')
    fig = px.imshow(pivot, labels=dict(x='Week', y='Day of Week', color=metric), 
                   title="Calendar Heatmap")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Festive season (Oct-Nov) shows intense activity.")

def donut_attribution():
    """Donut chart comparing attribution models."""
    st.subheader("Attribution Model Comparison")
    df = df_or_warn('attribution')
    if df.empty or 'channel' not in df.columns:
        st.warning("channel_attribution.csv missing or invalid.")
        return
    models = [c for c in df.columns if c != 'channel']
    model = st.selectbox("Attribution model", models, key="attr_model")
    fig = px.pie(df, names='channel', values=model, hole=0.5, 
                title=f"Attribution - {model}")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Google Ads gets more credit in first-touch; Email in last-touch.")

def treemap_products():
    """Treemap showing product sales hierarchy."""
    st.subheader("Product Sales Treemap")
    df = df_or_warn('product')
    if df.empty:
        return
    path = [c for c in ['category','subcategory','product_name'] if c in df.columns]
    if not path:
        st.warning("product_sales.csv missing hierarchy columns.")
        return
    fig = px.treemap(df, path=path, values='sales', 
                    color='profit_margin' if 'profit_margin' in df.columns else None,
                    title="Product Treemap",
                    color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Electronics dominates sales volume; Fashion has highest margins.")

def sunburst_segmentation():
    """NEW: Sunburst chart for customer segmentation breakdown."""
    st.subheader("Customer Segmentation Breakdown")
    df = df_or_warn('customer')
    if df.empty:
        return
    
    # Build hierarchy path
    path_cols = []
    for col in ['region', 'city_tier', 'segment']:
        if col in df.columns:
            path_cols.append(col)
    
    if len(path_cols) < 2:
        st.warning("customer_data.csv needs at least 2 of: region, city_tier, segment")
        return
    
    # Aggregate customer counts
    agg = df.groupby(path_cols).size().reset_index(name='customer_count')
    
    fig = px.sunburst(agg, path=path_cols, values='customer_count',
                     title="Customer Segmentation: Region > City Tier > Segment")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Metro cities (Tier 1) have higher Premium segment concentration.")
    export_data(agg, "customer_segmentation.csv")

def funnel_viz():
    """Funnel chart showing conversion funnel."""
    st.subheader("Marketing Conversion Funnel")
    df = df_or_warn('funnel')
    if df.empty or not set(['stage','visitors']).issubset(df.columns):
        st.warning("funnel_data.csv requires 'stage' and 'visitors'.")
        return
    fig = px.funnel(df, x='visitors', y='stage', title="Conversion Funnel")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Biggest drop-off between Awareness and Interest (55% loss).")

def sankey_journey():
    """BONUS: Sankey diagram for customer journey."""
    st.subheader("Customer Journey Flow (Sankey)")
    df = df_or_warn('journey')
    if df.empty:
        return
    
    required = ['source', 'target', 'value']
    if not all(col in df.columns for col in required):
        st.warning("customer_journey.csv must contain: source, target, value")
        return
    
    # Create node labels
    all_nodes = list(set(df['source'].tolist() + df['target'].tolist()))
    node_dict = {node: idx for idx, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=0.5),
            label=all_nodes
        ),
        link=dict(
            source=[node_dict[src] for src in df['source']],
            target=[node_dict[tgt] for tgt in df['target']],
            value=df['value'].tolist()
        )
    )])
    fig.update_layout(title="Customer Journey Paths", font_size=10)
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Visualizes multi-touchpoint customer paths and conversion flows.")

def choropleth_map():
    """NEW: Choropleth map showing state-wise performance."""
    st.subheader("State-wise Performance (Choropleth Map)")
    df = df_or_warn('geo')
    if df.empty:
        return
    
    candidates = [c for c in ['total_revenue','total_customers','market_penetration','yoy_growth'] 
                  if c in df.columns]
    if not candidates:
        st.warning("geographic_data.csv needs metrics: total_revenue/total_customers/market_penetration/yoy_growth.")
        return
    
    metric = st.selectbox("Metric", candidates, key="choro_metric")
    
    # India GeoJSON (simplified for demonstration - you'd use full GeoJSON in production)
    india_states = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Check if we have state column
    if 'state' not in df.columns:
        st.warning("geographic_data.csv must contain 'state' column for choropleth.")
        return
    
    # Attempt to create choropleth using state names
    fig = px.choropleth(
        df,
        locations='state',
        locationmode='country names',
        color=metric,
        hover_name='state',
        color_continuous_scale='Blues',
        title=f"India State-wise {metric.replace('_', ' ').title()}"
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Maharashtra and Karnataka are top performers. Eastern states show growth potential.")
    export_data(df, f"geographic_{metric}.csv")

def bubble_map():
    """Bubble map showing store performance by location."""
    st.subheader("Store Performance (Bubble Map)")
    df = df_or_warn('geo')
    if df.empty:
        return
    
    lat_col = 'latitude' if 'latitude' in df.columns else ('lat' if 'lat' in df.columns else None)
    lon_col = 'longitude' if 'longitude' in df.columns else ('lon' if 'lon' in df.columns else None)
    
    if not (lat_col and lon_col):
        st.warning("geographic_data.csv needs latitude/longitude columns.")
        return
    
    size_col = 'store_count' if 'store_count' in df.columns else 'total_customers'
    color_col = 'customer_satisfaction' if 'customer_satisfaction' in df.columns else 'total_revenue'
    
    fig = px.scatter_geo(
        df, lat=lat_col, lon=lon_col, size=size_col, color=color_col,
        hover_name='state' if 'state' in df.columns else None,
        projection="natural earth",
        title="Store Performance by Location"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Insight**: Large clusters in metro areas. Kerala shows high satisfaction.")

def confusion_matrix_viz():
    """Confusion matrix with threshold slider."""
    st.subheader("Confusion Matrix - Lead Scoring Model")
    df = df_or_warn('lead')
    if df.empty or not set(['actual_converted','predicted_probability']).issubset(df.columns):
        st.warning("lead_scoring_results.csv needs actual_converted and predicted_probability.")
        return
    
    thr = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.
