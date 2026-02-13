import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ─── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title = "Customer Revenue Intelligence",
    page_icon  = "📊",
    layout     = "wide"
)

# ─── LOAD DATA ────────────────────────────────────────
@st.cache_data
def load_data():
    base = '../data/processed/'
    master   = pd.read_csv(base + 'master_orders.csv',
                            parse_dates=['order_purchase_timestamp'])
    rfm      = pd.read_csv(base + 'rfm_scores.csv')
    features = pd.read_csv(base + 'customer_features.csv')
    return master, rfm, features

@st.cache_resource
def load_model():
    model    = joblib.load('../models/churn_model.pkl')
    features = joblib.load('../models/feature_names.pkl')
    return model, features

master, rfm, customer_features = load_data()
model, FEATURES = load_model()

# ─── SIDEBAR NAVIGATION ───────────────────────────────
st.sidebar.title("📊 CRIS Dashboard")
st.sidebar.markdown("Customer Revenue Intelligence System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📈 Overview",
     "👥 RFM Segments",
     "🤖 Churn Predictor",
     "🚚 Delivery Insights"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Olist Brazilian E-Commerce")
st.sidebar.caption("Model: XGBoost | AUC: 0.85")


# ════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════
if page == "📈 Overview":
    st.title("📈 Business Overview")
    st.markdown("Key metrics from 25 months of e-commerce data")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue",
               f"R$ {master['total_payment'].sum()/1e6:.2f}M")
    col2.metric("Total Orders",
               f"{len(master):,}")
    col3.metric("Unique Customers",
               f"{master['customer_unique_id'].nunique():,}")
    col4.metric("Avg Order Value",
               f"R$ {master['total_payment'].mean():.0f}")
    col5.metric("Late Delivery Rate",
               f"{master['is_late'].mean()*100:.1f}%")

    st.markdown("---")

    # Monthly Revenue Chart
    st.subheader("Monthly Revenue Trend")
    master['month'] = master['order_purchase_timestamp'].dt.to_period('M').astype('str')
    monthly = master.groupby('month')['total_payment'].sum().reset_index()
    fig = px.area(monthly, x='month', y='total_payment',
                  color_discrete_sequence=['#00ff88'],
                  labels={'total_payment':'Revenue (R$)', 'month':'Month'})
    fig.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Top Categories
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Categories by Revenue")
        cat_rev = (master.groupby('product_category_name_english')['total_payment']
                   .sum().sort_values(ascending=False).head(10).reset_index())
        fig2 = px.bar(cat_rev,
                     x='total_payment',
                     y='product_category_name_english',
                     orientation='h',
                     color_discrete_sequence=['#60aaff'])
        fig2.update_layout(template='plotly_dark', height=380,
                           yaxis_title="", xaxis_title="Revenue (R$)")
        fig2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Payment Method Split")
        pay_split = master['payment_type'].value_counts().reset_index()
        fig3 = px.pie(pay_split, values='count', names='payment_type',
                      color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig3.update_layout(template='plotly_dark', height=380)
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════
# PAGE 2 — RFM SEGMENTS
# ════════════════════════════════════════════════════
elif page == "👥 RFM Segments":
    st.title("👥 Customer Segmentation")
    st.markdown("RFM analysis — Recency, Frequency, Monetary segmentation")

    # Use SQL RFM output (customer_segment column)
    if 'customer_segment' in rfm.columns:
        seg_col = 'customer_segment'
    else:
        seg_col = 'segment'

    seg_summary = rfm.groupby(seg_col).agg(
        customers = (seg_col,      'count'),
        avg_spend = ('monetary',  'mean'),
        total_rev = ('monetary',  'sum')
    ).reset_index()
    seg_summary['revenue_pct'] = (seg_summary['total_rev'] /
                                     seg_summary['total_rev'].sum() * 100).round(1)

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Segments", seg_summary[seg_col].nunique())
    col2.metric("Total Customers", f"{seg_summary['customers'].sum():,}")
    col3.metric("Avg Spend / Customer", f"R$ {rfm['monetary'].mean():.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customers per Segment")
        fig = px.bar(seg_summary.sort_values('customers'),
                     x='customers', y=seg_col,
                     orientation='h',
                     color='customers',
                     color_continuous_scale='Teal')
        fig.update_layout(template='plotly_dark', height=380,
                          yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue % per Segment")
        fig2 = px.pie(seg_summary, values='total_rev', names=seg_col,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(template='plotly_dark', height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Segment Strategy Table
    st.subheader("Segment Strategy")
    strategy = {
        'Champions'          : '🏆 Reward them. Ask for reviews. Early access.',
        'Loyal Customers'    : '💚 Upsell higher value products. Ask for referrals.',
        'Potential Loyalists': '💙 Offer loyalty program. Recommend related items.',
        'Need Attention'     : '🟡 Limited time offer. Re-engagement campaign.',
        'At Risk'            : '🔴 Personalized reactivation email + discount.',
    }
    strat_df = pd.DataFrame([
        {'Segment': k,
         'Customers': seg_summary.loc[seg_summary[seg_col]==k, 'customers'].values[0]
                      if k in seg_summary[seg_col].values else 0,
         'Action': v}
        for k, v in strategy.items()
    ])
    st.dataframe(strat_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# PAGE 3 — CHURN PREDICTOR
# ════════════════════════════════════════════════════
elif page == "🤖 Churn Predictor":
    st.title("🤖 Live Churn Predictor")
    st.markdown("Enter a customer profile to predict churn probability in real time.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Customer Profile")
        frequency       = st.slider("Number of Orders",        1, 20, 1)
        monetary        = st.slider("Total Spend (R$)",        10, 5000, 200)
        avg_order_value = st.slider("Avg Order Value (R$)",    10, 2000, 160)
        avg_review      = st.slider("Avg Review Score",         1, 5,    4)
        avg_delivery    = st.slider("Avg Delivery Days",        1, 60,   12)
        avg_items       = st.slider("Avg Items per Order",      1, 10,    1)
        avg_freight     = st.slider("Avg Freight Value (R$)",  5, 200,  20)
        installments    = st.slider("Avg Installments",         1, 12,    1)
        pct_late        = st.slider("Late Delivery Rate",      0.0, 1.0, 0.1)
        lifespan_days   = st.slider("Customer Lifespan (days)", 0, 730,  0)

    with col2:
        st.subheader("Prediction")

        input_data = pd.DataFrame([{
            'frequency'      : frequency,
            'monetary'       : monetary,
            'avg_order_value': avg_order_value,
            'avg_items'      : avg_items,
            'avg_freight'    : avg_freight,
            'installments'   : installments,
            'avg_review'     : avg_review,
            'pct_late'       : pct_late,
            'avg_delivery'   : avg_delivery,
            'lifespan_days'  : lifespan_days,
        }])[FEATURES]

        churn_prob = model.predict_proba(input_data)[0][1]
        prediction = "🔴 LIKELY TO CHURN" if churn_prob > 0.5 else "🟢 LIKELY ACTIVE"

        st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
        st.markdown(f"### {prediction}")
        st.progress(float(churn_prob))

        if churn_prob > 0.7:
            st.error("⚠️ High churn risk. Send immediate reactivation offer.")
        elif churn_prob > 0.5:
            st.warning("⚡ Moderate risk. Monitor and send engagement email.")
        else:
            st.success("✅ Low churn risk. Focus on upselling.")

        st.markdown("---")
        st.subheader("Top Churn Drivers (Feature Importance)")
        feat_imp = pd.Series(
            model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=True)
        fig = px.bar(feat_imp, orientation='h',
                     color_discrete_sequence=['#c084fc'])
        fig.update_layout(template='plotly_dark', height=320,
                          showlegend=False, xaxis_title="Importance",
                          yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════
# PAGE 4 — DELIVERY INSIGHTS
# ════════════════════════════════════════════════════
elif page == "🚚 Delivery Insights":
    st.title("🚚 Delivery Impact Analysis")
    st.markdown("How delivery speed affects customer satisfaction")
    st.markdown("---")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Delivery Days",  f"{master['delivery_days'].mean():.1f}")
    col2.metric("Late Delivery Rate", f"{master['is_late'].mean()*100:.1f}%")
    col3.metric("Avg Review Score",   f"{master['review_score'].mean():.2f} / 5")

    st.markdown("---")

    # Review score by delivery bucket
    master['delivery_bucket'] = pd.cut(
        master['delivery_days'],
        bins   = [0, 7, 14, 21, 30, 999],
        labels = ['≤1 week','1-2 weeks','2-3 weeks','3-4 weeks','>1 month']
    )
    delivery_summary = master.groupby('delivery_bucket', observed=True).agg(
        avg_review  = ('review_score', 'mean'),
        order_count = ('order_id',     'count'),
        late_pct    = ('is_late',      'mean')
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avg Review Score by Delivery Speed")
        fig = px.bar(delivery_summary,
                     x='delivery_bucket', y='avg_review',
                     color='avg_review',
                     color_continuous_scale='RdYlGn',
                     text=round(delivery_summary['avg_review'],2))
        fig.update_layout(template='plotly_dark', height=360,
                          yaxis_range=[0,5], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Orders per Delivery Bucket")
        fig2 = px.bar(delivery_summary,
                      x='delivery_bucket', y='order_count',
                      color_discrete_sequence=['#60aaff'])
        fig2.update_layout(template='plotly_dark', height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # Key insight callout
    on_time_score = master[master['is_late']==0]['review_score'].mean()
    late_score    = master[master['is_late']==1]['review_score'].mean()
    st.info(f"""
    📊 **Key Finding:** On-time deliveries score **{on_time_score:.2f}/5** vs
    late deliveries at **{late_score:.2f}/5** —
    a drop of **{on_time_score-late_score:.2f} stars** from being late.
    Late orders have higher avg order value (R$172 vs R$158),
    meaning high-value customers are most affected.
    """)