#  Customer Revenue Intelligence System (CRIS)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.85-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![SQL](https://img.shields.io/badge/SQL-SQLite-orange)

An end-to-end data analytics project built on the Olist Brazilian E-Commerce dataset (96K+ delivered orders). Covers data engineering, advanced SQL analysis, RFM customer segmentation, churn prediction with SHAP explainability, and a live interactive dashboard.

**Live Dashboard → https://customer-revenue-intelligence-pw8qwxaqb3qwu3cntgayt2.streamlit.app/**

---

## Business Problem
70% of customers never return after their first purchase. This project identifies at-risk customers using RFM segmentation and predicts churn probability using machine learning — enabling targeted retention campaigns.

**Churn defined as:** no purchase within 180 days of last order.

---

## Key Findings
- Total Revenue: **R$ 15.4M** across 25 months (15x growth Oct 2016 → Aug 2017)
- Just **11 of 71 categories** drive 80% of revenue (Pareto analysis)
- Late deliveries score **2.57 vs 4.29 stars** — a 1.72 star drop in satisfaction
- **Avg delivery days** is the single strongest churn driver (feature importance)
- Champions (top 5% of customers) generate **~50% of total revenue**
- Churn model **ROC-AUC: 0.8483** — XGBoost with scale_pos_weight + SHAP explainability
- High-value customers most affected by late deliveries (avg order R$172 vs R$158 on-time)

---

## Architecture
```
Raw CSVs (9 tables)
    → Python Cleaning (Pandas)
        → SQLite Database
            → SQL Analysis (5 queries)
            → RFM Segmentation
            → XGBoost Churn Model
                → Streamlit Dashboard (live)
```

---

## Project Structure
| Folder | Contents |
|--------|----------|
| `data/raw/` | Original Olist CSV files (from Kaggle) |
| `data/processed/` | Cleaned master table + feature files |
| `sql/` | 5 advanced SQL query files |
| `notebooks/` | EDA, cleaning, RFM, churn model notebooks |
| `models/` | Trained XGBoost model + feature names (.pkl) |
| `dashboard/` | Streamlit app (app.py) |

---

## Dashboard Pages
| Page | What it shows |
|------|--------------|
| Business Overview | Revenue trend, top categories, payment split |
| RFM Segments | 7 customer segments, revenue per segment, retention strategy |
| Live Churn Predictor | Real-time churn probability from customer profile sliders |
| Delivery Insights | How delivery speed impacts review scores and revenue |

---

## Model Details
| Item | Value |
|------|-------|
| Algorithm | XGBoost Classifier |
| Training rows | 74,685 customers |
| Test rows | 18,672 customers |
| ROC-AUC | 0.8483 |
| Precision (churned) | 0.89 |
| Recall (churned) | 0.74 |
| Class imbalance | Handled via scale_pos_weight |
| Explainability | SHAP TreeExplainer + waterfall plots |
| Top churn driver | avg_delivery (delivery days) |

---

## SQL Analyses
| File | What it does |
|------|-------------|
| `cohort_analysis.sql` | Monthly cohort retention + repeat purchase rate |
| `pareto_revenue.sql` | Running revenue total — identifies top 80% categories |
| `rfm_scoring.sql` | NTILE-based RFM scoring + segment classification |
| `seller_performance.sql` | Seller ranking by revenue, rating, delivery, tier |
| `late_delivery_impact.sql` | Review score vs delivery speed bucketed analysis |

---

## Tech Stack
Python · Pandas · NumPy · SQLite · XGBoost · Scikit-learn · SHAP · Plotly · Streamlit

---

## Run Locally
```bash
git clone https://github.com/your-username/customer-revenue-intelligence
cd customer-revenue-intelligence
pip install -r requirements.txt

# Run notebooks in order: data_cleaning → data_exploration → rfm_segmentation → churn_model
# Then launch dashboard:
cd dashboard
streamlit run app.py
```

> Note: Download the Olist dataset from Kaggle and place CSVs in `data/raw/` before running notebooks.

---

## Dataset
Olist Brazilian E-Commerce — publicly available on Kaggle.
9 tables · 96,477 delivered orders · Sep 2016 – Oct 2018 · 93,357 unique customers.
