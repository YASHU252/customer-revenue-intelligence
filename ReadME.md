# 📊 Customer Revenue Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.85-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)

An end-to-end data analytics project built on the Olist Brazilian E-Commerce dataset (100k+ orders). Covers data engineering, advanced SQL analysis, customer segmentation, churn prediction, and a live interactive dashboard.

🚀 **Live Dashboard → [your-app-name].streamlit.app**

---

## 🎯 Business Problem
70% of customers never return after their first purchase. This project identifies at-risk customers using RFM segmentation and predicts churn probability using machine learning — enabling targeted retention campaigns.
Churn defined as no purchase within 180 days of last order.
## 📈 Key Findings
- Total Revenue: **R$ 15.4M** across 25 months (15x growth Oct'16→Aug'17)
- Just **11 of 71 categories** drive 80% of revenue (Pareto)
- Late deliveries score **2.57 vs 4.29 stars** — a 1.72 star drop
- Champions (5% of customers) generate **~50% of revenue**
- Churn model **AUC: 0.85** using XGBoost + SHAP explainability

## 🏗️ Architecture
```
Raw CSVs → Python Cleaning → SQLite DB → SQL Analysis → RFM Segmentation → ML Model → Streamlit Dashboard
```

## 📁 Structure
| Folder | Contents |
|--------|----------|
| `data/processed/` | Cleaned master tables |
| `sql/` | 5 advanced SQL query files |
| `notebooks/` | EDA, cleaning, RFM, ML notebooks |
| `models/` | Trained XGBoost model (.pkl) |
| `dashboard/` | Streamlit app |

## 🛠️ Tech Stack
Python · Pandas · NumPy · SQLite · XGBoost · Scikit-learn · SHAP · Plotly · Streamlit

## ▶️ Run Locally
```bash
pip install -r requirements.txt
cd dashboard
streamlit run app.py
```

## 📊 Dataset
Olist Brazilian E-Commerce — free on Kaggle. 9 tables, 100k+ orders, Sep 2016 – Oct 2018.
