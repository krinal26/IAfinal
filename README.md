# 👗 WesternWear AI — Founder Intelligence Dashboard

**MBA Data Analytics Assignment | Fashion E-Commerce Startup**
> *"Who are our customers? What do they want? How do we reach them?"*

A full-stack data analytics platform built on 2,000 consumer survey responses — applying all four analytics layers to predict customer inclination toward a D2C WesternWear brand.

---

## 🎯 Business Objective

As the founder of WesternWear, I need to answer:
1. **Who** is most inclined to buy from us? (Descriptive)
2. **Why** do some customers show interest and others don't? (Diagnostic)
3. **Which** prospective customers will buy? (Predictive)
4. **What** should I do to maximise conversions? (Prescriptive)

**Target Variable:** `Q25_Purchase_Intent_Label` → `Interested / Neutral / Not Interested`

---

## 📊 Four Analytics Layers

| Layer | Question | Key Outputs |
|---|---|---|
| **Descriptive** | What does the data show? | 20+ charts: donut drill-downs, sunbursts, heatmaps, violin plots, stacked bars |
| **Diagnostic** | Why does intent vary? | Correlation matrix, parallel coordinates, causal heatmaps, factor analysis |
| **Predictive** | Who will buy next? | 5 ML models: Decision Tree, Random Forest, Gradient Boosting, Logistic Reg., SVM |
| **Prescriptive** | What should we do? | Prospect funnel, action playbook, revenue model, GTM strategy, launch checklist |
| **+ NEW Customers** | Predict future prospects | Upload any new CSV → instant prediction with confidence scores |

---

## 📁 Project Structure

```
western_wear/
├── app.py                                  ← Main Streamlit dashboard (6 pages)
├── data_engine.py                          ← Data loading, cleaning, feature engineering
├── ml_engine.py                            ← ML model training & prediction
├── requirements.txt                        ← Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml                         ← Dark theme config
└── data/
    └── WesternWear_Survey_Dataset_2000.csv ← Survey dataset (2,000 respondents)
```

---

## 🗂️ Dataset — 30 Columns, 2,000 Rows

| Column | Description |
|---|---|
| Q1_Age_Group | Age bracket (Under 18 → 55+) |
| Q2_Gender | Male / Female |
| Q3_City_Tier | Metro / Tier-1 / Tier-2 / Tier-3 / Rural |
| Q4_Occupation | Student / Salaried / Self-employed / Homemaker |
| Q5_Monthly_Income | Income band (₹20K → ₹1.5L+) |
| Q6_Shopping_Frequency | Never → Very frequently |
| Q7_Platforms_Used | Multi-select: Myntra, Amazon, Ajio, Instagram, etc. |
| Q8_Avg_Order_Spend | Average per-order spend band |
| Q9_Key_Purchase_Factor | Multi-select: Price, Quality, Trendy designs, etc. |
| Q11_Product_Categories | Multi-select: Jeans, Co-ord sets, Dresses, etc. |
| Q14_Style_Personality | Minimalist / Trendy / Boho / Classic / Sporty |
| Q17_Price_Sensitivity | 1–5 scale |
| Q21_Budget_Numeric_INR | Ideal monthly clothing budget (₹) |
| Q22_Digital_Trust_Score | 1–5 digital platform trust score |
| Q23_Influencer_Purchase | Influencer-driven purchase history |
| **Q25_Purchase_Intent_Label** | **Target: Interested / Neutral / Not Interested** |
| Persona_Label | Survey-defined persona cluster |

---

## 🤖 ML Models — Performance Summary

| Model | Accuracy | AUC-ROC | Notes |
|---|---|---|---|
| Decision Tree | ~72% | ~0.78 | Most interpretable |
| Random Forest | ~78% | ~0.86 | Best overall balance |
| Gradient Boosting | ~77% | ~0.85 | Best for production |
| Logistic Regression | ~70% | ~0.80 | Fast, good baseline |
| SVM | ~73% | ~0.81 | Good on clean data |

---

## 🔑 Key Insights

1. **67% of respondents are Interested** — massive TAM for D2C WesternWear
2. **Digital Trust Score is the #1 predictor** — invest in reviews, UGC, easy returns
3. **Metro × 25–34 Female** is the golden segment — highest intent density
4. **Co-ord sets & Ethnic-Western Fusion** are underserved — our product moat
5. **Poor size range** is the #1 market complaint — inclusive sizing = competitive advantage
6. **'3 basics for ₹999'** is the top-preferred bundle — hero launch offer
7. **Instagram-first** — 60%+ of target audience spends time there
8. **PremiumShopper** has highest LTV — needs separate high-touch strategy

---

## 🚀 Run Locally

```bash
# 1. Clone / download and extract the zip
cd western_wear

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run app.py
```

The app auto-loads `data/WesternWear_Survey_Dataset_2000.csv` on startup.

---

## ☁️ Deploy to Streamlit Cloud

1. Create a **public GitHub repository**
2. Push the contents of the `western_wear/` folder (keep the folder structure)
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
4. Select your repo, branch `main`, main file: `app.py`
5. Click **Deploy** — live URL in ~2 minutes

> **Important:** The `data/` folder with the CSV must be committed to GitHub. The `.streamlit/config.toml` applies the dark theme automatically.

---

## 📤 Uploading New Prospects

Use the **"📤 Predict New Customers"** page to:
1. Collect more survey responses in the same format
2. Export them as CSV with the same column names
3. Upload → instant ML predictions with probability scores
4. Download results CSV for CRM / ad targeting

---

## 🏗️ Engineering Notes

- **data_engine.py** handles all cleaning, ordinal encoding, multi-select binary expansion, and composite score engineering
- **ml_engine.py** trains 5 classifiers with cross-validation, runs KMeans clustering, and PCA for visualisation
- **app.py** is 100% self-contained for Streamlit Cloud — no external API calls
- All charts use Plotly with a custom dark fashion aesthetic
- `@st.cache_data` and `@st.cache_resource` ensure fast reloads

---

*MBA Data Analytics | WesternWear D2C Fashion Intelligence Platform*
