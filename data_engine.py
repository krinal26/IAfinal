"""
data_engine.py  — WesternWear Analytics
All data loading, cleaning, feature engineering and encoding in one place.
Import this in app.py and every page module.
"""

import pandas as pd
import numpy as np
import streamlit as st

# ── Ordered mappings ────────────────────────────────────────────────────────
AGE_ORDER      = ['Under 18','18–24','25–34','35–44','45–54','55+']
INCOME_ORDER   = ['Below ₹20,000','₹20,000–₹40,000','₹40,000–₹75,000','₹75,000–₹1.5L','Above ₹1.5L']
FREQ_ORDER     = ['Never','Rarely','Occasionally','Frequently','Very frequently']
SPEND_ORDER    = ['Below ₹500','₹500–₹1,000','₹1,000–₹2,500','₹2,500–₹5,000','Above ₹5,000']
INTENT_ORDER   = ['Definitely not','Unlikely','Neutral','Somewhat likely','Very likely']
INTENT_LABEL   = {'Definitely not': 0,'Unlikely': 1,'Neutral': 2,'Somewhat likely': 3,'Very likely': 4}
LABEL_MAP      = {'Not Interested': 0,'Neutral': 1,'Interested': 2}
CITY_ORDER     = ['Metro','Tier-1','Tier-2','Tier-3 / Rural']

INCOME_NUMERIC = {
    'Below ₹20,000': 15000,
    '₹20,000–₹40,000': 30000,
    '₹40,000–₹75,000': 57500,
    '₹75,000–₹1.5L': 112500,
    'Above ₹1.5L': 175000,
}
SPEND_NUMERIC = {
    'Below ₹500': 300,
    '₹500–₹1,000': 750,
    '₹1,000–₹2,500': 1750,
    '₹2,500–₹5,000': 3750,
    'Above ₹5,000': 6000,
}
FREQ_NUMERIC = {
    'Never': 0,'Rarely': 1,'Occasionally': 2,'Frequently': 3,'Very frequently': 4
}

MULTI_SEP = "|"   # separator used in multi-select columns

MULTI_COLS = [
    'Q7_Platforms_Used','Q9_Key_Purchase_Factor','Q11_Product_Categories',
    'Q12_Colour_Preferences','Q13_Occasions','Q16_Bundle_Preferences','Q24_Market_Gap_Perception'
]


@st.cache_data(show_spinner=False)
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _clean(df)
    df = _engineer(df)
    return df


def load_raw(path: str) -> pd.DataFrame:
    """Load without caching — for uploaded new files."""
    df = pd.read_csv(path)
    df = _clean(df)
    df = _engineer(df)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop confirmed outliers for analysis (keep flag column)
    df['Is_Outlier'] = df['Is_Outlier'].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)

    # Strip whitespace from string cols
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # Target — fill missing with mode, encode
    df['Q25_Purchase_Intent_Label'] = df['Q25_Purchase_Intent_Label'].fillna(df['Q25_Purchase_Intent_Label'].mode()[0])
    df['Target'] = df['Q25_Purchase_Intent_Label'].map(LABEL_MAP).fillna(1).astype(int)

    # Categorical fills
    for col in ['Q1_Age_Group','Q2_Gender','Q3_City_Tier','Q4_Occupation',
                'Q5_Monthly_Income','Q6_Shopping_Frequency','Q8_Avg_Order_Spend',
                'Q14_Style_Personality','Q15_Ethnic_Wear','Q18_Social_Platform',
                'Q19_D2C_Purchase_Hist','Q20_Sustainability_Pay','Persona_Label']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if df[col].notna().any() else 'Unknown')

    # Numeric fills
    df['Q17_Price_Sensitivity']  = df['Q17_Price_Sensitivity'].fillna(df['Q17_Price_Sensitivity'].median())
    df['Q21_Budget_Numeric_INR'] = df['Q21_Budget_Numeric_INR'].fillna(df['Q21_Budget_Numeric_INR'].median())
    df['Q22_Digital_Trust_Score']= df['Q22_Digital_Trust_Score'].fillna(df['Q22_Digital_Trust_Score'].median())

    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal numeric encodings
    df['Age_Num']    = pd.Categorical(df['Q1_Age_Group'],  categories=AGE_ORDER,    ordered=True).codes
    df['Income_Num'] = pd.Categorical(df['Q5_Monthly_Income'], categories=INCOME_ORDER, ordered=True).codes
    df['Freq_Num']   = pd.Categorical(df['Q6_Shopping_Frequency'], categories=FREQ_ORDER, ordered=True).codes
    df['Spend_Num']  = pd.Categorical(df['Q8_Avg_Order_Spend'],    categories=SPEND_ORDER, ordered=True).codes
    df['City_Num']   = pd.Categorical(df['Q3_City_Tier'], categories=CITY_ORDER, ordered=True).codes
    df['Intent_Num'] = pd.Categorical(df['Q25_Purchase_Intent_Raw'], categories=INTENT_ORDER, ordered=True).codes

    df['Income_INR'] = df['Q5_Monthly_Income'].map(INCOME_NUMERIC).fillna(df['Q21_Budget_Numeric_INR'].median())
    df['Spend_INR']  = df['Q8_Avg_Order_Spend'].map(SPEND_NUMERIC).fillna(750)

    # Gender binary
    df['Gender_Bin'] = (df['Q2_Gender'] == 'Female').astype(int)

    # Multi-value binary flags — most common items across all multi cols
    _expand_multi(df, 'Q7_Platforms_Used',       ['Myntra','Ajio','Amazon','Flipkart','Meesho','Instagram/Social','Brand website'])
    _expand_multi(df, 'Q9_Key_Purchase_Factor',   ['Price/Discounts','Fabric quality','Trendy designs','Brand trust','Easy returns','Fast delivery'])
    _expand_multi(df, 'Q11_Product_Categories',   ['Jeans/Denim','T-shirts','Tops/Blouses','Dresses','Co-ord sets','Jackets/Blazers','Ethnic-Western Fusion','Activewear','Shorts/Skirts'])
    _expand_multi(df, 'Q13_Occasions',            ['Office/Work','Casual outings','Dates/Parties','College/Campus','Festivals','Travel','Gym/Workout','Work-from-home'])
    _expand_multi(df, 'Q24_Market_Gap_Perception',['Poor size range','Low fabric quality','High prices','Limited ethnic-fusion','Designs not India-appropriate','Poor returns/service'])

    # Composite scores
    df['Engagement_Score'] = (
        df['Freq_Num'] * 0.4 +
        df['Q22_Digital_Trust_Score'].fillna(3) * 0.3 +
        df['Q17_Price_Sensitivity'].fillna(3) * 0.1 +   # lower sens = more likely to buy
        df['Spend_Num'] * 0.2
    ).round(2)

    df['Value_Score'] = (
        df['Income_Num'] * 0.4 +
        df['Spend_Num'] * 0.4 +
        df['Q21_Budget_Numeric_INR'].fillna(2000) / 25000 * 5 * 0.2
    ).round(2)

    # D2C familiarity
    d2c_map = {'Yes': 2, 'No': 0, 'Not aware': -1}
    df['D2C_Score'] = df['Q19_D2C_Purchase_Hist'].map(d2c_map).fillna(0)

    # Influencer receptivity
    inf_map = {'Yes – many times': 3, 'Yes – once or twice': 2,
               'No – but tempted': 1, 'No – never': 0}
    df['Influencer_Score'] = df['Q23_Influencer_Purchase'].map(inf_map).fillna(1)

    # Sustainability willingness
    sus_map = {'Definitely yes': 3, 'Maybe – if quality is great': 2,
               'Haven\'t thought': 1, 'No – price matters': 0}
    df['Sustainability_Score'] = df['Q20_Sustainability_Pay'].map(sus_map).fillna(1)

    # High-value customer flag
    df['Is_High_Value'] = ((df['Income_Num'] >= 3) & (df['Spend_Num'] >= 3)).astype(int)

    # Priority segment (our focus: Interested + high engagement)
    df['Is_Target_Segment'] = (
        (df['Target'] == 2) &
        (df['Freq_Num'] >= 2) &
        (df['Age_Num'].between(1, 4))
    ).astype(int)

    return df


def _expand_multi(df: pd.DataFrame, col: str, items: list):
    """Create binary flag columns for each item in a pipe-separated multi-select column."""
    for item in items:
        safe_name = col + '_' + item.replace('/', '_').replace(' ', '_').replace('-', '_')
        df[safe_name] = df[col].fillna('').str.contains(item, regex=False).astype(int)


def get_model_features(df: pd.DataFrame) -> list:
    base = ['Age_Num','Gender_Bin','City_Num','Income_Num','Freq_Num','Spend_Num',
            'Q17_Price_Sensitivity','Q21_Budget_Numeric_INR','Q22_Digital_Trust_Score',
            'D2C_Score','Influencer_Score','Sustainability_Score',
            'Engagement_Score','Value_Score']
    binary_flags = [c for c in df.columns if any(
        c.startswith(m) for m in ['Q7_Platforms_Used_','Q9_Key_Purchase_Factor_',
                                   'Q11_Product_Categories_','Q13_Occasions_','Q24_Market_Gap_'])]
    return base + binary_flags


def encode_new_row(row_dict: dict) -> pd.DataFrame:
    """Convert a single new-customer form submission into a model-ready DataFrame."""
    df = pd.DataFrame([row_dict])
    df = _clean(df)
    df = _engineer(df)
    return df
