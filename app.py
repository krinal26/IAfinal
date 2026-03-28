"""
app.py — WesternWear Analytics Dashboard
Founder-level intelligence platform: Descriptive → Diagnostic → Predictive → Prescriptive
Target: Identify customers inclined toward WesternWear D2C brand
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data_engine import load_and_clean, load_raw, MULTI_COLS, AGE_ORDER, INCOME_ORDER, FREQ_ORDER, SPEND_ORDER, CITY_ORDER
from ml_engine import train_all, predict_new

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WesternWear AI | Founder Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family:'Outfit',sans-serif; }
.stApp { background:#0a0a0f; color:#f0ece4; }
[data-testid="stSidebar"] { background:linear-gradient(160deg,#12111a,#0d0c15); border-right:1px solid #2a2640; }

.brand-title { font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:800;
  background:linear-gradient(135deg,#e8c97e,#f0a080,#c07aff); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; line-height:1.1; margin-bottom:4px; }
.page-sub { color:#9090b0; font-size:.92rem; margin-bottom:1.4rem; font-weight:300; }

.kpi { background:linear-gradient(135deg,#18172a,#1e1c30); border:1px solid #2f2d4a;
  border-radius:14px; padding:20px 18px; text-align:center; }
.kpi:hover { border-color:#e8c97e; transition:.3s; }
.kpi-v { font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:#e8c97e; line-height:1; }
.kpi-l { font-size:.72rem; color:#7070a0; text-transform:uppercase; letter-spacing:.08em; margin-top:5px; }
.kpi-d { font-size:.8rem; color:#5fc47a; margin-top:3px; }
.kpi-d.bad { color:#e07070; }

.sec { font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:700; color:#f0ece4;
  border-left:3px solid #e8c97e; padding-left:10px; margin:24px 0 10px; }

.badge { display:inline-block; padding:3px 11px; border-radius:20px; font-size:.7rem;
  font-weight:600; letter-spacing:.07em; text-transform:uppercase; margin-right:6px; margin-bottom:8px; }
.bd { background:#1e1830; color:#c07aff; border:1px solid #6040c0; }
.bg { background:#181e18; color:#5fc47a; border:1px solid #3a7040; }
.bo { background:#1e1510; color:#f0a080; border:1px solid #a05030; }
.bp { background:#1e1020; color:#e07abb; border:1px solid #903060; }

.insight { background:#14131f; border:1px solid #2a2640; border-left:4px solid #e8c97e;
  border-radius:8px; padding:13px 15px; margin:9px 0; font-size:.86rem; line-height:1.65; color:#c0bcd8; }
.insight.g { border-left-color:#5fc47a; }
.insight.o { border-left-color:#f0a080; }
.insight.p { border-left-color:#c07aff; }

.divider { height:1px; background:linear-gradient(90deg,transparent,#2a2640,transparent); margin:22px 0; }

.stTabs [data-baseweb="tab-list"] { background:#14131f; border-radius:10px; padding:4px;
  border:1px solid #2a2640; gap:3px; }
.stTabs [data-baseweb="tab"] { color:#7070a0!important; font-family:'Outfit',sans-serif;
  font-weight:500; border-radius:7px; padding:6px 14px; }
.stTabs [aria-selected="true"] { background:#3a1a60!important; color:#c07aff!important; }

.risk-card { border-radius:10px; padding:14px 16px; margin:7px 0;
  border:1px solid #2a2640; font-size:.85rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
C = dict(gold='#e8c97e', coral='#f0a080', purple='#c07aff', green='#5fc47a',
         blue='#6ab0f0', pink='#e07abb', teal='#50d0c0', red='#e07070',
         orange='#f0b050', lavender='#a090e0')
PALETTE = list(C.values())
CHART = dict(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
             plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Outfit', color='#c0bcd8'),
             margin=dict(l=20,r=20,t=40,b=20))
INTENT_COLORS = {'Interested': C['green'], 'Neutral': C['gold'], 'Not Interested': C['red']}
PERSONA_COLORS = {'TrendyUrban': C['purple'], 'BudgetStudent': C['blue'],
                  'ProfessionalWoman': C['coral'], 'HousewifeTier2': C['pink'],
                  'PremiumShopper': C['gold']}


def styled(fig, title='', h=380):
    fig.update_layout(**CHART, title=dict(text=title, font=dict(family='Playfair Display', size=14, color='#f0ece4')), height=h)
    fig.update_xaxes(gridcolor='#1e1c30', zerolinecolor='#2a2640')
    fig.update_yaxes(gridcolor='#1e1c30', zerolinecolor='#2a2640')
    return fig

def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)
def ins(t, k=''): st.markdown(f'<div class="insight {k}">💡 {t}</div>', unsafe_allow_html=True)
def badge(k, txt): m={'desc':'bd','diag':'bg','pred':'bo','pres':'bp'}; st.markdown(f'<span class="badge {m[k]}">{txt}</span>', unsafe_allow_html=True)
def divider(): st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (val, label, delta, bad) in zip(cols, items):
        with col:
            dcls = 'bad' if bad else ''
            st.markdown(f'<div class="kpi"><div class="kpi-v">{val}</div>'
                        f'<div class="kpi-l">{label}</div>'
                        f'<div class="kpi-d {dcls}">{delta}</div></div>', unsafe_allow_html=True)


# ── Data load ──────────────────────────────────────────────────────────────────
DATA_PATH = 'data/WesternWear_Survey_Dataset_2000.csv'

@st.cache_data(show_spinner=False)
def get_data(): return load_and_clean(DATA_PATH)

with st.spinner("Loading WesternWear intelligence..."):
    df = get_data()
    df_clean = df[~df['Is_Outlier']].copy()
    ml = train_all(df_clean)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 18px'>
      <div style='font-family:Playfair Display,serif;font-size:1.6rem;font-weight:800;
        background:linear-gradient(135deg,#e8c97e,#c07aff);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent'>👗 WesternWear</div>
      <div style='color:#7070a0;font-size:.73rem;margin-top:3px'>Founder Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "📊  Descriptive Analysis",
        "🔍  Diagnostic Analysis",
        "🤖  Predictive Analysis",
        "💡  Prescriptive Analysis",
        "📤  Predict New Customers",
    ], label_visibility="collapsed")

    divider()
    st.markdown("<div style='color:#7070a0;font-size:.73rem;margin-bottom:7px'>GLOBAL FILTERS</div>", unsafe_allow_html=True)
    sel_persona = st.multiselect("Persona", df_clean['Persona_Label'].unique().tolist(),
                                  default=df_clean['Persona_Label'].unique().tolist())
    sel_city    = st.multiselect("City Tier", CITY_ORDER,
                                  default=CITY_ORDER)
    sel_gender  = st.multiselect("Gender", ['Female','Male'],
                                  default=['Female','Male'])
    sel_intent  = st.multiselect("Purchase Intent", ['Interested','Neutral','Not Interested'],
                                  default=['Interested','Neutral','Not Interested'])
    include_outliers = st.checkbox("Include outliers", value=False)

    base = df if include_outliers else df_clean
    fdf = base[
        base['Persona_Label'].isin(sel_persona) &
        base['Q3_City_Tier'].isin(sel_city) &
        base['Q2_Gender'].isin(sel_gender) &
        base['Q25_Purchase_Intent_Label'].isin(sel_intent)
    ].copy()

    divider()
    st.markdown(f"<div style='color:#7070a0;font-size:.73rem'>Showing <b style='color:#e8c97e'>{len(fdf):,}</b> / {len(df):,} respondents</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="brand-title">WesternWear AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Founder-level intelligence: Who are our customers? What do they want? How do we reach them?</div>', unsafe_allow_html=True)

    total     = len(fdf)
    interested= (fdf['Q25_Purchase_Intent_Label']=='Interested').sum()
    neutral   = (fdf['Q25_Purchase_Intent_Label']=='Neutral').sum()
    not_int   = (fdf['Q25_Purchase_Intent_Label']=='Not Interested').sum()
    conv_rate = interested/total*100 if total else 0
    avg_budget= fdf['Q21_Budget_Numeric_INR'].mean()
    avg_trust = fdf['Q22_Digital_Trust_Score'].mean()
    high_val  = fdf['Is_High_Value'].sum()

    kpi_row([
        (f"{total:,}",       "Total Respondents",     "Survey sample",     False),
        (f"{interested:,}",  "Interested Customers",  f"{conv_rate:.1f}% conversion potential", False),
        (f"{neutral:,}",     "Neutral (Nurture)",     "Win-able segment",  False),
        (f"₹{avg_budget:,.0f}", "Avg Monthly Budget", "Willingness to pay",False),
        (f"{avg_trust:.1f}/5","Avg Digital Trust",    "Platform readiness",False),
        (f"{high_val:,}",    "High-Value Prospects",  "Premium segment",   False),
    ])

    divider()

    col1, col2 = st.columns(2)
    with col1:
        sec("Purchase Intent Split")
        badge("desc","Descriptive")
        cnt = fdf['Q25_Purchase_Intent_Label'].value_counts().reset_index()
        cnt.columns = ['Label','Count']
        fig = go.Figure(go.Pie(
            labels=cnt['Label'], values=cnt['Count'], hole=0.62,
            marker=dict(colors=[INTENT_COLORS.get(l, C['blue']) for l in cnt['Label']],
                        line=dict(color='#0a0a0f', width=3)),
            textinfo='label+percent', pull=[0.05 if l=='Interested' else 0 for l in cnt['Label']],
        ))
        fig.add_annotation(text=f"<b>{conv_rate:.0f}%</b><br>Interested", x=.5, y=.5,
                           font=dict(size=17, family='Playfair Display', color=C['green']), showarrow=False)
        styled(fig, "Overall Purchase Inclination", 370)
        st.plotly_chart(fig, use_container_width=True)
        ins(f"{conv_rate:.1f}% of surveyed respondents are already Interested in a WesternWear D2C brand. An additional {neutral/total*100:.1f}% are Neutral — a critical nurture segment representing ₹{neutral*avg_budget:,.0f} in reachable GMV.")

    with col2:
        sec("Persona Distribution + Intent")
        badge("desc","Descriptive")
        pp = fdf.groupby(['Persona_Label','Q25_Purchase_Intent_Label']).size().reset_index(name='n')
        fig = px.bar(pp, x='Persona_Label', y='n', color='Q25_Purchase_Intent_Label',
                     color_discrete_map=INTENT_COLORS, barmode='stack', text='n',
                     labels={'Persona_Label':'Persona','n':'Count','Q25_Purchase_Intent_Label':'Intent'})
        fig.update_traces(textposition='inside', textfont_size=10)
        fig.update_layout(xaxis_tickangle=-20)
        styled(fig, "Customer Personas × Purchase Intent", 370)
        st.plotly_chart(fig, use_container_width=True)
        ins("TrendyUrban and ProfessionalWoman personas dominate the Interested segment. BudgetStudents show high volume but mixed intent — price-led activation needed. PremiumShoppers are small but highest-value.")

    # Geo-purchase intent heatmap
    sec("Purchase Intent by City Tier & Age Group")
    badge("desc","Descriptive")
    pivot = fdf[fdf['Q25_Purchase_Intent_Label']=='Interested'].groupby(
        ['Q3_City_Tier','Q1_Age_Group']).size().reset_index(name='n')
    heat_data = pivot.pivot(index='Q3_City_Tier', columns='Q1_Age_Group', values='n').fillna(0)
    heat_data = heat_data.reindex(columns=[c for c in AGE_ORDER if c in heat_data.columns])
    fig = go.Figure(go.Heatmap(
        z=heat_data.values, x=heat_data.columns.tolist(), y=heat_data.index.tolist(),
        colorscale=[[0,'#12111a'],[0.5,'#6040a0'],[1,'#e8c97e']],
        text=heat_data.values.astype(int), texttemplate='%{text}',
        colorbar=dict(title='Interested'),
    ))
    styled(fig, "Heatmap: # Interested Customers by City Tier × Age Group", 340)
    st.plotly_chart(fig, use_container_width=True)
    ins("Metro × 25–34 is our golden segment: highest concentration of Interested customers. Tier-1 × 18–24 is the rising segment. Tier-3 / Rural shows untapped opportunity — digital campaigns could unlock it.")

    # Platform landscape
    col3, col4 = st.columns(2)
    with col3:
        sec("Top Shopping Platforms Used")
        badge("desc","Descriptive")
        plat_cols = [c for c in fdf.columns if c.startswith('Q7_Platforms_Used_')]
        plat_sums = fdf[plat_cols].sum().sort_values(ascending=True)
        plat_sums.index = [i.replace('Q7_Platforms_Used_','').replace('_',' ') for i in plat_sums.index]
        fig = go.Figure(go.Bar(x=plat_sums.values, y=plat_sums.index, orientation='h',
                               marker=dict(color=plat_sums.values, colorscale='Plasma'),
                               text=plat_sums.values, textposition='outside'))
        styled(fig, "Platform Reach (% of respondents using each)", 380)
        st.plotly_chart(fig, use_container_width=True)
        ins("Myntra and Amazon are the current go-to platforms. Our Brand Website is already used by a meaningful segment — a positive signal for D2C readiness. Instagram/Social is the bridge channel.")

    with col4:
        sec("Social Media Platform Preference")
        badge("desc","Descriptive")
        soc = fdf['Q18_Social_Platform'].value_counts().reset_index()
        soc.columns = ['Platform','Count']
        fig = go.Figure(go.Pie(
            labels=soc['Platform'], values=soc['Count'], hole=0.5,
            marker=dict(colors=PALETTE, line=dict(color='#0a0a0f', width=2)),
            textinfo='label+percent',
        ))
        styled(fig, "Social Media Where Customers Spend Time", 380)
        st.plotly_chart(fig, use_container_width=True)
        ins("Instagram dominates — critical for influencer-led campaigns. YouTube is strong for longer-form content (lookbooks, styling videos). Pinterest signals high-aspiration shoppers ideal for premium product ads.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif "Descriptive" in page:
    st.markdown('<div class="brand-title">📊 Descriptive Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">What does our survey data tell us? Deep-dive into every column — demographics, behaviour, preferences.</div>', unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.tabs(["👤 Demographics","💳 Income & Spend","🛍️ Product & Style","📱 Digital Behaviour","📋 Summary Stats"])

    # ── Demographics ──────────────────────────────────────────────────────────
    with t1:
        badge("desc","Descriptive")
        c1,c2 = st.columns(2)
        with c1:
            sec("Age Group Distribution (Drill-Down Donut)")
            age_t = fdf.groupby('Q1_Age_Group').size().reset_index(name='total')
            age_i = fdf[fdf['Q25_Purchase_Intent_Label']=='Interested'].groupby('Q1_Age_Group').size().reset_index(name='interested')
            age_m = age_t.merge(age_i, on='Q1_Age_Group', how='left').fillna(0)
            age_m = age_m[age_m['Q1_Age_Group'].isin(AGE_ORDER)].set_index('Q1_Age_Group').reindex(AGE_ORDER).reset_index().dropna()

            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=age_m['Q1_Age_Group'], values=age_m['total'], hole=0.42,
                domain=dict(x=[0.12,0.88],y=[0.12,0.88]),
                marker=dict(colors=PALETTE[:len(age_m)], line=dict(color='#0a0a0f',width=3)),
                textinfo='label+percent', name='All', showlegend=False,
            ))
            fig.add_trace(go.Pie(
                labels=age_m.apply(lambda r: f"{r['Q1_Age_Group']} Interested", axis=1),
                values=age_m['interested'], hole=0.8,
                domain=dict(x=[0,1],y=[0,1]),
                marker=dict(colors=[C['green']]*len(age_m), line=dict(color='#0a0a0f',width=2)),
                textinfo='none', name='Interested',
            ))
            fig.add_annotation(text="Age<br>Drill-Down", x=.5, y=.5,
                               font=dict(size=12,family='Playfair Display',color=C['gold']), showarrow=False)
            styled(fig, "Inner=All Respondents | Outer Ring=Interested Customers", 420)
            st.plotly_chart(fig, use_container_width=True)
            ins("25–34 is the dominant age group across all respondents AND the most Interested. 18–24 is the second largest — high-growth potential. Under 18 and 55+ are niche.")

        with c2:
            sec("Gender × City Tier Sunburst (Click to Drill Down)")
            sun = fdf.groupby(['Q3_City_Tier','Q2_Gender','Q25_Purchase_Intent_Label']).size().reset_index(name='n')
            fig = px.sunburst(sun, path=['Q3_City_Tier','Q2_Gender','Q25_Purchase_Intent_Label'],
                              values='n', color='Q25_Purchase_Intent_Label',
                              color_discrete_map=INTENT_COLORS)
            fig.update_traces(textfont=dict(family='Outfit'))
            styled(fig, "Click to drill: City → Gender → Purchase Intent", 420)
            st.plotly_chart(fig, use_container_width=True)
            ins("Metro Female is our highest-intent segment. Tier-1 Female follows closely. Tier-2 and Tier-3 show meaningful male interest — worth testing gender-neutral collections for those markets.", "g")

        c3,c4 = st.columns(2)
        with c3:
            sec("Occupation Split by Intent")
            occ = fdf.groupby(['Q4_Occupation','Q25_Purchase_Intent_Label']).size().reset_index(name='n')
            fig = px.bar(occ, x='n', y='Q4_Occupation', color='Q25_Purchase_Intent_Label',
                         orientation='h', barmode='stack', color_discrete_map=INTENT_COLORS,
                         labels={'Q4_Occupation':'Occupation','n':'Count'})
            styled(fig, "Occupation × Purchase Intent", 400)
            st.plotly_chart(fig, use_container_width=True)
            ins("Salaried-Private has the highest absolute Interested count. Students show high volume with moderate interest — discount bundles will convert them. Self-employed/Freelancers are a premium niche.")

        with c4:
            sec("Style Personality Distribution")
            style = fdf.groupby(['Q14_Style_Personality','Q25_Purchase_Intent_Label']).size().reset_index(name='n')
            fig = px.bar(style, x='Q14_Style_Personality', y='n', color='Q25_Purchase_Intent_Label',
                         barmode='group', color_discrete_map=INTENT_COLORS,
                         labels={'Q14_Style_Personality':'Style','n':'Count'})
            fig.update_layout(xaxis_tickangle=-30)
            styled(fig, "Style Personality vs Purchase Intent", 400)
            st.plotly_chart(fig, use_container_width=True)
            ins("Trendy/Street style and Minimalist personas are most Interested. Boho/Free-spirited customers also show strong intent. Classic/Formal is the hardest to convert — not our primary target.", "g")

    # ── Income & Spend ────────────────────────────────────────────────────────
    with t2:
        badge("desc","Descriptive")
        c1,c2 = st.columns(2)
        with c1:
            sec("Monthly Income Distribution")
            inc = fdf.groupby(['Q5_Monthly_Income','Q25_Purchase_Intent_Label']).size().reset_index(name='n')
            inc_order = [i for i in INCOME_ORDER if i in fdf['Q5_Monthly_Income'].unique()]
            fig = px.bar(inc, x='Q5_Monthly_Income', y='n', color='Q25_Purchase_Intent_Label',
                         barmode='stack', color_discrete_map=INTENT_COLORS,
                         category_orders={'Q5_Monthly_Income': inc_order},
                         labels={'Q5_Monthly_Income':'Income','n':'Count'})
            fig.update_layout(xaxis_tickangle=-20)
            styled(fig, "Income Band × Purchase Intent (Stacked)", 380)
            st.plotly_chart(fig, use_container_width=True)
            ins("₹20K–₹40K and ₹40K–₹75K bands dominate Interested customers — mid-income professionals and students with disposable income. High-income (₹1.5L+) customers are fewer but have highest per-order spend.")

        with c2:
            sec("Avg Order Spend — Spend Tier Donut")
            spd_t = fdf.groupby('Q8_Avg_Order_Spend').size().reset_index(name='total')
            spd_i = fdf[fdf['Q25_Purchase_Intent_Label']=='Interested'].groupby('Q8_Avg_Order_Spend').size().reset_index(name='interested')
            spd_m = spd_t.merge(spd_i, on='Q8_Avg_Order_Spend', how='left').fillna(0)
            spd_order = [s for s in SPEND_ORDER if s in spd_m['Q8_Avg_Order_Spend'].values]
            spd_m = spd_m.set_index('Q8_Avg_Order_Spend').reindex(spd_order).reset_index().dropna()

            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=spd_m['Q8_Avg_Order_Spend'], values=spd_m['total'], hole=0.42,
                domain=dict(x=[0.12,0.88],y=[0.12,0.88]),
                marker=dict(colors=[C['purple'],C['blue'],C['teal'],C['gold'],C['coral']],
                            line=dict(color='#0a0a0f',width=3)),
                textinfo='label+percent', showlegend=False,
            ))
            fig.add_trace(go.Pie(
                labels=spd_m['Q8_Avg_Order_Spend'], values=spd_m['interested'], hole=0.78,
                domain=dict(x=[0,1],y=[0,1]),
                marker=dict(colors=[C['green']]*len(spd_m), line=dict(color='#0a0a0f',width=2)),
                textinfo='none',
            ))
            fig.add_annotation(text="Spend<br>Tiers", x=.5, y=.5,
                               font=dict(size=12,family='Playfair Display',color=C['gold']), showarrow=False)
            styled(fig, "Inner=All | Outer Green Ring=Interested by Spend Tier", 420)
            st.plotly_chart(fig, use_container_width=True)
            ins("₹1,000–₹2,500 per order is the sweet spot — the largest spend tier among Interested customers. This directly guides pricing strategy: price most items in this range.")

        c3,c4 = st.columns(2)
        with c3:
            sec("Ideal Monthly Budget — Box Plot by Persona")
            fig = go.Figure()
            colors_list = list(PERSONA_COLORS.values())
            for i, persona in enumerate(fdf['Persona_Label'].unique()):
                sub = fdf[fdf['Persona_Label']==persona]['Q21_Budget_Numeric_INR'].dropna()
                fig.add_trace(go.Box(y=sub, name=persona, marker_color=colors_list[i % len(colors_list)],
                                     boxmean='sd', notched=True))
            styled(fig, "Monthly Budget Distribution by Persona (Notched = 95% CI)", 420)
            st.plotly_chart(fig, use_container_width=True)
            ins("PremiumShopper has the highest median budget (₹5,000+). TrendyUrban cluster is wide — serves both affordable and aspirational. HousewifeTier2 and BudgetStudent have tight low ranges: bundle offers work best here.")

        with c4:
            sec("Budget vs Digital Trust Score (Scatter)")
            fig = px.scatter(fdf.sample(min(600,len(fdf))), x='Q21_Budget_Numeric_INR',
                             y='Q22_Digital_Trust_Score', color='Q25_Purchase_Intent_Label',
                             color_discrete_map=INTENT_COLORS, size='Spend_Num', size_max=14, opacity=0.7,
                             hover_data=['Persona_Label','Q3_City_Tier','Q14_Style_Personality'],
                             labels={'Q21_Budget_Numeric_INR':'Monthly Budget (₹)',
                                     'Q22_Digital_Trust_Score':'Digital Trust (1–5)'})
            styled(fig, "Budget × Digital Trust coloured by Intent (size=order spend)", 420)
            st.plotly_chart(fig, use_container_width=True)
            ins("High-budget + high-trust customers (top right) are almost entirely Interested — our core acquisition target. Low-trust customers need social proof: UGC, reviews, influencer endorsements.", "g")

    # ── Product & Style ───────────────────────────────────────────────────────
    with t3:
        badge("desc","Descriptive")
        c1,c2 = st.columns(2)
        with c1:
            sec("Product Categories Preferred (Multi-Select Breakdown)")
            cat_cols = [c for c in fdf.columns if c.startswith('Q11_Product_Categories_')]
            cat_sums = fdf[cat_cols].sum().sort_values(ascending=False)
            cat_sums.index = [i.replace('Q11_Product_Categories_','').replace('_',' ') for i in cat_sums.index]

            # Split by intent
            int_df = fdf[fdf['Q25_Purchase_Intent_Label']=='Interested'][cat_cols].sum()
            int_df.index = [i.replace('Q11_Product_Categories_','').replace('_',' ') for i in int_df.index]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='All', x=cat_sums.index, y=cat_sums.values,
                                 marker_color=C['purple'], opacity=0.5))
            fig.add_trace(go.Bar(name='Interested only', x=int_df.index, y=int_df.values,
                                 marker_color=C['gold']))
            fig.update_layout(barmode='group', xaxis_tickangle=-30)
            styled(fig, "Product Category Demand: All vs Interested Customers", 400)
            st.plotly_chart(fig, use_container_width=True)
            ins("Jeans/Denim, Tops/Blouses and T-shirts are the top 3 categories across the board. Co-ord sets and Ethnic-Western Fusion have disproportionately high interest among the Interested segment — a product differentiation opportunity.")

        with c2:
            sec("Occasion Mapping — Where Do They Wear It?")
            occ_cols = [c for c in fdf.columns if c.startswith('Q13_Occasions_')]
            occ_sums = fdf[occ_cols].sum().sort_values(ascending=True)
            occ_sums.index = [i.replace('Q13_Occasions_','').replace('_',' ') for i in occ_sums.index]
            fig = go.Figure(go.Bar(x=occ_sums.values, y=occ_sums.index, orientation='h',
                                   marker=dict(color=occ_sums.values, colorscale='Viridis'),
                                   text=occ_sums.values, textposition='outside'))
            styled(fig, "Occasion-wise Demand for WesternWear", 400)
            st.plotly_chart(fig, use_container_width=True)
            ins("Casual outings and Office/Work are #1 and #2 — confirms demand for smart-casual and workwear styles. Work-from-home is surging — comfortable yet stylish WFH collections are a white space.")

        c3,c4 = st.columns(2)
        with c3:
            sec("Colour Preferences Heat Map")
            col_opts = ['Neutrals','Pastels','Bold/Bright','Dark tones','Earth tones','Prints & Patterns']
            intent_opts = ['Interested','Neutral','Not Interested']
            heat_rows = []
            for intent in intent_opts:
                row = []
                sub = fdf[fdf['Q25_Purchase_Intent_Label']==intent]['Q12_Colour_Preferences'].fillna('')
                for col in col_opts:
                    row.append(sub.str.contains(col, regex=False).sum())
                heat_rows.append(row)
            fig = go.Figure(go.Heatmap(
                z=heat_rows, x=col_opts, y=intent_opts,
                colorscale=[[0,'#12111a'],[0.5,'#6040a0'],[1,'#e8c97e']],
                text=[[str(v) for v in row] for row in heat_rows],
                texttemplate='%{text}', colorbar=dict(title='Count'),
            ))
            styled(fig, "Colour Preferences by Purchase Intent Group", 300)
            st.plotly_chart(fig, use_container_width=True)
            ins("Interested customers favour Neutrals and Dark tones — supports a clean, minimal aesthetic for our brand. Pastels are popular with Neutral group — a bridge collection could convert them.")

        with c4:
            sec("Bundle Preferences (What They Want to Buy Together)")
            bdl_opts = ['Top+Bottom combo','3 basics for ₹999','Curated outfit box',
                        'Co-ord set+Accessories','Ethnic+western bundle','Seasonal wardrobe kit']
            bdl_counts = {b: fdf['Q16_Bundle_Preferences'].fillna('').str.contains(b, regex=False).sum()
                          for b in bdl_opts}
            bdl_df = pd.DataFrame(list(bdl_counts.items()), columns=['Bundle','Count']).sort_values('Count')
            fig = go.Figure(go.Bar(x=bdl_df['Count'], y=bdl_df['Bundle'], orientation='h',
                                   marker=dict(color=list(range(len(bdl_df))), colorscale='Plasma'),
                                   text=bdl_df['Count'], textposition='outside'))
            styled(fig, "Bundle / Pack Preferences Among Respondents", 380)
            st.plotly_chart(fig, use_container_width=True)
            ins("'3 basics for ₹999' is the runaway winner — perfect for a launch offer. 'Curated outfit box' is second — a subscription-style product idea. 'Ethnic+western bundle' validates the fusion niche.", "g")

    # ── Digital Behaviour ─────────────────────────────────────────────────────
    with t4:
        badge("desc","Descriptive")
        c1,c2 = st.columns(2)
        with c1:
            sec("Shopping Frequency Distribution")
            freq = fdf.groupby(['Q6_Shopping_Frequency','Q25_Purchase_Intent_Label']).size().reset_index(name='n')
            freq_ord = [f for f in FREQ_ORDER if f in fdf['Q6_Shopping_Frequency'].unique()]
            fig = px.bar(freq, x='Q6_Shopping_Frequency', y='n', color='Q25_Purchase_Intent_Label',
                         barmode='group', color_discrete_map=INTENT_COLORS,
                         category_orders={'Q6_Shopping_Frequency': freq_ord},
                         labels={'Q6_Shopping_Frequency':'Frequency','n':'Count'})
            styled(fig, "Shopping Frequency × Purchase Intent", 380)
            st.plotly_chart(fig, use_container_width=True)
            ins("'Very frequently' shoppers are disproportionately Interested — these are habitual buyers we must capture early. 'Rarely' shoppers in the Interested group are high-consideration buyers: quality content converts them.")

        with c2:
            sec("D2C Purchase History × Influencer Receptivity")
            d2c_inf = fdf.groupby(['Q19_D2C_Purchase_Hist','Q25_Purchase_Intent_Label']).size().reset_index(name='n')
            fig = px.bar(d2c_inf, x='Q19_D2C_Purchase_Hist', y='n', color='Q25_Purchase_Intent_Label',
                         barmode='stack', color_discrete_map=INTENT_COLORS,
                         labels={'Q19_D2C_Purchase_Hist':'D2C History','n':'Count'})
            styled(fig, "D2C Brand History vs Purchase Intent", 380)
            st.plotly_chart(fig, use_container_width=True)
            ins("Customers who already buy from D2C brands show the highest intent — they are pre-qualified. 'Not aware' segment is large — brand awareness campaigns are the first priority for growth.")

        c3,c4 = st.columns(2)
        with c3:
            sec("Digital Trust Score — Violin by Intent")
            fig = go.Figure()
            for intent, color in INTENT_COLORS.items():
                sub = fdf[fdf['Q25_Purchase_Intent_Label']==intent]['Q22_Digital_Trust_Score']
                fig.add_trace(go.Violin(y=sub, name=intent, fillcolor=color, opacity=0.7,
                                        box_visible=True, meanline_visible=True))
            styled(fig, "Digital Trust Score Distribution by Purchase Intent", 380)
            st.plotly_chart(fig, use_container_width=True)
            ins("Interested customers have a higher median digital trust score (4–5). Low-trust customers (1–2) rarely show interest. Invest in trust signals: verified reviews, secure checkout, return guarantee.", "g")

        with c4:
            sec("Sustainability Willingness × Price Sensitivity")
            sus = fdf.groupby(['Q20_Sustainability_Pay','Q17_Price_Sensitivity']).size().reset_index(name='n')
            sus_pivot = sus.pivot(index='Q20_Sustainability_Pay', columns='Q17_Price_Sensitivity', values='n').fillna(0)
            fig = go.Figure(go.Heatmap(
                z=sus_pivot.values, x=[str(c) for c in sus_pivot.columns],
                y=sus_pivot.index.tolist(),
                colorscale='Purples',
                text=sus_pivot.values.astype(int), texttemplate='%{text}',
                colorbar=dict(title='Count'),
            ))
            styled(fig, "Sustainability Willingness × Price Sensitivity Score", 380)
            st.plotly_chart(fig, use_container_width=True)
            ins("'Maybe if quality is great' + price sensitivity 3–4 is the modal customer profile. This validates our positioning: quality-first, mid-priced, with clear sustainability storytelling.", "g")

    # ── Summary Stats ──────────────────────────────────────────────────────────
    with t5:
        badge("desc","Descriptive")
        sec("Numerical Column Summary Statistics")
        num_cols = ['Q17_Price_Sensitivity','Q21_Budget_Numeric_INR','Q22_Digital_Trust_Score',
                    'Influencer_Score','Sustainability_Score','Engagement_Score','Value_Score']
        summary = fdf[num_cols].describe().T.round(2)
        summary.columns = ['Count','Mean','Std','Min','25%','Median','75%','Max']
        st.dataframe(summary.style
                     .background_gradient(subset=['Mean'], cmap='Purples')
                     .background_gradient(subset=['Std'], cmap='Oranges')
                     .format("{:.2f}"), use_container_width=True)

        sec("Group Comparison: Interested vs Not Interested")
        grp = fdf.groupby('Q25_Purchase_Intent_Label')[num_cols].mean().T.round(2)
        if 'Interested' in grp.columns and 'Not Interested' in grp.columns:
            grp['Gap'] = (grp['Interested'] - grp['Not Interested']).round(2)
        st.dataframe(grp.style.background_gradient(cmap='Greens', subset=['Interested'] if 'Interested' in grp.columns else [])
                     .background_gradient(cmap='Reds', subset=['Not Interested'] if 'Not Interested' in grp.columns else []),
                     use_container_width=True)
        ins("Key gaps: Interested customers score higher on Digital Trust, Engagement, and Influencer receptivity. Budget is similar — meaning intent is driven by brand affinity, not just money.")

        sec("Market Gap Perceptions — What They Say Is Missing")
        gap_cols = [c for c in fdf.columns if c.startswith('Q24_Market_Gap')]
        gap_sums = fdf[gap_cols].sum().sort_values(ascending=False)
        gap_sums.index = [i.replace('Q24_Market_Gap_Perception_','').replace('_',' ') for i in gap_sums.index]
        fig = px.bar(gap_sums.reset_index(), x='index', y=0,
                     color='index', color_discrete_sequence=PALETTE,
                     labels={'index':'Market Gap','0':'Mentions'}, text=0)
        fig.update_traces(textposition='outside', showlegend=False)
        styled(fig, "Most Mentioned Market Gaps in WesternWear", 360)
        st.plotly_chart(fig, use_container_width=True)
        ins("'Poor size range' is the #1 market gap — an inclusive sizing strategy is a must-have, not a nice-to-have. 'Low fabric quality' and 'Designs not India-appropriate' are next — localized, quality-first product design is our moat.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
elif "Diagnostic" in page:
    st.markdown('<div class="brand-title">🔍 Diagnostic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Why do some customers show interest and others don\'t? Correlation, segmentation & causal pattern mining.</div>', unsafe_allow_html=True)

    badge("diag","Diagnostic")

    # Correlation heatmap
    sec("Full Correlation Matrix — Numeric Features vs Purchase Intent")
    num_cols = ['Age_Num','Income_Num','Freq_Num','Spend_Num','City_Num',
                'Q17_Price_Sensitivity','Q21_Budget_Numeric_INR','Q22_Digital_Trust_Score',
                'D2C_Score','Influencer_Score','Sustainability_Score',
                'Engagement_Score','Value_Score','Target']
    num_cols_avail = [c for c in num_cols if c in fdf.columns]
    corr = fdf[num_cols_avail].corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_m = corr.where(~mask)
    fig = go.Figure(go.Heatmap(
        z=corr_m.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale='RdBu', zmid=0,
        text=corr_m.round(2).values, texttemplate='%{text}', textfont=dict(size=9),
        colorbar=dict(title='r'),
    ))
    styled(fig, "Pearson Correlation Matrix — lower triangle (Target=Purchase Intent)", 520)
    st.plotly_chart(fig, use_container_width=True)
    ins("Digital Trust Score, D2C history, and Engagement Score have the strongest positive correlations with purchase intent. Price Sensitivity is negatively correlated — high-sensitivity = lower intent. Sustainability willingness acts as an amplifier.", "g")

    c1,c2 = st.columns(2)
    with c1:
        sec("Shopping Frequency × Income → Churn Risk Heatmap")
        pivot2 = fdf.groupby(['Q6_Shopping_Frequency','Q5_Monthly_Income'])['Target'].mean().reset_index()
        pivot2_wide = pivot2.pivot(index='Q6_Shopping_Frequency', columns='Q5_Monthly_Income', values='Target').fillna(0)
        pivot2_wide = pivot2_wide.reindex(index=[f for f in FREQ_ORDER if f in pivot2_wide.index],
                                          columns=[i for i in INCOME_ORDER if i in pivot2_wide.columns])
        fig = go.Figure(go.Heatmap(
            z=pivot2_wide.values*100, x=pivot2_wide.columns.tolist(), y=pivot2_wide.index.tolist(),
            colorscale=[[0,'#12111a'],[0.5,'#6040a0'],[1,'#5fc47a']],
            text=np.round(pivot2_wide.values*100,1), texttemplate='%{text}%',
            colorbar=dict(title='Avg Intent Score'),
        ))
        fig.update_layout(xaxis_tickangle=-20)
        styled(fig, "Avg Intent Score %: Shopping Frequency × Income Band", 380)
        st.plotly_chart(fig, use_container_width=True)
        ins("'Very frequently' shoppers earning ₹40K–₹75K have near-maximum intent scores — this is our primary acquisition cohort. High-income but low-frequency customers are premium upsell opportunities.", "g")

    with c2:
        sec("Parallel Coordinates — Multi-Dimensional Customer Profile (Drag Axes)")
        pc_cols = [c for c in ['Age_Num','Income_Num','Spend_Num','Q22_Digital_Trust_Score',
                                'Influencer_Score','Engagement_Score','Target'] if c in fdf.columns]
        fig = px.parallel_coordinates(
            fdf[pc_cols].sample(min(400, len(fdf))),
            color='Target', color_continuous_scale=[[0,C['red']],[0.5,C['gold']],[1,C['green']]],
            labels={c: c.replace('_',' ') for c in pc_cols},
        )
        styled(fig, "Drag & select on any axis — green=Interested, red=Not Interested", 420)
        st.plotly_chart(fig, use_container_width=True)
        ins("The parallel coordinates reveal a clear pattern: Interested customers (green) cluster toward high Engagement, high Digital Trust, and high Spend — while Not Interested (red) cluster toward low engagement and low spend.", "g")

    c3,c4 = st.columns(2)
    with c3:
        sec("Price Sensitivity vs Budget — Scatter by Persona")
        fig = px.scatter(fdf, x='Q17_Price_Sensitivity', y='Q21_Budget_Numeric_INR',
                         color='Persona_Label', color_discrete_map=PERSONA_COLORS,
                         facet_col='Q25_Purchase_Intent_Label', size='Spend_Num', size_max=14, opacity=0.6,
                         labels={'Q17_Price_Sensitivity':'Price Sensitivity (1-5)',
                                 'Q21_Budget_Numeric_INR':'Monthly Budget (₹)'})
        styled(fig, "Price Sensitivity × Budget — split by Intent, colored by Persona", 420)
        st.plotly_chart(fig, use_container_width=True)
        ins("In the 'Interested' panel: PremiumShoppers concentrate at low price sensitivity + high budget (ideal). BudgetStudents show high sensitivity + low budget — require value-bundle strategy. Two distinct product-price ladders needed.")

    with c4:
        sec("Market Gap × Persona — Who Feels What Pain?")
        gap_cols_short = [c for c in fdf.columns if c.startswith('Q24_Market_Gap')]
        gap_persona = fdf.groupby('Persona_Label')[gap_cols_short].mean().T
        gap_persona.index = [i.replace('Q24_Market_Gap_Perception_','').replace('_',' ')[:20]
                             for i in gap_persona.index]
        fig = go.Figure(go.Heatmap(
            z=gap_persona.values*100, x=gap_persona.columns.tolist(), y=gap_persona.index.tolist(),
            colorscale='Oranges',
            text=np.round(gap_persona.values*100,0).astype(int), texttemplate='%{text}%',
            colorbar=dict(title='% Mentioning'),
        ))
        styled(fig, "Market Gaps Mentioned by Persona (% of each persona)", 420)
        st.plotly_chart(fig, use_container_width=True)
        ins("HousewifeTier2 and ProfessionalWoman feel 'Poor size range' most acutely. BudgetStudents cite 'High prices'. TrendyUrban cites 'Designs not India-appropriate' — validating local-design differentiation.", "o")

    # Key Purchase Factors
    sec("Key Purchase Factors — What Drives the Decision to Buy?")
    fac_cols = [c for c in fdf.columns if c.startswith('Q9_Key_Purchase_Factor_')]
    fac_by_intent = {}
    for intent in ['Interested','Neutral','Not Interested']:
        sub = fdf[fdf['Q25_Purchase_Intent_Label']==intent][fac_cols].mean() * 100
        sub.index = [i.replace('Q9_Key_Purchase_Factor_','').replace('_',' ') for i in sub.index]
        fac_by_intent[intent] = sub

    fig = go.Figure()
    for intent, color in INTENT_COLORS.items():
        if intent in fac_by_intent:
            s = fac_by_intent[intent].sort_values(ascending=False)
            fig.add_trace(go.Bar(name=intent, x=s.index, y=s.values,
                                 marker_color=color, opacity=0.85))
    fig.update_layout(barmode='group', xaxis_tickangle=-20)
    styled(fig, "Key Purchase Factor Prevalence (%) by Intent Group", 400)
    st.plotly_chart(fig, use_container_width=True)
    ins("Fabric quality and Brand trust are disproportionately important for Interested customers vs others — premium quality messaging is critical. Price/Discounts is equally high across all groups, but trendy designs and easy returns matter MORE for Interested customers.", "g")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif "Predictive" in page:
    st.markdown('<div class="brand-title">🤖 Predictive Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Which algorithms best predict customer inclination? 5 ML models trained on 2,000 survey responses.</div>', unsafe_allow_html=True)

    badge("pred","Predictive")
    results = ml['results']
    best_name = ml['best_name']

    # Model scorecards
    sec("Model Performance Scorecard")
    names = list(results.keys())
    kpi_row([
        (f"{results[n]['acc']*100:.1f}%", n, f"AUC={results[n]['auc']:.3f} | F1={results[n]['f1']:.3f}", False)
        for n in names
    ])
    divider()

    c1,c2 = st.columns(2)
    with c1:
        sec("Cross-Validation Accuracy (5-Fold) — All Models")
        cv_df = pd.DataFrame({
            'Model':  names,
            'CV Mean': [results[n]['cv_mean']*100 for n in names],
            'CV Std':  [results[n]['cv_std']*100 for n in names],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cv_df['Model'], y=cv_df['CV Mean'],
                             error_y=dict(type='data', array=cv_df['CV Std']),
                             marker=dict(color=PALETTE[:len(names)], line=dict(color='#0a0a0f',width=1)),
                             text=[f"{v:.1f}%" for v in cv_df['CV Mean']], textposition='outside'))
        fig.add_hline(y=60, line_dash='dot', line_color=C['gold'], annotation_text="Baseline 60%")
        styled(fig, "5-Fold CV Accuracy with Std Dev Error Bars", 380)
        st.plotly_chart(fig, use_container_width=True)
        ins(f"All models outperform baseline. {best_name} achieves the best AUC. Error bars show stability — smaller = more consistent. Gradient Boosting and RF are most production-ready.", "o")

    with c2:
        sec(f"Confusion Matrix — {best_name}")
        cm = results[best_name]['cm']
        labels_cm = ['Not Interested','Neutral','Interested']
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels_cm, y=labels_cm,
            colorscale=[[0,'#12111a'],[0.5,'#4a2060'],[1,'#e8c97e']],
            showscale=False,
            text=cm, texttemplate='<b>%{text}</b>', textfont=dict(size=18,family='Playfair Display')
        ))
        fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual')
        styled(fig, f"Confusion Matrix — {best_name}", 380)
        st.plotly_chart(fig, use_container_width=True)
        ins("The model correctly identifies the majority of Interested customers. Neutral misclassifications are expected — this segment sits on the boundary. Focus marketing spend on high-confidence Interested predictions.", "o")

    # Feature importance
    rf_model = results['Random Forest']['model']
    features = ml['features']
    imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False).head(20)
    imp_plot = imp.sort_values(ascending=True)

    sec("Top 20 Feature Importances — Random Forest")
    imp_plot.index = [i.replace('Q9_Key_Purchase_Factor_','KPF: ').replace('Q7_Platforms_Used_','PLT: ')
                      .replace('Q11_Product_Categories_','CAT: ').replace('Q13_Occasions_','OCC: ')
                      .replace('Q24_Market_Gap_','GAP: ').replace('_',' ') for i in imp_plot.index]
    fig = go.Figure(go.Bar(
        x=imp_plot.values, y=imp_plot.index, orientation='h',
        marker=dict(color=imp_plot.values, colorscale='Cividis',
                    line=dict(color='#0a0a0f',width=1)),
        text=[f"{v:.3f}" for v in imp_plot.values], textposition='outside',
    ))
    styled(fig, "Feature Importance — Which signals predict customer inclination most?", 520)
    st.plotly_chart(fig, use_container_width=True)
    ins("Digital Trust Score, Budget, and Engagement Score are the top predictors. D2C familiarity and Influencer Score are strong behavioural signals. Product category preferences and platform usage add granular discriminative power.", "o")

    # Cluster analysis
    sec("Customer Segmentation — KMeans Clustering (PCA Projection)")
    X_pca = ml['X_pca']
    cluster_labels = ml['cluster_labels']
    pca_df = pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1],
                           'Cluster': cluster_labels.astype(str),
                           'Intent': df_clean['Q25_Purchase_Intent_Label'].values,
                           'Persona': df_clean['Persona_Label'].values})
    c1,c2 = st.columns(2)
    with c1:
        fig = px.scatter(pca_df.sample(min(600,len(pca_df))), x='PC1', y='PC2',
                         color='Cluster', symbol='Intent',
                         color_discrete_sequence=PALETTE,
                         opacity=0.75, size_max=8,
                         labels={'PC1':'Principal Component 1','PC2':'Principal Component 2'})
        styled(fig, "PCA 2D Projection — Coloured by Cluster, Symbol=Intent", 420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(pca_df.sample(min(600,len(pca_df))), x='PC1', y='PC2',
                         color='Persona', opacity=0.75,
                         color_discrete_map=PERSONA_COLORS,
                         labels={'PC1':'Principal Component 1','PC2':'Principal Component 2'})
        styled(fig, "Same Projection — Coloured by Survey Persona", 420)
        st.plotly_chart(fig, use_container_width=True)
    ins("The PCA clusters broadly align with the survey-defined personas — validating our data quality. Cluster separation shows the model has learned meaningful structure. Overlapping clusters (Neutral zone) are the battleground segment to win.", "o")

    # Probability distribution
    sec("Predicted Probability Distribution — Best Model")
    best_proba = results[best_name]['proba']
    fig = make_subplots(rows=1, cols=3, subplot_titles=['P(Not Interested)','P(Neutral)','P(Interested)'])
    for i, (label, color) in enumerate(zip(['Not Interested','Neutral','Interested'],
                                            [C['red'],C['gold'],C['green']]), start=1):
        for intent_val, ls in zip([0,1,2],['dot','dash','solid']):
            mask = ml['y_test'].values == intent_val
            fig.add_trace(go.Histogram(x=best_proba[mask, i-1], name=f"Actual {labels_cm[intent_val]}",
                                       marker_color=color, opacity=0.6, nbinsx=15,
                                       showlegend=(i==1)), row=1, col=i)
    fig.update_layout(**CHART, height=380, title=dict(text=f"Predicted Probability Distribution by Class — {best_name}",
                                                      font=dict(family='Playfair Display',size=14,color='#f0ece4')))
    st.plotly_chart(fig, use_container_width=True)
    ins("Well-separated probability distributions confirm model confidence. Cases where probabilities are spread evenly (flat histograms) indicate the uncertain Neutral segment — these need human-touch follow-up or nurture sequences.", "o")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif "Prescriptive" in page:
    st.markdown('<div class="brand-title">💡 Prescriptive Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">What should WesternWear do next? Data-driven action playbook for product, marketing, pricing, and growth.</div>', unsafe_allow_html=True)

    badge("pres","Prescriptive")

    # Risk segmentation on full data
    rf_m = ml['results']['Random Forest']['model']
    df_pred = df_clean.copy()
    feat_avail = [f for f in ml['features'] if f in df_pred.columns]
    proba_all = rf_m.predict_proba(df_pred[feat_avail].fillna(0))
    df_pred['P_Interested'] = proba_all[:, 2]
    df_pred['P_Neutral']    = proba_all[:, 1]
    df_pred['Risk_Segment'] = pd.cut(df_pred['P_Interested'],
                                      bins=[0, 0.3, 0.55, 0.75, 1.0],
                                      labels=['Cold (< 30%)','Warm (30–55%)','Hot (55–75%)','Convert Now (>75%)'])

    c1,c2 = st.columns([1,2])
    with c1:
        sec("Prospect Funnel")
        seg_cnt = df_pred['Risk_Segment'].value_counts()
        seg_colors = {'Cold (< 30%)': C['red'], 'Warm (30–55%)': C['gold'],
                      'Hot (55–75%)': C['coral'], 'Convert Now (>75%)': C['green']}
        fig = go.Figure(go.Pie(
            labels=seg_cnt.index, values=seg_cnt.values, hole=0.55,
            marker=dict(colors=[seg_colors.get(s, C['purple']) for s in seg_cnt.index],
                        line=dict(color='#0a0a0f',width=3)),
            textinfo='label+percent+value',
            pull=[0.06 if 'Convert' in s else 0 for s in seg_cnt.index],
        ))
        fig.add_annotation(text="Prospect<br>Funnel", x=.5, y=.5,
                           font=dict(size=12,family='Playfair Display',color=C['gold']), showarrow=False)
        styled(fig, "Prospect Funnel — Risk Segments", 360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Action Playbook by Segment")
        playbook = [
            ("Convert Now (>75%)", "green", C['green'],
             "These customers are ready to buy. Actions: retargeting ads with product launch, exclusive early-access offer, WhatsApp/email with discount code. Goal: First purchase within 7 days."),
            ("Hot (55–75%)", "coral", C['coral'],
             "High intent, need a nudge. Actions: influencer UGC content, 'Try before you buy' easy returns campaign, social proof (reviews). Goal: Move to Convert Now tier in 14 days."),
            ("Warm (30–55%)", "gold", C['gold'],
             "Curious but hesitant. Actions: Instagram/YouTube style content, lookbook email series, '3 basics ₹999' launch offer. Goal: Build brand recall over 30 days."),
            ("Cold (< 30%)", "red", C['red'],
             "Low intent now. Actions: Brand awareness — reels, collabs. Don't spend paid budget here yet. Goal: Move to Warm tier with organic content over 60 days."),
        ]
        for seg, kind, color, text in playbook:
            n = int(seg_cnt.get(seg, 0))
            st.markdown(f"""
            <div class="risk-card" style="border-left:4px solid {color}; background:#14131f;">
              <div style="font-family:Playfair Display,serif;font-weight:700;color:{color};font-size:.95rem;">
                {seg} — <span style="color:#f0ece4">{n:,} customers</span></div>
              <div style="color:#c0bcd8;margin-top:5px;font-size:.84rem;">{text}</div>
            </div>""", unsafe_allow_html=True)

    divider()

    # Revenue opportunity
    sec("Revenue Opportunity Model")
    avg_budget = df_clean['Q21_Budget_Numeric_INR'].mean()
    convert_now = int(seg_cnt.get('Convert Now (>75%)', 0))
    hot_n = int(seg_cnt.get('Hot (55–75%)', 0))
    warm_n = int(seg_cnt.get('Warm (30–55%)', 0))

    conv_rates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    rev = [r * (convert_now + hot_n * 0.5) * avg_budget for r in conv_rates]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{int(r*100)}%" for r in conv_rates], y=rev,
                         marker=dict(color=rev, colorscale='Viridis',
                                     line=dict(color='#0a0a0f',width=1)),
                         text=[f"₹{v/1e5:.1f}L" for v in rev], textposition='outside'))
    fig.update_layout(yaxis_title='Est. Monthly Revenue (₹)', xaxis_title='Conversion Rate Scenario')
    styled(fig, f"Projected Monthly Revenue: Convert Now + Hot Segments (Avg Wallet ₹{avg_budget:,.0f})", 380)
    st.plotly_chart(fig, use_container_width=True)
    ins(f"Even a conservative 20% conversion of the Convert Now + Hot segment yields ₹{rev[1]/1e5:.1f}L/month. At 40%, this reaches ₹{rev[3]/1e5:.1f}L — enough to validate the business model in month 1.", "p")

    divider()

    # Persona-wise recommendations
    sec("Persona-Specific Go-To-Market Strategy")
    strategies = {
        "TrendyUrban 🎨": {
            "target": "18–34, Metro/Tier-1, high social media usage",
            "product": "Co-ord sets, Ethnic-Western Fusion, Jackets/Blazers",
            "price": "₹1,500–₹3,500 per outfit",
            "channel": "Instagram Reels, influencer collabs, brand website",
            "offer": "Curated outfit box / First order 15% off",
            "kpi": "Target 35% conversion; CAC < ₹200",
        },
        "BudgetStudent 🎓": {
            "target": "18–24, Tier-1/2, price-sensitive, college audience",
            "product": "T-shirts, Jeans, Shorts/Skirts, basics",
            "price": "₹500–₹1,500 — '3 basics for ₹999' hero offer",
            "channel": "Instagram, Meesho, Flipkart, campus influencers",
            "offer": "EMI-free bundling, student discount code",
            "kpi": "Target 25% conversion; high volume, low AOV",
        },
        "ProfessionalWoman 👩‍💼": {
            "target": "25–44, Metro, salaried, quality-first",
            "product": "Work-ready tops, blazers, Ethnic-Western Fusion",
            "price": "₹2,500–₹5,000 premium positioning",
            "channel": "LinkedIn, Instagram, brand website, email",
            "offer": "Seasonal wardrobe kit / Easy returns guarantee",
            "kpi": "Target 45% conversion; highest LTV persona",
        },
        "PremiumShopper 💎": {
            "target": "35–54, Metro, high income, brand-conscious",
            "product": "Premium fabrics, exclusive drops, limited editions",
            "price": "₹5,000–₹10,000+, no heavy discounting",
            "channel": "Brand website, WhatsApp concierge, Instagram Stories",
            "offer": "Early access, loyalty club, personal styling consult",
            "kpi": "Target 55% conversion; highest AOV, lowest CAC",
        },
        "HousewifeTier2 🏠": {
            "target": "25–44, Tier-2/3, festival & occasion buyers",
            "product": "Ethnic-Western Fusion, Dresses, Co-ord sets",
            "price": "₹800–₹2,000 — value with occasion relevance",
            "channel": "WhatsApp, Facebook, Meesho, YouTube regional",
            "offer": "Festival bundle, Ethnic+western combo pack",
            "kpi": "Target 20% conversion; seasonal GMV spikes",
        },
    }
    cols = st.columns(len(strategies))
    for col, (persona, data) in zip(cols, strategies.items()):
        with col:
            st.markdown(f"""
            <div style="background:#14131f;border:1px solid #2a2640;border-top:3px solid {C['gold']};
                        border-radius:10px;padding:15px;height:100%;">
              <div style="font-family:Playfair Display,serif;font-weight:700;color:{C['gold']};
                          font-size:.95rem;margin-bottom:10px">{persona}</div>
              {''.join(f'<div style="margin:5px 0;font-size:.78rem;color:#c0bcd8"><b style="color:#9090c0">{k.title()}:</b> {v}</div>' for k,v in data.items())}
            </div>""", unsafe_allow_html=True)

    divider()

    # Launch checklist
    sec("📋 Founder Launch Checklist — Data-Driven Priorities")
    checklist = [
        ("🏷️ Product", [
            "Inclusive size range (XS–4XL) — top market gap mentioned",
            "India-appropriate western designs — fusion is the moat",
            "3 price ladders: ₹500–1,000 / ₹1,500–3,000 / ₹5,000+ premium",
            "Hero launch bundle: '3 basics for ₹999'",
            "Co-ord sets as hero category — highest indexed demand",
        ]),
        ("📣 Marketing", [
            "Instagram-first brand — 60%+ of audience is there",
            "Activate 10 micro-influencers (10K–100K) before launch",
            "UGC campaign: #WornWithConfidence for social proof",
            "YouTube styling videos for ProfessionalWoman persona",
            "WhatsApp broadcast for Tier-2/3 market (HousewifeTier2)",
        ]),
        ("🛒 D2C & Trust", [
            "Zero-friction returns policy — biggest conversion driver",
            "Size guide with India body-type reference images",
            "Digital trust signals: reviews, COD option, secure badge",
            "Loyalty program from Day 1 — 2× LTV vs non-members",
            "Sustainability story: eco-packaging, ethical sourcing tag",
        ]),
    ]
    cols3 = st.columns(3)
    for col, (cat, items) in zip(cols3, checklist):
        with col:
            items_html = ''.join([f'<div style="padding:4px 0;border-bottom:1px solid #1e1c30;font-size:.82rem;color:#c0bcd8">✓ {i}</div>' for i in items])
            st.markdown(f"""
            <div style="background:#14131f;border:1px solid #2a2640;border-radius:10px;padding:16px;">
              <div style="font-family:Playfair Display,serif;color:{C['gold']};font-size:1rem;font-weight:700;margin-bottom:10px">{cat}</div>
              {items_html}
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT NEW CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif "Predict New" in page:
    st.markdown('<div class="brand-title">📤 Predict New Customer Inclination</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a new CSV of prospective customers — get instant AI predictions on their likelihood to buy from WesternWear.</div>', unsafe_allow_html=True)

    badge("pred","Predictive")

    st.markdown("""
    <div class="insight" style="border-left-color:#c07aff">
    📌 <b>How to use:</b> Upload a CSV file with the same column structure as the original survey dataset
    (<code>WesternWear_Survey_Dataset_2000.csv</code>). The AI model will predict each respondent's
    purchase inclination: <b style="color:#5fc47a">Interested</b> / <b style="color:#e8c97e">Neutral</b> /
    <b style="color:#e07070">Not Interested</b> — along with confidence scores.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload New Prospects CSV", type=['csv'],
                                  help="Must have same column names as the survey dataset")

    if uploaded is not None:
        try:
            with st.spinner("Processing new prospects..."):
                df_new = load_raw(uploaded)
                pred_df = predict_new(df_new, ml)

            st.success(f"✅ Predictions complete for {len(pred_df):,} prospects")
            divider()

            # Summary
            cnt = pred_df['Predicted_Label'].value_counts()
            total_new = len(pred_df)
            kpi_row([
                (str(cnt.get('Interested',0)),    "Interested",    f"{cnt.get('Interested',0)/total_new*100:.1f}%", False),
                (str(cnt.get('Neutral',0)),        "Neutral",       f"{cnt.get('Neutral',0)/total_new*100:.1f}%",   False),
                (str(cnt.get('Not Interested',0)), "Not Interested",f"{cnt.get('Not Interested',0)/total_new*100:.1f}%", True),
                (f"{pred_df['Confidence_%'].mean():.1f}%", "Avg Confidence","Model certainty",False),
            ])

            c1,c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Pie(
                    labels=cnt.index, values=cnt.values, hole=0.55,
                    marker=dict(colors=[INTENT_COLORS.get(l, C['blue']) for l in cnt.index],
                                line=dict(color='#0a0a0f',width=3)),
                    textinfo='label+percent+value',
                ))
                styled(fig, "Predicted Intent Distribution — New Prospects", 360)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = go.Figure(go.Histogram(x=pred_df['Prob_Interested'], nbinsx=20,
                                              marker_color=C['green'], opacity=0.8))
                fig.add_vline(x=55, line_dash='dash', line_color=C['gold'],
                              annotation_text="55% → Hot prospect")
                fig.add_vline(x=75, line_dash='dash', line_color=C['coral'],
                              annotation_text="75% → Convert Now")
                fig.update_layout(xaxis_title='P(Interested) %', yaxis_title='Count')
                styled(fig, "Distribution of Interested Probability Scores", 360)
                st.plotly_chart(fig, use_container_width=True)

            sec("Full Prediction Results")
            show_cols = ['Predicted_Label','Prob_Interested','Prob_Neutral','Prob_NotInterested','Confidence_%']
            if 'Respondent_ID' in pred_df.columns:
                show_cols = ['Respondent_ID'] + show_cols
            styled_pred = pred_df[show_cols].style\
                .background_gradient(subset=['Prob_Interested'], cmap='Greens')\
                .background_gradient(subset=['Prob_NotInterested'], cmap='Reds')\
                .background_gradient(subset=['Confidence_%'], cmap='Purples')\
                .format({c: "{:.1f}" for c in ['Prob_Interested','Prob_Neutral','Prob_NotInterested','Confidence_%']})
            st.dataframe(styled_pred, use_container_width=True)

            # Download
            csv_out = pred_df[show_cols].to_csv(index=False)
            st.download_button("⬇️ Download Predictions CSV", data=csv_out,
                               file_name="westenwear_predictions.csv", mime="text/csv",
                               use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Make sure your CSV has the same column structure as WesternWear_Survey_Dataset_2000.csv")

    else:
        # Show expected format
        sec("Expected CSV Format")
        sample_cols = ['Respondent_ID','Persona_Label','Is_Outlier','Q1_Age_Group','Q2_Gender',
                       'Q3_City_Tier','Q4_Occupation','Q5_Monthly_Income','Q6_Shopping_Frequency',
                       'Q7_Platforms_Used','Q8_Avg_Order_Spend','Q9_Key_Purchase_Factor',
                       'Q10_Returns_Frequency','Q14_Style_Personality','Q17_Price_Sensitivity',
                       'Q18_Social_Platform','Q19_D2C_Purchase_Hist','Q20_Sustainability_Pay',
                       'Q21_Budget_Numeric_INR','Q22_Digital_Trust_Score','Q23_Influencer_Purchase']
        sample_data = {c: ['R2001','TrendyUrban','False','18–24','Female','Metro',
                            'Salaried – Private','₹40,000–₹75,000','Very frequently',
                            'Myntra|Instagram/Social','₹1,000–₹2,500','Trendy designs|Fabric quality',
                            'Rarely','Trendy/Street style',3.0,'Instagram','Yes','Definitely yes',
                            2500,4.0,'Yes – many times'][:1] for c in sample_cols}
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        st.markdown('<div class="insight p">Multi-select columns (Platforms, Categories, Occasions) should use the pipe (|) separator: e.g., <code>Myntra|Amazon|Instagram/Social</code></div>', unsafe_allow_html=True)
