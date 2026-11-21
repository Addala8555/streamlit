import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS (Matches UI Mockups)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Home Credit | Risk Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f4f6f9; }
    
    /* Dark Navy Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b192c;
        color: white;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0; }

    /* Make selected filter values readable against the sidebar */
    [data-testid="stSidebar"] .react-aria-ComboBox [role="group"] {
        background-color: #132b46;
        border: 1px solid #38bdf8;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .react-aria-ComboBox [role="combobox"] {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] .react-aria-ComboBox [role="group"]:focus-within {
        border-color: #22d3ee;
        box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.25);
    }

    [role="listbox"] [role="option"] {
        color: #0f172a !important;
    }
    [role="listbox"] [role="option"]:hover,
    [role="listbox"] [role="option"][aria-selected="true"] {
        background-color: #ccfbf1;
        color: #134e4a !important;
    }
    
    /* KPI Cards */
    .kpi-card {
        background-color: white;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-left: 5px solid #0f766e;
        margin-bottom: 10px;
    }
    .kpi-card.red { border-left-color: #dc2626; background-color: #fef2f2; }
    .kpi-card.amber { border-left-color: #d97706; background-color: #fffbeb; }
    .kpi-card.blue { border-left-color: #2563eb; background-color: #eff6ff; }
    .kpi-title { font-size: 13px; color: #64748b; font-weight: 600; margin-bottom: 4px; }
    .kpi-value { font-size: 26px; font-weight: 800; color: #0f172a; }
    .kpi-sub { font-size: 12px; color: #475569; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA LOADING (Uses real Kaggle CSV or generates sample)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/application_train.csv")
    except FileNotFoundError:
        # Fallback synthetic dataset matching Kaggle schema so app runs immediately
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "SK_ID_CURR": range(100001, 100001 + n),
            "TARGET": np.random.choice([0, 1], size=n, p=[0.919, 0.081]),
            "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], size=n, p=[0.90, 0.10]),
            "CODE_GENDER": np.random.choice(["F", "M"], size=n, p=[0.66, 0.34]),
            "AMT_INCOME_TOTAL": np.random.lognormal(mean=11.9, sigma=0.5, size=n).round(-3),
            "AMT_CREDIT": np.random.lognormal(mean=13.1, sigma=0.6, size=n).round(-3),
            "AMT_ANNUITY": np.random.normal(27000, 10000, size=n).clip(5000).round(-2),
            "AMT_GOODS_PRICE": np.random.lognormal(mean=13.0, sigma=0.6, size=n).round(-3),
            "NAME_EDUCATION_TYPE": np.random.choice(
                ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
                size=n, p=[0.71, 0.24, 0.03, 0.015, 0.005]
            ),
            "NAME_FAMILY_STATUS": np.random.choice(
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
                size=n, p=[0.64, 0.15, 0.10, 0.06, 0.05]
            ),
            "OCCUPATION_TYPE": np.random.choice(
                ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers", "Accountants", "IT staff"],
                size=n
            ),
            "DAYS_BIRTH": -np.random.randint(21*365, 68*365, size=n),
            "DAYS_EMPLOYED": -np.random.randint(100, 7000, size=n),
            "CNT_CHILDREN": np.random.choice([0, 1, 2, 3], size=n, p=[0.70, 0.20, 0.08, 0.02]),
            "CNT_FAM_MEMBERS": np.random.choice([1.0, 2.0, 3.0, 4.0], size=n, p=[0.20, 0.50, 0.20, 0.10])
        })
    
    # Derived Features (Matching Screenshot 2)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    df["EMP_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).clip(0, 45).round(1)
    df["LOAN_TO_INCOME"] = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).round(2)
    df["ANNUITY_TO_INCOME"] = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).round(2)
    return df

df = load_data()

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION & GLOBAL FILTERS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏠 Home Credit\n**Risk Analytics Dashboard**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. Customer Demographics",
            "3. Loan Application Analysis",
            "4. Default Risk Analysis",
            "5. Income & Employment"
        ]
    )
    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")
    contract_filter = st.selectbox("Contract Type", ["All"] + list(df["NAME_CONTRACT_TYPE"].unique()))
    gender_filter_codes = {"All": None, "Female": "F", "Male": "M"}
    gender_filter = st.selectbox("Gender", list(gender_filter_codes))

    active_filters = []
    if contract_filter != "All":
        active_filters.append(f"Contract: {contract_filter}")
    if gender_filter != "All":
        active_filters.append(f"Gender: {gender_filter}")
    st.markdown("**Active selection**")
    st.caption(" · ".join(active_filters) if active_filters else "No filters selected")

    st.markdown("---")
    st.caption(f"**Dataset:** Home Credit Default Risk\n\n**Rows Loaded:** {len(df):,}")

# Apply Filters
filtered_df = df.copy()
if contract_filter != "All":
    filtered_df = filtered_df[filtered_df["NAME_CONTRACT_TYPE"] == contract_filter]
gender_code = gender_filter_codes[gender_filter]
if gender_code is not None:
    filtered_df = filtered_df[filtered_df["CODE_GENDER"] == gender_code]

def kpi_html(title, value, sub="", style=""):
    return f"""
    <div class="kpi-card {style}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """

# ---------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW (Matches Screenshot 5)
# ---------------------------------------------------------
if page == "1. Executive Overview":
    st.title("Home Credit | Executive Overview")
    st.caption("Key performance indicators and overall loan portfolio summary")

    total_apps = len(filtered_df)
    defaults = int(filtered_df["TARGET"].sum())
    non_defaults = total_apps - defaults
    default_rate = (defaults / total_apps * 100) if total_apps else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(kpi_html("Applications", f"{total_apps:,}", "Filtered records", "blue"), unsafe_allow_html=True)
    c2.markdown(kpi_html("Defaults", f"{defaults:,}", "TARGET = 1", "red"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Default Rate", f"{default_rate:.2f}%", "Portfolio risk", "amber"), unsafe_allow_html=True)
    c4.markdown(kpi_html("Non-defaults", f"{non_defaults:,}", "TARGET = 0"), unsafe_allow_html=True)
    c5.markdown(kpi_html("Average Credit", f"{filtered_df['AMT_CREDIT'].mean():,.0f}", "AMT_CREDIT", "blue"), unsafe_allow_html=True)
    c6.markdown(kpi_html("Average Income", f"{filtered_df['AMT_INCOME_TOTAL'].mean():,.0f}", "AMT_INCOME"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_donut = px.pie(
            names=["Non-default", "Default"],
            values=[non_defaults, defaults],
            hole=0.6,
            title="Default vs Non-default",
            color_discrete_sequence=["#12DAC9", "#ED4C08"]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            x=["Non-default", "Default"],
            y=[non_defaults, defaults],
            text=[f"{non_defaults:,}", f"{defaults:,}"],
            title="Application Outcome",
            color=["Non-default", "Default"],
            color_discrete_sequence=["#12DAC9", "#ED4C08"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col3:
        fig_hist = px.histogram(
            filtered_df, x="AMT_CREDIT", nbins=30,
            title="Loan Amount Distribution (AMT_CREDIT)",
            color_discrete_sequence=["#12DAC9"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    col4, col5 = st.columns(2)
    with col4:
        sample_scatter = filtered_df.sample(min(1500, len(filtered_df)), random_state=42)
        fig_scat = px.scatter(
            sample_scatter, x="AMT_INCOME_TOTAL", y="AMT_CREDIT",
            color=sample_scatter["TARGET"].astype(str),
            title="Income vs Credit Amount",
            color_discrete_sequence=["#1708dd", "#dc2626"]
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with col5:
        contract_risk = filtered_df.groupby("NAME_CONTRACT_TYPE")["TARGET"].mean().reset_index()
        contract_risk["TARGET"] *= 100
        fig_cr = px.bar(
            contract_risk, x="NAME_CONTRACT_TYPE", y="TARGET",
            text=contract_risk["TARGET"].round(2).astype(str) + "%",
            title="Default Rate by Contract Type (%)",
            color="NAME_CONTRACT_TYPE",
            color_discrete_sequence=["#12DAC9", "#ED4C08"]
        )
        st.plotly_chart(fig_cr, use_container_width=True)

    st.subheader("Sample Application Data")
    st.dataframe(filtered_df[["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_BIRTH"]].head(10), use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: CUSTOMER DEMOGRAPHICS (Matches Screenshot 2)
# ---------------------------------------------------------
elif page == "2. Customer Demographics":
    st.title("Customer Demographics")
    st.caption("Customer profile, family structure, education, and age analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Average Age", f"{filtered_df['AGE_YEARS'].mean():.2f} Years", "Derived from DAYS_BIRTH", "blue"), unsafe_allow_html=True)
    married_pct = (filtered_df["NAME_FAMILY_STATUS"] == "Married").mean() * 100
    c2.markdown(kpi_html("Married Customers", f"{married_pct:.2f}%", "Family status share", "amber"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Avg Family Size", f"{filtered_df['CNT_FAM_MEMBERS'].mean():.2f}", "CNT_FAM_MEMBERS"), unsafe_allow_html=True)
    children_pct = (filtered_df["CNT_CHILDREN"] > 0).mean() * 100
    c4.markdown(kpi_html("Customers with Children", f"{children_pct:.2f}%", "CNT_CHILDREN > 0"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(filtered_df, x="AGE_YEARS", nbins=25, title="Age Distribution (Years)", color_discrete_sequence=["#6366f1"])
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        edu_risk = filtered_df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values().reset_index()
        edu_risk["TARGET"] *= 100
        fig_edu = px.bar(
            edu_risk, x="TARGET", y="NAME_EDUCATION_TYPE", orientation="h",
            text=edu_risk["TARGET"].round(2).astype(str) + "%",
            title="Default Rate by Education (%)", color_discrete_sequence=["#6366f1"]
        )
        st.plotly_chart(fig_edu, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: LOAN APPLICATION ANALYSIS (Matches Screenshot 6)
# ---------------------------------------------------------
elif page == "3. Loan Application Analysis":
    st.title("Loan Application Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Total Loan Amount", f"{filtered_df['AMT_CREDIT'].sum():,.0f}", "Sum of AMT_CREDIT", "blue"), unsafe_allow_html=True)
    c2.markdown(kpi_html("Median Loan Amount", f"{filtered_df['AMT_CREDIT'].median():,.0f}", "50th Percentile", "amber"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Avg Loan-to-Income", f"{filtered_df['LOAN_TO_INCOME'].mean():.2f}", "AMT_CREDIT / INCOME"), unsafe_allow_html=True)
    c4.markdown(kpi_html("Avg Annuity-to-Income", f"{filtered_df['ANNUITY_TO_INCOME'].mean():.2f}", "AMT_ANNUITY / INCOME"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_lti = px.histogram(filtered_df[filtered_df["LOAN_TO_INCOME"] < 12], x="LOAN_TO_INCOME", nbins=30, title="Loan-to-Income Ratio Distribution", color_discrete_sequence=["#0f766e"])
        st.plotly_chart(fig_lti, use_container_width=True)
    with col2:
        ann_comp = filtered_df.groupby("TARGET")["ANNUITY_TO_INCOME"].mean().reset_index()
        ann_comp["Status"] = ann_comp["TARGET"].map({0: "Non-default (0)", 1: "Default (1)"})
        fig_ann = px.bar(ann_comp, x="Status", y="ANNUITY_TO_INCOME", color="Status", text=ann_comp["ANNUITY_TO_INCOME"].round(2), title="Avg Annuity-to-Income by Outcome", color_discrete_sequence=["#0f766e", "#f59e0b"])
        st.plotly_chart(fig_ann, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: DEFAULT RISK ANALYSIS (Matches Screenshot 3)
# ---------------------------------------------------------
elif page == "4. Default Risk Analysis":
    st.title("Default Risk Analysis")
    defaulters = filtered_df[filtered_df["TARGET"] == 1]
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_html("Total Defaults", f"{len(defaulters):,}", "High-risk segment", "red"), unsafe_allow_html=True)
    c2.markdown(kpi_html("Avg Credit (Defaulters)", f"{defaulters['AMT_CREDIT'].mean():,.0f}", "AMT_CREDIT", "red"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Avg Income (Defaulters)", f"{defaulters['AMT_INCOME_TOTAL'].mean():,.0f}", "AMT_INCOME_TOTAL", "red"), unsafe_allow_html=True)

    filtered_df["AGE_GROUP"] = pd.cut(filtered_df["AGE_YEARS"], bins=[20, 25, 35, 45, 55, 75], labels=["≤25", "26-35", "36-45", "46-55", "≥56"])
    age_risk = filtered_df.groupby("AGE_GROUP", observed=False)["TARGET"].mean().reset_index()
    age_risk["TARGET"] *= 100
    fig_ar = px.bar(age_risk, x="AGE_GROUP", y="TARGET", text=age_risk["TARGET"].round(2).astype(str) + "%", title="Default Rate by Age Group (%)", color_discrete_sequence=["#dc2626"])
    st.plotly_chart(fig_ar, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: INCOME & EMPLOYMENT (Matches Screenshot 4)
# ---------------------------------------------------------
elif page == "5. Income & Employment":
    st.title("Income & Employment")
    occ_risk = filtered_df.groupby("OCCUPATION_TYPE").agg(
        Avg_Income=("AMT_INCOME_TOTAL", "mean"),
        Default_Rate=("TARGET", "mean")
    ).reset_index()
    occ_risk["Default_Rate"] *= 100

    col1, col2 = st.columns(2)
    with col1:
        fig_occ_inc = px.bar(occ_risk.sort_values("Avg_Income"), x="Avg_Income", y="OCCUPATION_TYPE", orientation="h", title="Average Income by Occupation", color_discrete_sequence=["#065f46"])
        st.plotly_chart(fig_occ_inc, use_container_width=True)
    with col2:
        fig_occ_def = px.bar(occ_risk.sort_values("Default_Rate", ascending=False), x="OCCUPATION_TYPE", y="Default_Rate", title="Default Rate by Occupation (%)", color_discrete_sequence=["#d97706"])
        st.plotly_chart(fig_occ_def, use_container_width=True)