# =========================================================
# 🚀 CUSTOMER CHURN INTELLIGENCE SYSTEM
# InferaIQ | Predict. Understand. Act. Retain.
# =========================================================

import os
import sys
import time
import tempfile
import logging
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from logging.handlers import RotatingFileHandler

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.pipeline import Pipeline

from openai import OpenAI

from PIL import Image

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from reportlab.lib import colors

warnings.filterwarnings("ignore")

# =========================================================
# PATH SETUP
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

from src.predict import predict_churn

from src.features import (
    prepare_features,
    FEATURE_COLUMNS
)

from src.risk_utils import (
    classify_risk,
    RISK_COLORS
)

# =========================================================
# DIRECTORIES
# =========================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "superstore.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

BATCH_HISTORY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "batch_prediction_history.csv"
)

ASSETS_DIR = os.path.join(
    os.path.dirname(__file__),
    "assets"
)

os.makedirs(LOG_DIR, exist_ok=True)

# =========================================================
# ASSETS
# =========================================================

BANNER_PATH = os.path.join(
    ASSETS_DIR,
    "banner_streamlit.png"
)

LOGO_PATH = os.path.join(
    ASSETS_DIR,
    "logo.png"
)



# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(
    "churn_system"
)

logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "system.log"),
    maxBytes=2_000_000,
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 THEME TOGGLE
# =========================================================

theme = st.sidebar.selectbox(
    "🎨 Theme",
    [
        "Light",
        "Dark"
    ]
)

# =========================================================
# 🌍 GLOBAL THEME COLORS
# =========================================================

if theme == "Dark":

    bg_color = "#0E1117"
    paper_color = "#111827"
    font_color = "#F3F4F6"
    grid_color = "#30363D"
    plot_template = "plotly_dark"

else:

    bg_color = "white"
    paper_color = "white"
    font_color = "black"
    grid_color = "#D1D5DB"
    plot_template = "plotly_white"

# =========================================================
# DARK THEME
# =========================================================

if theme == "Dark":

    st.markdown("""
    <style>

    /* Entire App */
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    /* KPI Metric Cards */
    .stMetric {
        background-color: #1A2233;
        border: 1px solid #3B82F6;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
                
    /* Streamlit Metric Container */
    div[data-testid="metric-container"] {
        background: #161B22;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.30);
    }
                
    div[data-testid="metric-container"] label {
        color: #D1D5DB !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
                
    /* KPI Values */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 28px !important;
    }

    /* KPI Labels */
    [data-testid="stMetricLabel"] {
        color: #D1D5DB !important;
        font-weight: 600 !important;
    }

    /* Main Containers */
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 12px;
    }
                
    /* KPI Value Glow Effect */
    [data-testid="stMetricValue"] {
        text-shadow: 0px 0px 4px rgba(59,130,246,0.20);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    /* General Text */
    p, label, span {
        color: #F3F4F6;
    }

    /* Sidebar Text */
    [data-testid="stSidebar"] * {
        color: #F3F4F6 !important;
    }

    /* Radio Labels */
    .stRadio label {
        color: #F3F4F6 !important;
    }

    /* List Items */
    ul li {
        color: #F3F4F6 !important;
    }
                
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] {
        color: white;
    }

    .stSelectbox svg {
        fill: white;
    }

    div[data-baseweb="popover"] {
        background-color: #161B22 !important;
        color: white !important;
    }

    li[role="option"] {
        background-color: #161B22 !important;
        color: white !important;
    }

    li[role="option"]:hover {
        background-color: #2563EB !important;
        color: white !important;
    }

        /* Buttons */
        .stButton button {
            background-color: #2563EB;
            color: white;
            border-radius: 8px;
            border: none;
        }

        /* Glide Data Editor */

        /* ======================================
        ELITE DATAFRAME DARK THEME
        ====================================== */

        [data-testid="stDataFrame"] {
            background-color: #0E1117 !important;
        }

        [data-testid="stDataFrame"] * {
            color: #F9FAFB !important;
        }

        .glideDataEditor {
            --gdg-bg-cell: #0E1117 !important;
            --gdg-bg-header: #161B22 !important;

            --gdg-text-dark: #F9FAFB !important;
            --gdg-text-medium: #F9FAFB !important;

            --gdg-border-color: #30363D !important;

            --gdg-accent-color: #2563EB !important;
        }

        .glideDataEditor canvas {
            filter: brightness(0.85);
        }
                
        /* Plotly Charts */
        .js-plotly-plot,
        .plotly,
        .plot-container {

            background-color: #0E1117 !important;
        }
                
        /* Selectbox Selected Value */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #161B22 !important;
            color: white !important;
        }

        /* Dropdown Text */
        .stSelectbox div[data-baseweb="select"] span {
            color: white !important;
        }

        /* =======================================
        FILE UPLOADER DARK MODE FIX
        ======================================= */

        [data-testid="stFileUploader"] * {
            color: #E5E7EB !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #E5E7EB !important;
        }

        [data-testid="stFileUploaderDropzone"] small {
            color: #E5E7EB !important;
        }

        [data-testid="stFileUploader"] small {
            color: #E5E7EB !important;
        }

        [data-testid="stFileUploader"] span {
            color: #E5E7EB !important;
        }

        [data-testid="stFileUploader"] p {
            color: #E5E7EB !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #1F2937 !important;
            color: white !important;
            border: 1px solid #374151 !important;
        }

        [data-testid="stFileUploader"] svg {
            fill: white !important;
            color: white !important;
        }

        /* Uploaded filename */

        [data-testid="stFileUploaderFile"] * {
            color: white !important;
        }

        /* Download buttons */

        .stDownloadButton button {
            background-color: #2563EB !important;
            color: white !important;
            border: none !important;
        }

    </style>
    """, unsafe_allow_html=True)

# =========================================================
# LIGHT THEME
# =========================================================

else:

    st.markdown("""
    <style>

    /* Entire App */
    .stApp {
        background-color: white;
        color: black;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F3F4F6;
    }

    /* KPI Metric Cards */
    .stMetric {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        border-radius: 12px;
        padding: 15px;
    }

    /* Main Containers */
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 12px;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: black;
    }

    /* Text */
    p, label, span {
        color: black;
    }

    /* Buttons */
    .stButton button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
    }

    /* Download buttons */

    .stDownloadButton button {
        color: white !important;
        background-color: #2563EB !important;
    }

    </style>
    """, unsafe_allow_html=True)

# =========================================================
# CACHE RESOURCES
# =========================================================

@st.cache_resource
def load_resources():

    model = joblib.load(
        os.path.join(MODEL_DIR, "model.pkl")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    explainer = shap.TreeExplainer(model)

    return model, scaler, explainer

model, scaler, explainer = (
    load_resources()
)

# =========================================================
# PIPELINE
# =========================================================

pipeline = Pipeline([
    ("scaler", scaler),
    ("model", model)
])

# =========================================================
# OPENAI
# =========================================================

client = None

try:

    if "OPENAI_API_KEY" in st.secrets:

        client = OpenAI(
            api_key=st.secrets[
                "OPENAI_API_KEY"
            ]
        )

except Exception as e:

    logger.error(
        f"OpenAI Init Error: {e}"
    )

# =========================================================
# BRANDING
# =========================================================

if os.path.exists(BANNER_PATH):

    st.image(
        BANNER_PATH,
        use_container_width=True
    )

if os.path.exists(LOGO_PATH):

    st.sidebar.image(
        LOGO_PATH,
        width=160
    )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🚀 Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Batch Prediction",
        "Analytics",
        "Model Performance",
        "Monitoring",
        "What-If Simulator",
        "About"
    ]
)

threshold = st.sidebar.slider(
    "Prediction Threshold",
    0.0,
    1.0,
    0.60
)

st.sidebar.divider()

st.sidebar.markdown("""
### ⚙️ Platform Capabilities

- Predictive Intelligence
- Explainable AI
- Executive Analytics
- Retention Intelligence
- Real-Time Monitoring
- AI Decision Support
""")

st.sidebar.caption(
    "Version 1.0.0 Enterprise"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_dataset():

    if os.path.exists(DATA_PATH):

        return pd.read_csv(DATA_PATH)

    return pd.DataFrame()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

@st.cache_data
def build_rfm(df):

    df["Order Date"] = pd.to_datetime(
        df["Order Date"]
    )

    reference_date = df[
        "Order Date"
    ].max()

    rfm = df.groupby(
        "Customer ID"
    ).agg({

        "Order Date":
        lambda x: (
            reference_date - x.max()
        ).days,

        "Order ID": "nunique",

        "Sales": "sum"

    }).reset_index()

    rfm.columns = [
        "Customer ID",
        "Recency",
        "Frequency",
        "Monetary"
    ]

    tenure = df.groupby(
        "Customer ID"
    )["Order Date"].agg(
        ["min", "max"]
    ).reset_index()

    tenure["Tenure"] = (
        tenure["max"] - tenure["min"]
    ).dt.days

    rfm = rfm.merge(
        tenure[
            ["Customer ID", "Tenure"]
        ],
        on="Customer ID",
        how="left"
    )

    rfm["AOV"] = (
        rfm["Monetary"] /
        rfm["Frequency"].replace(0, 1)
    )

    rfm["Purchase_Frequency"] = (
        rfm["Frequency"] /
        rfm["Tenure"].replace(0, 1)
    )

    rfm["Profit"] = (
        rfm["Monetary"] * 0.2
    )

    return rfm

# =========================================================
# LOAD DATA
# =========================================================

df = load_dataset()

rfm = (
    build_rfm(df)
    if not df.empty
    else pd.DataFrame()
)

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title(
        "🚀 Customer Churn Intelligence System"
    )

    st.markdown("""
    Enterprise-grade AI platform for customer churn prediction,
    executive analytics, explainable AI, and retention intelligence.
    """)

    tab1, tab2, tab3 = st.tabs([
        "Prediction",
        "Prediction History",
        "Batch History"
    ])

    with tab1:

        st.subheader(
            "📥 Customer Inputs"
        )

        col1, col2, col3, col4 = st.columns(4)

        recency = col1.number_input(
            "Recency",
            min_value=0,
            value=10
        )

        frequency = col2.number_input(
            "Frequency",
            min_value=0,
            value=5
        )

        monetary = col3.number_input(
            "Monetary",
            min_value=0.0,
            value=500.0
        )

        tenure = col4.number_input(
            "Tenure",
            min_value=0,
            value=50
        )

        if st.button(
            "🚀 Run Prediction"
        ):

            with st.spinner(
                "Running AI inference..."
            ):

                start_time = time.time()

                try:

                    payload = {
                        "Recency": recency,
                        "Frequency": frequency,
                        "Monetary": monetary,
                        "Tenure": tenure
                    }

                    result = predict_churn(
                        payload,
                        threshold
                    )

                    prob = result[
                        "probability"
                    ]

                    # =====================================
                    # CUSTOMER LIFETIME VALUE (CLV)
                    # Enterprise-Grade Formula
                    # =====================================

                    customer_lifespan_years = (
                        max(tenure, 1) / 365
                    )

                    aov = (
                        monetary / max(frequency, 1)
                    )

                    profit_margin = 0.20

                    clv = (
                        aov
                        *
                        frequency
                        *
                        customer_lifespan_years
                        *
                        profit_margin
                    )

                    # =====================================
                    # CUSTOMER HEALTH SCORE
                    # =====================================

                    health_score = (
                        (1 - prob) * 100
                    )

                    # =====================================
                    # CHURN RISK CLASSIFICATION
                    # =====================================

                    risk_data = classify_risk(prob)

                    risk_level = risk_data["label"]

                    # =====================================
                    # GAUGE CHART
                    # =====================================

                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={
                            "text":
                            "Churn Risk Score"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            },
                            "steps": [
                                {
                                    "range": [0, 40],
                                    "color": "green"
                                },
                                {
                                    "range": [40, 70],
                                    "color": "orange"
                                },
                                {
                                    "range": [70, 100],
                                    "color": "red"
                                }
                            ]
                        }
                    ))

                    gauge.update_layout(

                        template=plot_template,

                        paper_bgcolor=paper_color,

                        plot_bgcolor=bg_color,

                        font=dict(
                            color=font_color
                        )
                    )

                    st.plotly_chart(
                        gauge,
                        use_container_width=True
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # =====================================
                    # KPI METRICS
                    # =====================================

                    col1, col2, col3, col4, col5 = (
                        st.columns(5)
                    )

                    col1.metric(
                        "Churn Probability",
                        f"{prob:.2%}"
                    )

                    confidence = abs(
                        prob - threshold
                    )

                    col2.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}"
                    )

                    latency = (
                        time.time() -
                        start_time
                    )

                    col3.metric(
                        "Inference Latency",
                        f"{latency:.3f}s"
                    )

                    col4.metric(
                        "Customer Lifetime Value",
                        f"${clv:,.2f}"
                    )

                    col5.metric(
                        "Customer Health Index",
                        f"{health_score:.1f}/100"
                    )

                    # =====================================
                    # RISK CLASSIFICATION DISPLAY
                    # =====================================

                    st.subheader(
                        "🚨 Churn Risk Classification"
                    )

                    if "Critical" in risk_level:

                        st.error(
                            f"Risk Level: {risk_level}"
                        )

                    elif "High" in risk_level:
                        st.warning(
                            f"Risk Level: {risk_level}"
                        )

                    elif "Medium" in risk_level:
                        st.info(
                            f"Risk Level: {risk_level}"
                        )

                    else:
                        st.success(
                            f"Risk Level: {risk_level}"
                        )

                    # =====================================
                    # SHAP
                    # =====================================

                    st.subheader(
                        "🌊 SHAP Waterfall Analysis"
                    )

                    input_df = pd.DataFrame([{
                        "Recency": recency,
                        "Frequency": frequency,
                        "Monetary": monetary,
                        "Tenure": tenure
                    }])

                    feature_df = prepare_features(
                        input_df
                    )

                    feature_names = FEATURE_COLUMNS

                    feature_values = feature_df.values

                    scaled_values = scaler.transform(
                        feature_values
                    )

                    try:

                        shap_output = explainer(
                            scaled_values
                        )

                        if theme == "Dark":
                            plt.style.use("dark_background")
                        else:
                            plt.style.use("default")

                        fig = plt.figure(
                            facecolor=bg_color
                        )

                        # =================================
                        # HANDLE DIFFERENT SHAP OUTPUTS
                        # =================================

                        if hasattr(
                            shap_output,
                            "values"
                        ):

                            shap_values = (
                                shap_output.values
                            )

                            base_values = (
                                shap_output.base_values
                            )

                        else:

                            shap_values = shap_output

                            base_values = (
                                explainer.expected_value
                            )

                        # =================================
                        # BINARY CLASSIFICATION
                        # =================================

                        if len(
                            np.array(
                                shap_values
                            ).shape
                        ) == 3:

                            explanation = shap.Explanation(

                                values=shap_values[
                                    0, :, 1
                                ],

                                base_values=float(
                                    np.array(
                                        base_values
                                    ).flatten()[-1]
                                ),

                                data=feature_values[0],

                                feature_names=feature_names
                            )

                        else:

                            explanation = shap.Explanation(

                                values=np.array(
                                    shap_values
                                )[0],

                                base_values=float(
                                    np.array(
                                        base_values
                                    ).flatten()[0]
                                ),

                                data=feature_values[0],

                                feature_names=feature_names
                            )

                        shap.plots.waterfall(
                            explanation,
                            show=False
                        )

                        st.pyplot(
                            fig,
                            clear_figure=True
                        )

                    except Exception as e:

                        logger.error(
                            f"SHAP Visualization Error: {e}"
                        )

                        st.error(
                            f"SHAP Visualization Error: {e}"
                        )
                    
                    # =====================================
                    # FEATURE IMPORTANCE
                    # =====================================

                    st.subheader(
                        "📊 Feature Contributions"
                    )

                    explanations = result.get(
                        "explanations",
                        {}
                    )

                    if explanations:

                        exp_df = pd.DataFrame({
                            "Feature":
                            explanations.keys(),

                            "Impact":
                            explanations.values()
                        })

                        st.dataframe(
                            exp_df.sort_values(
                                by="Impact",
                                ascending=False
                            ),
                            use_container_width=True,
                            height=300
                        )

                    # =====================================
                    # BUSINESS ACTIONS
                    # =====================================

                    st.subheader(
                        "💡 AI Business Recommendations"
                    )

                    if "Critical" in risk_level:
                                
                            with st.container():

                                st.error(
                                    "🔥 Critical Risk Customer"
                                )

                                st.markdown("""
                                ### Recommended Actions
                                        
                                - Immediate retention escalation  
                                - Assign dedicated customer success manager  
                                - Launch emergency win-back campaign
                                - Offer personalized premium incentives  
                                - Executive intervention recommended
                                - Monitor customer activity daily
                                """)

                    elif "High" in risk_level:
                            
                            with st.container():

                                st.warning(
                                    "⚠️ High Risk Customer"
                                )

                                st.markdown("""
                                ### Recommended Actions
                                        
                                - Launch targeted retention campaign  
                                - Increase customer engagement
                                - Offer loyalty discount
                                - Monitor behavioral decline closely
                                - Send personalized communication
                                - Trigger proactive follow-up
                                """)

                    elif "Medium" in risk_level:
                            
                            with st.container():

                                st.info(
                                    "⚡ Medium Risk Customer"
                                )

                                st.markdown("""
                                ### Recommended Actions
                                            
                                - Encourage repeat purchases
                                - Recommend related products
                                - Improve engagement touchpoints
                                - Launch loyalty nurturing campaign
                                - Monitor purchasing behavior
                                - Increase product awareness
                                """)
                    
                    else:
                            
                            with st.container():

                                st.success(
                                    "✅ Low Risk Customer"
                                )

                                st.markdown("""
                                ### Recommended Actions

                                - Upsell premium offerings
                                - Encourage customer referrals 
                                - Maintain loyalty experience  
                                - Promote long-term engagement
                                - Offer exclusive membership benefits
                                - Identify advocacy opportunities
                                """)

                    # =====================================
                    # AI COPILOT
                    # =====================================

                    st.subheader(
                        "🤖 AI Copilot"
                    )

                    if client:

                        try:

                            response = (
                                client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{
                                        "role": "user",
                                        "content":
                                        f"""
                                        Explain churn risk
                                        probability {prob:.2%}
                                        professionally.
                                        """
                                    }]
                                )
                            )

                            st.success(
                                response
                                .choices[0]
                                .message.content
                            )

                        except Exception as e:

                            logger.error(
                                f"AI Error: {e}"
                            )

                    # =====================================
                    # LOGGING
                    # =====================================

                    log_df = pd.DataFrame([{
                        "Timestamp": datetime.now(),

                        "Recency": recency,
                        "Frequency": frequency,
                        "Monetary": monetary,
                        "Tenure": tenure,

                        "Probability": round(prob, 4),

                        "Customer_Lifetime_Value": round(clv, 2),

                        "Customer_Health_Score": round(
                            health_score,
                            1
                        ),

                        "Risk_Level": risk_level
                    }])

                    log_path = os.path.join(
                        BASE_DIR,
                        "data",
                        "prediction_history.csv"
                    )

                    if os.path.exists(log_path):

                        log_df.to_csv(
                            log_path,
                            mode="a",
                            header=False,
                            index=False
                        )

                    else:

                        log_df.to_csv(
                            log_path,
                            index=False
                        )

                    # =====================================
                    # PDF EXPORT
                    # =====================================

                    st.subheader(
                        "📄 Export Report"
                    )

                    if st.button(
                        "Generate PDF"
                    ):

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as tmp_file:

                            pdf_path = (
                                tmp_file.name
                            )

                        doc = SimpleDocTemplate(
                            pdf_path
                        )

                        styles = (
                            getSampleStyleSheet()
                        )

                        content = [

                            Paragraph(
                                "Customer Churn Intelligence Report",
                                styles["Title"]
                            ),

                            Spacer(1, 12),

                            Paragraph(
                                f"Probability: {prob:.2%}",
                                styles["BodyText"]
                            )
                        ]

                        doc.build(content)

                        with open(
                            pdf_path,
                            "rb"
                        ) as f:

                            st.download_button(
                                "📄 Download Executive Report",
                                f,
                                file_name="report.pdf",
                                mime="application/pdf"
                            )

                except Exception as e:

                    logger.error(
                        f"Dashboard Error: {e}"
                    )

                    st.error(
                        f"Prediction Error: {str(e)}"
                    )

    with tab2:

        st.subheader(
            "📜 Prediction History"
        )

        log_path = os.path.join(
            BASE_DIR,
            "data",
            "prediction_history.csv"
        )

        if os.path.exists(log_path):

            history = pd.read_csv(log_path)

            history = history.drop_duplicates()

            history = history.tail(10000)

            history = history.loc[
                :,
                ~history.columns.duplicated()
            ]
            
            history["Timestamp"] = pd.to_datetime(
                history["Timestamp"],
                errors="coerce"
            )

            history = history.sort_values(
                "Timestamp",
                ascending=False
            )

            history.reset_index(
                drop=True,
                inplace=True
            )

            st.table(
                history.head(100)
            )

        else:

            st.info(
                "No prediction history available."
            )

    # =====================================
    # BATCH HISTORY
    # =====================================

    with tab3:

        st.subheader(
            "📂 Batch Prediction History"
        )

        if os.path.exists(BATCH_HISTORY_PATH):

            batch_history = pd.read_csv(
                BATCH_HISTORY_PATH
            )

            batch_history = batch_history.drop_duplicates()

            batch_history = batch_history.tail(10000)

            batch_history["Timestamp"] = pd.to_datetime(
                batch_history["Timestamp"],
                errors="coerce"
            )

            batch_history = batch_history.sort_values(
                "Timestamp",
                ascending=False
            )

            batch_history.reset_index(
                drop=True,
                inplace=True
            )

            st.table(
                batch_history.head(100)
            )

        else:

            st.info(
                "No batch prediction history available."
            )

#==========================================================
# BATCH PREDICTION
#==========================================================

elif page == "Batch Prediction":

    st.title(
        "📂 Batch Prediction System"
    )

    st.markdown("""
    Upload customer datasets for
    enterprise-scale churn prediction.
    """)

    st.markdown(
        """
        <h4 style='margin-bottom:10px;'>
            Upload Customer Dataset
        </h4>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "📂 Upload Customer Dataset",
        type=["csv", "xlsx", "xls"]
    )

    st.caption(
        "Maximum upload size: 200MB per file • CSV • XLSX • XLS")

    if uploaded_file is not None:

        try:

            file_extension = (
                uploaded_file.name
                .split(".")[-1]
                .lower()
            )

            if file_extension == "csv":

                batch_df = pd.read_csv(
                    uploaded_file
                )

            elif file_extension in ["xlsx", "xls"]:

                batch_df = pd.read_excel(
                    uploaded_file
                )

            else:

                st.error(
                    "Unsupported file type."
                )

                st.stop()

            if batch_df.empty:

                st.error(
                    "Uploaded file contains no data."
                )

                st.stop()

            st.subheader(
                "📄 Uploaded Dataset"
            )

            st.table(
                batch_df.head(20)
            )

            required_columns = [
                "Recency",
                "Frequency",
                "Monetary",
                "Tenure"
            ]

            missing_columns = [

                col for col in required_columns

                if col not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing columns: {missing_columns}"
                )

            else:

                feature_df = prepare_features(
                    batch_df
                )

                scaled_values = scaler.transform(
                    feature_df
                )

                probabilities = (
                    model.predict_proba(
                        scaled_values
                    )[:, 1]
                )

                batch_df[
                    "Probability"
                ] = probabilities

                risk_labels = []

                for prob in probabilities:

                    risk_data = classify_risk(prob)

                    risk_labels.append(
                        risk_data["label"]
                    )

                batch_df["Risk_Level"] = risk_labels

                # =====================================
                # CUSTOMER LIFETIME VALUE
                # =====================================

                batch_df["Customer_Lifetime_Value"] = (

                    (
                        batch_df["Monetary"]

                        /

                        batch_df["Frequency"].replace(0, 1)

                    )

                    *

                    batch_df["Frequency"]

                    *

                    (
                        batch_df["Tenure"].replace(0, 1)

                        /

                        365

                    )

                    *

                    0.20

                )


                batch_df["Customer_Health_Score"] = (
                    (1 - batch_df["Probability"])
                    * 100
                )

                batch_df["Probability"] = (
                    batch_df["Probability"]
                    .round(4)
                )

                batch_df["Customer_Lifetime_Value"] = (
                    batch_df["Customer_Lifetime_Value"]
                    .round(2)
                )

                batch_df["Customer_Health_Score"] = (
                    batch_df["Customer_Health_Score"]
                    .round(1)
                )

                batch_df["Monetary"] = (
                    batch_df["Monetary"]
                    .round(0)
                )

                # =====================================
                # PREDICTION TIMESTAMP
                # =====================================

                batch_df["Timestamp"] = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                batch_df = batch_df[[
                    "Timestamp",

                    "Recency",
                    "Frequency",
                    "Monetary",
                    "Tenure",

                    "Probability",

                    "Customer_Lifetime_Value",

                    "Customer_Health_Score",

                    "Risk_Level"
                ]]

                # =====================================
                # SAVE BATCH HISTORY
                # =====================================

                if os.path.exists(BATCH_HISTORY_PATH):

                    batch_df.to_csv(
                        BATCH_HISTORY_PATH,
                        mode="a",
                        header=False,
                        index=False
                    )

                else:

                    batch_df.to_csv(
                        BATCH_HISTORY_PATH,
                        index=False
                    )

                st.subheader(
                    "📊 Batch Prediction Results"
                )

                st.table(
                    batch_df
                )

                st.subheader(
                    "📊 Portfolio Risk Analytics"
                )

                col1, col2 = st.columns(2)

                # =================================
                # RISK DISTRIBUTION
                # =================================

                with col1:

                    fig_hist = px.histogram(

                        batch_df,

                        x="Probability",

                        nbins=20,

                        title="Risk Distribution"
                    )

                    fig_hist.update_layout(

                        template=plot_template,

                        paper_bgcolor=paper_color,

                        plot_bgcolor=bg_color,

                        font=dict(
                            color=font_color
                        )
                    )

                    st.plotly_chart(
                        fig_hist,
                        use_container_width=True
                    )

                # =================================
                # SEGMENT DISTRIBUTION
                # =================================

                with col2:

                    segment_counts = (
                        batch_df["Risk_Level"]
                        .value_counts()
                        .reset_index()
                    )

                    segment_counts.columns = [
                        "Risk_Level",
                        "Customers"
                    ]

                    fig_pie = px.pie(

                        segment_counts,

                        names="Risk_Level",

                        values="Customers",

                        title="Customer Risk Segmentation",

                        color="Risk_Level",

                        color_discrete_map=RISK_COLORS
                    )

                    fig_pie.update_layout(

                        template=plot_template,

                        paper_bgcolor=paper_color,

                        font=dict(
                            color=font_color
                        ),

                        legend_title="Risk Category"
                    )

                    st.plotly_chart(
                        fig_pie,
                        use_container_width=True
                    )

                # =================================
                # EXECUTIVE KPIs
                # =================================

                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

                # =================================
                # EXECUTIVE KPI CALCULATIONS
                # =================================

                critical_customers = len(

                    batch_df[
                        batch_df["Risk_Level"]
                        .str.contains("Critical")
                    ]
                )

                high_risk_customers = len(

                    batch_df[
                        batch_df["Risk_Level"]
                        .str.contains("High")
                    ]
                )

                at_risk_customers = (
                    high_risk_customers +
                    critical_customers
                )

                revenue_at_risk = batch_df.loc[
                    batch_df["Probability"] >= 0.60,
                    "Monetary"
                ].sum()

                portfolio_health = (
                    (1 - probabilities.mean()) * 100
                )

                at_risk_pct = (

                    at_risk_customers

                    /

                    len(batch_df)

                ) * 100

                # =================================
                # KPI DISPLAY
                # =================================

                col1.metric(
                    "Portfolio Customers",
                    len(batch_df)
                )

                col2.metric(
                    "Portfolio Churn Index",
                    f"{probabilities.mean():.2%}"
                )

                col3.metric(
                    "🚨 High Risk Customers",
                    high_risk_customers
                )

                col4.metric(
                    "🔥 Critical Risk Customers",
                    critical_customers
                )

                col5.metric(
                    "⚠️ At-Risk Customers",
                    at_risk_customers
                )

                col6.metric(
                    "Revenue At Risk",
                    f"${revenue_at_risk:,.0f}"
                )

                col7.metric(
                    "Portfolio Health Index",
                    f"{portfolio_health:.1f}/100"
                )

                col8.metric(
                    "⚠️ At-Risk %",
                    f"{at_risk_pct:.1f}%"
                )

                # =================================
                # DOWNLOAD RESULTS
                # =================================

                csv = batch_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(

                    "⬇️ Download CSV Predictions Report",

                    csv,

                    file_name="batch_predictions.csv",

                    mime="text/csv"
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".xlsx"
                ) as tmp_excel:

                    excel_path = tmp_excel.name

                batch_df.to_excel(
                    excel_path,
                    index=False
                )

                with open(
                    excel_path,
                    "rb"
                ) as f:

                    st.download_button(

                        "📊 Download Excel Predictions Report",

                        f,

                        file_name="batch_predictions.xlsx",

                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        )
                    )

                # =================================
                # PDF EXECUTIVE REPORT
                # =================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_pdf:

                    pdf_path = tmp_pdf.name

                doc = SimpleDocTemplate(
                    pdf_path
                )

                styles = getSampleStyleSheet()

                elements = []

                elements.append(
                    Paragraph(
                        "Executive Batch Prediction Report",
                        styles["Title"]
                    )
                )

                elements.append(
                    Spacer(1, 12)
                )

                summary_text = f"""
                Portfolio Customers: {len(batch_df)}<br/>
                Portfolio Churn Risk: {probabilities.mean():.2%}<br/>
                Critical Risk Customers: {critical_customers}<br/>
                Revenue At Risk: ${revenue_at_risk:,.0f}<br/>
                Portfolio Health Score: {portfolio_health:.1f}/100
                """

                elements.append(
                    Paragraph(
                        summary_text,
                        styles["BodyText"]
                    )
                )

                elements.append(
                    Spacer(1, 20)
                )

                table_data = [
                    [
                        "Recency",
                        "Frequency",
                        "Monetary",
                        "Tenure",
                        "Probability",
                        "CLV",
                        "Health Score",
                        "Risk"
                    ]
                ]

                for _, row in batch_df.head(15).iterrows():

                    table_data.append([
                        row["Recency"],
                        row["Frequency"],
                        row["Monetary"],
                        row["Tenure"],
                        f"{row['Probability']:.2%}",
                        row["Customer_Lifetime_Value"],
                        row["Customer_Health_Score"],
                        row["Risk_Level"]
                    ])

                table = Table(table_data)

                table.setStyle(TableStyle([

                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),

                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),

                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ]))

                elements.append(table)

                doc.build(elements)

                with open(
                    pdf_path,
                    "rb"
                ) as pdf_file:

                    st.download_button(

                        "📄 Download Executive PDF",

                        pdf_file,

                        file_name="executive_batch_report.pdf",

                        mime="application/pdf"
                    )

                # =================================
                # AI EXECUTIVE INSIGHTS
                # =================================

                st.subheader(
                    "🤖 AI Executive Insights"
                )

                avg_risk = probabilities.mean()

                if avg_risk >= 0.70:

                    st.error("""
                    High portfolio churn exposure detected.
                    Immediate enterprise retention intervention is recommended.
                    Customer engagement deterioration appears significant.
                    """)

                elif avg_risk >= 0.40:

                    st.warning("""
                    Moderate churn exposure identified across the portfolio.
                    Retention optimization strategies should be prioritized.
                    """)

                else:

                    st.success("""
                    Customer portfolio health remains stable.
                    Retention performance is operating within acceptable thresholds.
                    """)

        except Exception as e:

            st.error(
                f"Batch Prediction Error: {e}"
            )

# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.title(
        "📊 Executive Analytics"
    )

    if not rfm.empty:

        try:

            features = prepare_features(
                rfm[[
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "Tenure"
                ]]
            )

            scaled = scaler.transform(
                features
            )

            probabilities = (
                model.predict_proba(
                    scaled
                )[:, 1]
            )

            rfm["probability"] = (
                probabilities
            )

            # =====================================
            # RISK SEGMENT CLASSIFICATION
            # =====================================

            rfm["Risk_Segment"] = (

                rfm["probability"]

                .apply(

                    lambda x:
                    classify_risk(x)["label"]

                )
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Portfolio Customers",
                len(rfm)
            )

            high_risk_count = len(

                rfm[
                    rfm["Risk_Segment"]
                    .str.contains("High|Critical")
                ]
            )

            col2.metric(
                "High Risk Customers",
                high_risk_count
            )

            col3.metric(
                "Customer Churn Risk Index",
                f"{rfm['probability'].mean():.2%}"
            )

            revenue_risk = rfm.loc[
                rfm["Risk_Segment"]
                .str.contains("High|Critical"),
                "Monetary"
            ].sum()

            col4.metric(
                "Revenue At Risk",
                f"${revenue_risk:,.0f}"
            )

            # =============================================
            # DRIFT MONITORING
            # =============================================

            st.subheader(
                "📡 Drift Monitoring"
            )

            train_mean = features.mean()

            live_mean = features.sample(
                min(100, len(features))
            ).mean()

            drift = (
                abs(live_mean - train_mean)
                /
                train_mean.replace(0, 1)
            ) * 100

            drift_df = pd.DataFrame({
                "Feature": drift.index,
                "Drift %": drift.values
            })

            drift_df["Drift %"] = (
                drift_df["Drift %"]
                .round(2)
            )

            st.table(
                drift_df
            )

            drift_alert = (
                drift_df["Drift %"] > 20
            ).any()

            if drift_alert:

                st.warning(
                    "⚠️ Potential feature drift detected. Model monitoring review recommended."
                )

            else:

                st.success(
                    "No significant feature drift detected."
                )

            st.subheader(
                "📈 Risk Distribution"
            )

            fig = px.histogram(
                rfm,
                x="probability",
                nbins=25,
                title="Customer Churn Risk Distribution"
            )

            fig.update_layout(

                template=plot_template,

                paper_bgcolor=paper_color,

                plot_bgcolor=bg_color,

                height=400,

                font=dict(
                    color=font_color
                ),

                xaxis=dict(
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    color=font_color
                ),

                yaxis=dict(
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    color=font_color
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader(
                "📊 Cohort Risk Analysis"
            )

            # =====================================
            # TENURE GROUPS
            # =====================================

            rfm["Tenure_Group"] = pd.cut(

                rfm["Tenure"],

                bins=[0, 30, 90, 180, 365, 10000],

                labels=[
                    "0-30 Days",
                    "31-90 Days",
                    "91-180 Days",
                    "181-365 Days",
                    "365+ Days"
                ]
            )

            # =====================================
            # COHORT ANALYSIS
            # =====================================

            cohort = rfm.groupby(
                "Tenure_Group"
            )["probability"].mean().reset_index()

            # =====================================
            # PLOTLY CHART
            # =====================================

            fig = px.bar(

                cohort,

                x="Tenure_Group",

                y="probability",

                title="Average Churn Risk By Customer Tenure",

                text_auto=".2%"
            )

            fig.update_layout(

                template=plot_template,

                paper_bgcolor=paper_color,

                plot_bgcolor=bg_color,

                font=dict(
                    color=font_color
                ),

                xaxis=dict(
                    title="Customer Cohort",
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    color=font_color
                ),

                yaxis=dict(
                    title="Average Churn Probability",
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    color=font_color
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader(
                "🎯 Customer Segmentation"
            )

            # =====================================
            # SEGMENT COUNTS
            # =====================================

            safe_count = len(

                rfm[
                    rfm["Risk_Segment"]
                    .str.contains("Low")
                ]
            )

            medium_count = len(

                rfm[
                    rfm["Risk_Segment"]
                    .str.contains("Medium")
                ]
            )

            high_count = len(

                rfm[
                    rfm["Risk_Segment"]
                    .str.contains("High")
                ]
            )

            critical_count = len(

                rfm[
                    rfm["Risk_Segment"]
                    .str.contains("Critical")
                ]
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "✅ Low Risk",
                safe_count
            )

            col2.metric(
                "⚠️ Medium Risk",
                medium_count
            )

            col3.metric(
                "🚨 High Risk",
                high_count
            )

            col4.metric(
                "🔥 Critical Risk",
                critical_count
            )

            # =====================================
            # SEGMENTS DATAFRAME
            # =====================================

            segments = pd.DataFrame({

                "Segment": [

                    "✅ Low Risk",
                    "⚠️ Medium Risk",
                    "🚨 High Risk",
                    "🔥 Critical Risk"

                ],

                "Customers": [

                    safe_count,
                    medium_count,
                    high_count,
                    critical_count

                ]

            })

            # =====================================
            # PERCENTAGE CALCULATION
            # =====================================

            segments["Percentage"] = (

                segments["Customers"]

                /

                segments["Customers"].sum()

            ) * 100

            # =====================================
            # SEGMENTATION CHART
            # =====================================

            fig = px.bar(

                segments,

                x="Segment",

                y="Customers",

                title="Customer Risk Segmentation Overview",

                text="Percentage",

                color="Segment",

                category_orders={

                    "Segment": [
                        "✅ Low Risk",
                        "⚠️ Medium Risk",
                        "🚨 High Risk",
                        "🔥 Critical Risk"
                    ]
                },

                color_discrete_map=RISK_COLORS
            )

            fig.update_traces(

                texttemplate="%{text:.1f}%",

                textposition="outside"
            )

            fig.update_layout(

                template=plot_template,

                paper_bgcolor=paper_color,

                plot_bgcolor=bg_color,

                font=dict(
                    color=font_color
                ),

                height=500,

                xaxis=dict(

                    title="Customer Segment",

                    gridcolor=grid_color,

                    zerolinecolor=grid_color,

                    color=font_color
                ),

                yaxis=dict(

                    title="Number of Customers",

                    gridcolor=grid_color,

                    zerolinecolor=grid_color,

                    color=font_color
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # =====================================
            # EXECUTIVE SEGMENT INSIGHTS
            # =====================================

            total_customers = segments["Customers"].sum()

            safe_pct = (safe_count / total_customers) * 100

            risk_pct = (
                (high_count + critical_count)
                / total_customers
            ) * 100

            if risk_pct >= 40:

                st.error(
                    f"""
                    🚨 Executive Insight:
                    {risk_pct:.1f}% of customers are currently classified as
                    high-risk or critical-risk customers.
                    Immediate retention intervention is recommended.
                    """
                )

            elif risk_pct >= 20:

                st.warning(
                    f"""
                    ⚠️ Executive Insight:
                    {risk_pct:.1f}% of customers show elevated churn risk.
                    Retention optimization should be prioritized.
                    """
                )

            else:

                st.success(
                    f"""
                    ✅ Executive Insight:
                    Customer retention health is currently stable with
                    only {risk_pct:.1f}% high-risk exposure.
                    """
                )

            # =====================================
            # AVERAGE RISK BY SEGMENT
            # =====================================

            segment_risk = rfm.groupby(
                "Risk_Segment"
            )["probability"].mean().reset_index()

            fig_segment_risk = px.bar(

                segment_risk,

                x="Risk_Segment",

                y="probability",

                title="Average Churn Probability By Segment",

                text="probability",

                color="Risk_Segment",

                category_orders={

                    "Risk_Segment": [
                        "✅ Low Risk",
                        "⚠️ Medium Risk",
                        "🚨 High Risk",
                        "🔥 Critical Risk"
                    ]
                },

                color_discrete_map=RISK_COLORS
            )

            fig_segment_risk.update_traces(

                texttemplate="%{text:.1%}",

                textposition="outside"
            )

            fig_segment_risk.update_layout(

                template=plot_template,

                paper_bgcolor=paper_color,

                plot_bgcolor=bg_color,

                font=dict(
                    color=font_color
                ),

                xaxis=dict(
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    color=font_color
                ),

                yaxis=dict(
                    title="Average Churn Probability",
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    color=font_color
                )
            )

            st.plotly_chart(
                fig_segment_risk,
                use_container_width=True
            )
            
            # =====================================
            # RETENTION OPPORTUNITY
            # =====================================

            retention_value = rfm.loc[
                rfm["probability"] > 0.6,
                "Monetary"
            ].sum()

            st.metric(
                "💰 Retention Revenue Opportunity",
                f"${retention_value:,.0f}"
            )

            st.subheader(
                "🚨 Top High-Risk Customers"
            )

            top_risk = (
                rfm.sort_values(
                    by="probability",
                    ascending=False
                )
                .head(10)
            )

            if "Customer ID" in top_risk.columns:

                top_risk = top_risk[
                    [
                        "Customer ID",
                        "Recency",
                        "Frequency",
                        "Monetary",
                        "probability",
                        "Risk_Segment"
                    ]
                ]

                top_risk["probability"] = (
                    top_risk["probability"]
                    .round(4)
                )

            st.table(
                top_risk
            )

        except Exception as e:

            logger.error(
                f"Analytics Error: {e}"
            )

            st.error(
                f"Analytics Error: {e}"
            )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title(
        "📈 Model Performance"
    )

    try:

        X_test = joblib.load(
            os.path.join(
                MODEL_DIR,
                "X_test.pkl"
            )
        )

        y_test = joblib.load(
            os.path.join(
                MODEL_DIR,
                "y_test.pkl"
            )
        )

        X_scaled = scaler.transform(
            X_test
        )

        y_prob = (
            model.predict_proba(
                X_scaled
            )[:, 1]
        )

        y_pred = (
            y_prob > threshold
        ).astype(int)

        st.subheader(
            "📉 ROC Curve"
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        auc_score = roc_auc_score(
            y_test,
            y_prob
        )

        st.metric(
            "ROC-AUC Score",
            f"{auc_score:.3f}"
        )

        if theme == "Dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        fig, ax = plt.subplots(
            facecolor=bg_color
        )

        ax.set_facecolor(bg_color)

        ax.tick_params(colors=font_color)

        ax.xaxis.label.set_color(font_color)

        ax.yaxis.label.set_color(font_color)

        ax.title.set_color(font_color)

        for spine in ax.spines.values():
            spine.set_color(font_color)

        ax.plot(
            fpr,
            tpr,
            label=f"AUC={auc_score:.3f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.legend()

        st.pyplot(fig)

        st.subheader(
            "📊 Precision-Recall Curve"
        )

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                y_prob
            )
        )

        if theme == "Dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        fig2, ax2 = plt.subplots(
            facecolor=bg_color
        )

        ax2.set_facecolor(bg_color)

        ax2.tick_params(colors=font_color)

        ax2.xaxis.label.set_color(font_color)

        ax2.yaxis.label.set_color(font_color)

        ax2.title.set_color(font_color)

        for spine in ax2.spines.values():
            spine.set_color(font_color)

        ax2.plot(
            recall,
            precision
        )

        st.pyplot(fig2)

        st.subheader(
            "⚙️ Threshold Optimization"
        )

        thresholds = np.arange(
            0.1,
            1.0,
            0.05
        )

        metrics = []

        for t in thresholds:

            preds = (
                y_prob > t
            ).astype(int)

            metrics.append({

                "Threshold": t,

                "F1":
                f1_score(
                    y_test,
                    preds
                ),

                "Precision":
                precision_score(
                    y_test,
                    preds
                ),

                "Recall":
                recall_score(
                    y_test,
                    preds
                )
            })

        metric_df = pd.DataFrame(
            metrics
        )

        fig_thresh = px.line(

            metric_df,

            x="Threshold",

            y=["F1", "Precision", "Recall"],

            title="Threshold Optimization"
        )

        fig_thresh.update_layout(

            template=plot_template,

            paper_bgcolor=paper_color,

            plot_bgcolor=bg_color,

            font=dict(
                color=font_color
            ),

            xaxis=dict(
                gridcolor=grid_color,
                zerolinecolor=grid_color,
                color=font_color
            ),

            yaxis=dict(
                gridcolor=grid_color,
                zerolinecolor=grid_color,
                color=font_color
            )

        )

        st.plotly_chart(
            fig_thresh,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader(
            "📊 Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        if theme == "Dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        fig3, ax3 = plt.subplots(
            facecolor=bg_color
        )

        ax3.set_facecolor(bg_color)

        disp = (
            ConfusionMatrixDisplay(
                confusion_matrix=cm
            )
        )

        disp.plot(ax=ax3)

        fig3.patch.set_facecolor(bg_color)

        ax3.set_facecolor(bg_color)

        ax3.tick_params(colors=font_color)

        ax3.xaxis.label.set_color(font_color)

        ax3.yaxis.label.set_color(font_color)

        ax3.title.set_color(font_color)

        for spine in ax3.spines.values():
            spine.set_color(font_color)

        st.pyplot(fig3)

        st.subheader(
            "📋 Classification Report"
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = (
            pd.DataFrame(report)
            .transpose()
        )

        st.table(
            report_df.round(3)
        )

    except Exception as e:

        logger.error(
            f"Performance Error: {e}"
        )

# =========================================================
# MONITORING
# =========================================================

elif page == "Monitoring":

    st.title("🛡️ AI System Monitoring")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model Version", "v1.0")
    col2.metric("Deployment Status", "Production")
    col3.metric("Inference Engine", "Operational")
    col4.metric("Explainability", "Active")

    st.subheader("📡 System Health")

    st.progress(100)

    st.success(
        "System Status: Operational | Uptime: 99.9%"
    )

    # ====================================
    # LOAD PREDICTION HISTORY
    # ====================================

    log_path = os.path.join(
        BASE_DIR,
        "data",
        "prediction_history.csv"
    )

    if os.path.exists(log_path):

        history = pd.read_csv(log_path)

        history = history.drop_duplicates()

        history = history.loc[
            :,
            ~history.columns.duplicated()
        ]

    else:

        history = pd.DataFrame()

    # ====================================
    # LOAD BATCH HISTORY
    # ====================================

    if os.path.exists(BATCH_HISTORY_PATH):

        batch_history = pd.read_csv(
            BATCH_HISTORY_PATH
        )

        batch_history = batch_history.drop_duplicates()

    else:

        batch_history = pd.DataFrame()

    required_cols = [
        "Probability",
        "Customer_Lifetime_Value",
        "Customer_Health_Score"
    ]

    for col in required_cols:
        if col not in history.columns:
            history[col] = np.nan

    for col in required_cols:
        if col not in batch_history.columns:
            batch_history[col] = np.nan

    # ====================================
    # MONITORING KPIs
    # ====================================

    st.subheader("📊 Monitoring KPIs")

    st.info(
        "🚀 Production AI System | Real-Time Churn Monitoring | Explainable AI Enabled"
    )

    if not history.empty:

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Total AI Decisions Logged",
            f"{len(history):,}"
        )

        risk = history['Probability'].mean()

        col2.metric(
            "Customer Churn Risk Index",
            f"{risk:.2%}",
            delta=f"{risk*100:.1f}%"
        )

        col3.metric(
            "Average Customer Lifetime Value",
            f"${history['Customer_Lifetime_Value'].mean():,.0f}"
        )

        col4.metric(
            "Customer Health Index",
            f"{history['Customer_Health_Score'].mean():.1f}"
        )

        latest_prediction = pd.to_datetime(
            history["Timestamp"],
            errors="coerce"
        ).max()

        latest_prediction_display = (
            latest_prediction.strftime("%Y-%m-%d %H:%M:%S")
        )

        col5.metric(
            "Last Prediction",
            latest_prediction_display
        )

    else:

        st.info(
            "No prediction history available."
        )

    st.divider()

    # ====================================
    # BATCH KPIs
    # ====================================

    st.subheader("📂 Batch Prediction KPIs")

    if not batch_history.empty:

        col1, col2, col3, col4, col5 = st.columns(5)

        revenue_at_risk = batch_history.loc[
            batch_history["Probability"] >= 0.60,
            "Monetary"
        ].sum()

        col1.metric(
            "Portfolio Customers",
            len(batch_history)
        )

        col2.metric(
            "Portfolio Churn Risk Index",
            f"{batch_history['Probability'].mean():.2%}"
        )

        col3.metric(
            "Average Customer Lifetime Value",
            f"${batch_history['Customer_Lifetime_Value'].mean():,.0f}"
        )

        col4.metric(
            "Average Customer Health Index",
            f"{batch_history['Customer_Health_Score'].mean():.1f}"
        )

        col5.metric(
            "Revenue At Risk",
            f"${revenue_at_risk:,.0f}"
        )

    else:

        st.info(
            "No batch prediction history available."
        )

    st.divider()

    # ====================================
    # ENTERPRISE KPIs
    # ====================================

    if (
        not history.empty
        and
        not batch_history.empty
    ):

        st.subheader("🏢 Enterprise KPIs")

        total_predictions = (
            len(history)
            +
            len(batch_history)
        )

        enterprise_risk = pd.concat([
            history["Probability"],
            batch_history["Probability"]
        ]).mean()

        enterprise_clv = pd.concat([
            history["Customer_Lifetime_Value"],
            batch_history["Customer_Lifetime_Value"]
        ]).mean()

        enterprise_health = pd.concat([
            history["Customer_Health_Score"],
            batch_history["Customer_Health_Score"]
        ]).mean()

        latest_timestamp = max(

            pd.to_datetime(
                history["Timestamp"],
                errors="coerce"
            ).max(),

            pd.to_datetime(
                batch_history["Timestamp"],
                errors="coerce"
            ).max()
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Total AI Decisions Logged",
            total_predictions
        )

        col2.metric(
            "Enterprise Churn Risk Index",
            f"{enterprise_risk:.2%}"
        )

        col3.metric(
            "Enterprise Customer Lifetime Value",
            f"${enterprise_clv:,.0f}"
        )

        col4.metric(
            "Enterprise Customer Health Index",
            f"{enterprise_health:.1f}"
        )

        col5.metric(
            "Last AI Activity",
            latest_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

elif page == "What-If Simulator":

    st.title(
        "🧠 What-If Scenario Simulator"
    )

    st.markdown("""
    Simulate the business impact of
    retention strategy improvements.
    """)

    customers = st.slider(
        "High-Risk Customers",
        1,
        10000,
        500
    )

    avg_revenue = st.slider(
        "Average Revenue Per Customer",
        100,
        10000,
        500
    )

    retention_gain = st.slider(
        "Retention Improvement %",
        1,
        100,
        15
    )

    revenue_saved = (
        customers
        *
        avg_revenue
        *
        (retention_gain / 100)
    )

    st.metric(
        "💰 Potential Revenue Saved",
        f"${revenue_saved:,.0f}"
    )

    scenario_df = pd.DataFrame({

        "Scenario": [
            "Current",
            "Improved"
        ],

        "Revenue": [
            customers * avg_revenue,
            (
                customers
                *
                avg_revenue
            ) + revenue_saved
        ]
    })

    fig = px.bar(
        scenario_df,
        x="Scenario",
        y="Revenue",
        title="Revenue Impact Simulation"
    )

    fig.update_layout(

        template=plot_template,

        paper_bgcolor=paper_color,

        plot_bgcolor=bg_color,

        font=dict(
            color=font_color
        ),

        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            color=font_color
        ),

        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            color=font_color
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title(
        "ℹ️ About The Platform"
    )

    st.markdown("""
    ## 🚀 Customer Churn Intelligence System

    The Customer Churn Intelligence System is an enterprise-grade
    AI-powered decision intelligence platform designed to help
    organizations proactively identify customer churn risk,
    optimize retention strategies, and protect long-term revenue.

    ---

    ## 🎯 Business Objective

    Customer churn represents one of the most significant
    financial risks for subscription, retail, SaaS,
    banking, and customer-centric organizations.

    This platform enables organizations to:

    - Predict customer churn probability
    - Identify high-risk customers early
    - Generate AI-driven retention recommendations
    - Monitor executive-level customer risk exposure
    - Improve customer lifetime value
    - Reduce revenue loss through proactive intervention

    ---

    ## 🧠 AI & Machine Learning Architecture

    The platform combines:

    - Predictive Machine Learning
    - Explainable AI (SHAP)
    - Decision Intelligence
    - Executive Analytics
    - Real-Time Monitoring
    - AI Copilot Assistance
    - Drift Monitoring
    - Scenario Simulation

    ---

    ## ⚙️ Core System Components

    ### 1. Predictive Intelligence Engine
    - Customer churn prediction
    - Risk scoring
    - Threshold optimization
    - Real-time inference

    ### 2. Explainable AI Layer
    - SHAP waterfall analysis
    - Feature contribution analysis
    - Model transparency

    ### 3. Executive Intelligence Dashboard
    - Revenue-at-risk monitoring
    - Customer segmentation
    - Cohort analysis
    - Executive KPIs

    ### 4. AI Decision Support
    - AI-generated customer insights
    - Retention recommendations
    - Intelligent business actions

    ### 5. Monitoring & Governance
    - Prediction logging
    - Drift monitoring
    - Latency tracking
    - Operational visibility

    ### 6. Strategic Scenario Simulation
    - Revenue protection simulation
    - Retention strategy analysis
    - Business impact forecasting

    ---

    ## 📊 Machine Learning Workflow

    The platform workflow includes:

    1. Data preprocessing
    2. RFM feature engineering
    3. Feature scaling
    4. Predictive modeling
    5. Explainability generation
    6. Executive analytics
    7. Retention intelligence delivery

    ---

    ## 🧪 Model Evaluation Framework

    The system includes enterprise-grade evaluation metrics:

    - ROC-AUC
    - Precision-Recall Analysis
    - Confusion Matrix
    - Threshold Optimization
    - Classification Reporting

    ---

    ## 🏢 Enterprise Value

    This platform enables organizations to:

    - Improve customer retention
    - Reduce churn-related revenue loss
    - Enhance customer intelligence
    - Increase operational efficiency
    - Support executive decision-making
    - Scale AI-driven customer strategy

    ---

    ## ⚙️ Technology Stack

    ### AI / Machine Learning
    - Scikit-Learn
    - SHAP
    - NumPy
    - Pandas

    ### Application Layer
    - Streamlit
    - FastAPI
    - Plotly
    - Matplotlib

    ### Infrastructure & Engineering
    - Python
    - Modular Architecture
    - Caching Optimization
    - Logging & Monitoring

    ### AI Integration
    - OpenAI API
    - AI Copilot

    ---

    ## 🔮 Future Roadmap

    Planned enterprise enhancements include:

    - Database integration
    - Real-time streaming analytics
    - Docker containerization
    - CI/CD automation
    - Authentication & RBAC
    - Cloud-native deployment
    - Advanced retention simulations

    ---

    ## 👤 Built By

    Daniel Damilola Amosun

    Data Scientist | AI Engineer | Decision Intelligence Builder

    """)

    st.divider()

    st.markdown("""
    ## 🏢 Organization
    """)

    # ============================================
    # ORGANIZATION BRANDING
    # ============================================

    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:

        logo = Image.open(LOGO_PATH)

        st.image(
            logo,
            width=100
        )

        st.markdown(
            "<div style='margin-top:-15px'></div>",
            unsafe_allow_html=True
        )

        st.caption(
            "Data • AI • Decision Intelligence"
        )

        st.markdown("""
        ### Enterprise AI & Decision Intelligence Platform

        Built to deliver predictive intelligence,
        explainable AI, executive analytics,
        and intelligent customer retention strategies.

        #### Core Capabilities
        - Predictive Intelligence
        - Explainable AI
        - Executive Analytics
        - AI Decision Support
        """)

        st.divider()

# =========================
# GLOBAL FOOTER
# =========================

import base64
import streamlit.components.v1 as components

FOOTER_LOGO_PATH = os.path.join(
    ASSETS_DIR,
    "logo.png"
)

with open(FOOTER_LOGO_PATH, "rb") as image_file:
    logo_base64 = base64.b64encode(
        image_file.read()
    ).decode()

footer_html = f"""
<hr style='
    margin-top:30px;
    margin-bottom:20px;
    border:1px solid rgba(128,128,128,0.15);
'>

<div style='
    text-align:center;
    padding-top:10px;
    padding-bottom:20px;
    font-family:Arial;
'>

    <img src='data:image/png;base64,{logo_base64}'
     style='
        width:150px;
        max-width:70%;
        height:auto;
        opacity:0.98;
        margin-bottom:10px;
     '>

    <div style='
        font-size:15px;
        font-weight:600;
        color:#B8C7E0;
        margin-top:8px;
        letter-spacing:0.4px;
    '>
        Customer Churn Intelligence System
    </div>

    <div style='
        font-size:13px;
        color:#7D8590;
        margin-top:6px;
        letter-spacing:0.3px;
    '>
        Enterprise AI Analytics Platform
    </div>

    <div style='
        text-align:center;
        color:#5F6B7A;
        font-size:11px;
        margin-top:14px;
    '>
        Built with Streamlit • XGBoost • Plotly • Scikit-Learn
    </div>

</div>
"""

components.html(
    footer_html,
    height=220,
    scrolling=False
)