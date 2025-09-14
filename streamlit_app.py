import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical Device Risk Assessment Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.fda.gov/medical-devices',
        'Report a bug': None,
        'About': "Medical Device Risk Assessment Platform v2.0 - Powered by Advanced Machine Learning"
    }
)

# Professional CSS styling for a dark theme
st.markdown("""
<style>
    /* Dark background with subtle pattern */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* A dark, semi-transparent overlay to keep content readable */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(1px);
        z-index: -1;
    }
    
    /* Dark main content area */
    .main .block-container {
        background: #1e1e2d;
        border-radius: 12px;
        padding: 2.5rem;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid #333;
        max-width: 1200px;
    }
    
    /* Dark sidebar */
    .css-1d391kg {
        background: #1e1e2d;
        border-radius: 12px;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid #333;
    }
    
    /* Light header text for dark theme */
    .main-header, .sub-header {
        color: #f5f5f5;
    }
    
    /* Risk level cards - colors can remain as they are distinct */
    .risk-high {
        background: #dc2626;
        color: white;
        border: none;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
        border-left: 4px solid #b91c1c;
    }
    
    .risk-medium {
        background: #d97706;
        color: white;
        border: none;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.2);
        border-left: 4px solid #b45309;
    }
    
    .risk-low {
        background: #059669;
        color: white;
        border: none;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.2);
        border-left: 4px solid #047857;
    }
    
    /* Dark metric cards */
    .metric-card {
        background: #2b2b40;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
        transition: box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    /* Dark input styling */
    .stSelectbox > div > div {
        background: #2b2b40;
        border-radius: 6px;
        border: 1px solid #555;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        color: white;
    }
    
    /* Professional button styling can remain */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Professional progress bar can remain */
    .stProgress > div > div > div > div {
        background: #2563eb;
        border-radius: 4px;
    }
    
    /* Light typography for dark theme */
    h1, h2, h3, h4, p, span, li, strong {
        color: #f5f5f5 !important;
    }
    
    /* Dark footer */
    .footer {
        background: #2b2b40;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid #333;
        color: #c0c0c0;
    }
    
    /* Dark sidebar header */
    .sidebar-header {
        background: #2563eb;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    
    /* Dark info card */
    .info-card {
        background: #2b2b40;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid #333;
        color: #c0c0c0;
    }
</style>
""", unsafe_allow_html=True)

DATA_URL = 'https://github.com/thrishi0610/medical-device-risk-app/releases/download/v1/final_merged_dataset.csv'
@st.cache_data
def load_data():
    """Load the dataset for autocomplete suggestions"""
    try:
        df = pd.read_csv(DATA_URL, low_memory=False)
        # Get unique device names and manufacturers
        device_names = df['name'].dropna().unique().tolist()
        manufacturers = df['name_manufacturer'].dropna().unique().tolist()
        return device_names, manufacturers
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], []

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    try:
        # Load the balanced 2-feature model (properly balanced predictions)
        model = XGBClassifier()
        model.load_model("xgbModel_balanced_2feat.model")
        
        # Load balanced label encoders
        le_device = joblib.load("le_device_balanced.pkl")
        le_manuf = joblib.load("le_manuf_balanced.pkl")
        
        return model, le_device, le_manuf
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_risk(device_name, manufacturer_name, model, le_device, le_manuf):
    """Predict risk level for given device and manufacturer"""
    try:
        # Encode inputs
        if device_name in le_device.classes_:
            device_code = le_device.transform([device_name])[0]
        else:
            device_code = -1 # Unknown device
        
        if manufacturer_name in le_manuf.classes_:
            manuf_code = le_manuf.transform([manufacturer_name])[0]
        else:
            manuf_code = -1 # Unknown manufacturer
        
        # Create sample for prediction
        sample = pd.DataFrame([[device_code, manuf_code]], 
                              columns=['name', 'name_manufacturer'])
        
        # Predict
        pred_encoded = model.predict(sample)[0]
        
        # Map back to original labels (0→1, 1→2, 2→3)
        risk_mapping = {0: 1, 1: 2, 2: 3}
        risk_level = risk_mapping[pred_encoded]
        
        return risk_level
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def get_risk_display(risk_level):
    """Get risk level display information"""
    risk_info = {
        1: {"label": "HIGH RISK", "color": "red", "icon": "🔴", "description": "High risk devices require immediate attention and careful monitoring."},
        2: {"label": "MEDIUM RISK", "color": "orange", "icon": "🟡", "description": "Medium risk devices should be monitored regularly."},
        3: {"label": "LOW RISK", "color": "green", "icon": "🟢", "description": "Low risk devices have minimal safety concerns."}
    }
    return risk_info.get(risk_level, {"label": "UNKNOWN", "color": "gray", "icon": "❓", "description": "Risk level could not be determined."})

def main():
    # Professional Header
    st.markdown('<h1 class="main-header">Medical Device Risk Assessment Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered risk analysis for medical device safety and compliance</p>', unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("Loading data and model..."):
        device_names, manufacturers = load_data()
        model, le_device, le_manuf = load_model_and_encoders()
    
    if not device_names or not manufacturers or model is None:
        st.error("Failed to load required data or model. Please ensure all files are present.")
        return
    
    # Professional Sidebar Header
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2 style="color: white; margin: 0; text-align: center; font-size: 1.5rem;">Device Information</h2>
        <p style="color: rgba
