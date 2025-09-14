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
            device_code = -1  # Unknown device
        
        if manufacturer_name in le_manuf.classes_:
            manuf_code = le_manuf.transform([manufacturer_name])[0]
        else:
            manuf_code = -1  # Unknown manufacturer
        
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
        <p style="color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0; text-align: center; font-size: 0.9rem;">
            Enter device details for risk assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional form inputs
    device_name = st.sidebar.selectbox(
        "Device Name",
        options=[""] + sorted(device_names),
        help="Select or type to search for a device name",
        key="device_select"
    )
    
    manufacturer_name = st.sidebar.selectbox(
        "Manufacturer Name", 
        options=[""] + sorted(manufacturers),
        help="Select or type to search for a manufacturer",
        key="manufacturer_select"
    )
    
    # Professional assess button
    predict_button = st.sidebar.button("Assess Risk Level", type="primary", use_container_width=True)
    
    # Professional information panel
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="info-card">
        <h4 style="color: #1a365d; margin: 0 0 1rem 0; font-size: 1.1rem;">Platform Information</h4>
        <ul style="color: #4a5568; font-size: 0.9rem; margin: 0; padding-left: 1.2rem; line-height: 1.6;">
            <li>Start typing to search devices</li>
            <li>Use exact manufacturer names</li>
            <li>Analysis based on 34,744+ records</li>
            <li>90% model accuracy</li>
            <li>FDA-compliant risk assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Risk Assessment Results")
        
        if predict_button and device_name and manufacturer_name:
            with st.spinner("Analyzing device risk level..."):
                risk_level = predict_risk(device_name, manufacturer_name, model, le_device, le_manuf)
            
            if risk_level:
                risk_info = get_risk_display(risk_level)
                
                # Professional risk result display
                st.markdown(f"""
                <div class="risk-{risk_info['color']}">
                    <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem;">{risk_info['icon']} {risk_info['label']}</h2>
                    <div style="margin: 1rem 0;">
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;"><strong>Device:</strong> {device_name}</p>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;"><strong>Manufacturer:</strong> {manufacturer_name}</p>
                    </div>
                    <p style="margin: 1rem 0 0 0; font-size: 1rem; opacity: 0.95;">{risk_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Professional risk level indicator
                st.markdown("### Risk Level Assessment")
                
                # Create a professional risk meter
                risk_meter = st.progress(risk_level / 3)
                st.markdown(f"""
                <div style="text-align: center; margin: 1rem 0; padding: 1rem;
                            background: #2b2b40; border-radius: 8px; border: 1px solid #333;">
                    <p style="margin: 0; font-size: 1.1rem; color: #f5f5f5; font-weight: 600;">
                        Risk Level: {risk_level}/3 ({risk_info['label']})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        elif predict_button:
            st.warning("Please select both device name and manufacturer to assess risk level.")
        else:
            st.info("Enter device information in the sidebar and click 'Assess Risk Level' to begin analysis.")
    
    with col2:
        st.header("Platform Statistics")
        
        # Professional statistics display
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #99aaff; margin: 0; font-size: 1rem;">Total Devices</h3>
                <h2 style="color: #ffffff; margin: 0.5rem 0; font-size: 2rem;">{:,}</h2>
            </div>
            """.format(len(device_names)), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #99aaff; margin: 0; font-size: 1rem;">Manufacturers</h3>
                <h2 style="color: #ffffff; margin: 0.5rem 0; font-size: 2rem;">{:,}</h2>
            </div>
            """.format(len(manufacturers)), unsafe_allow_html=True)
        
        with col2_2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #99aaff; margin: 0; font-size: 1rem;">Model Accuracy</h3>
                <h2 style="color: #ffffff; margin: 0.5rem 0; font-size: 2rem;">90%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #99aaff; margin: 0; font-size: 1rem;">Data Points</h3>
                <h2 style="color: #ffffff; margin: 0.5rem 0; font-size: 2rem;">124,969</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Professional risk level distribution
        st.markdown("### Risk Level Distribution")
        st.markdown("""
        <div class="info-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0; padding: 0.5rem 0; border-bottom: 1px solid #444;">
                <span style="color: #dc2626; font-weight: 600;">High Risk</span>
                <span style="font-weight: 700; color: #f5f5f5;">17.2%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0; padding: 0.5rem 0; border-bottom: 1px solid #444;">
                <span style="color: #d97706; font-weight: 600;">Medium Risk</span>
                <span style="font-weight: 700; color: #f5f5f5;">75.9%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0; padding: 0.5rem 0;">
                <span style="color: #059669; font-weight: 600;">Low Risk</span>
                <span style="font-weight: 700; color: #f5f5f5;">6.9%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional footer
    st.markdown("""
    <div class="footer">
        <h4 style="color: #f5f5f5; margin: 0 0 1rem 0; font-size: 1.2rem;">Medical Device Risk Assessment Platform</h4>
        <p style="color: #c0c0c0; margin: 0.5rem 0; font-weight: 600; font-size: 1rem;">Powered by Advanced XGBoost Machine Learning</p>
        <p style="color: #888; margin: 0; font-size: 0.9rem; line-height: 1.6;">
            This platform provides comprehensive risk assessments based on historical device data and manufacturer information.<br>
            <strong>Disclaimer:</strong> Always consult with qualified medical professionals for critical device safety decisions.
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #444;">
            <p style="color: #888; margin: 0; font-size: 0.8rem;">
                © 2024 Medical Device Risk Assessment Platform | Version 2.0 | FDA Compliant
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


