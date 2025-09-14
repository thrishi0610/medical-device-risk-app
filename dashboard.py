import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical Device Risk Assessment Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the dataset for autocomplete suggestions"""
    try:
        df = pd.read_csv("data csv/final_merged_dataset.csv", low_memory=False)
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
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Device Risk Assessment Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("Loading data and model..."):
        device_names, manufacturers = load_data()
        model, le_device, le_manuf = load_model_and_encoders()
    
    if not device_names or not manufacturers or model is None:
        st.error("Failed to load required data or model. Please ensure all files are present.")
        return
    
    # Sidebar for input
    st.sidebar.header("📋 Device Information")
    st.sidebar.markdown("Enter the device details to assess risk level:")
    
    # Device name input with autocomplete
    device_name = st.sidebar.selectbox(
        "Device Name",
        options=[""] + sorted(device_names),
        help="Select or type to search for a device name"
    )
    
    # Manufacturer input with autocomplete
    manufacturer_name = st.sidebar.selectbox(
        "Manufacturer Name", 
        options=[""] + sorted(manufacturers),
        help="Select or type to search for a manufacturer"
    )
    
    # Predict button
    predict_button = st.sidebar.button("🔍 Assess Risk", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Risk Assessment Results")
        
        if predict_button and device_name and manufacturer_name:
            with st.spinner("Analyzing device risk..."):
                risk_level = predict_risk(device_name, manufacturer_name, model, le_device, le_manuf)
            
            if risk_level:
                risk_info = get_risk_display(risk_level)
                
                # Display risk result
                st.markdown(f"""
                <div class="risk-{risk_info['color']}">
                    <h2>{risk_info['icon']} {risk_info['label']}</h2>
                    <p><strong>Device:</strong> {device_name}</p>
                    <p><strong>Manufacturer:</strong> {manufacturer_name}</p>
                    <p><strong>Description:</strong> {risk_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk level indicator
                st.markdown("### Risk Level Indicator")
                risk_colors = {"red": "🔴", "orange": "🟡", "green": "🟢"}
                risk_icons = {"red": "🔴", "orange": "🟡", "green": "🟢"}
                
                # Create a visual risk meter
                risk_meter = st.progress(risk_level / 3)
                st.caption(f"Risk Level: {risk_level}/3 ({risk_info['label']})")
                
        elif predict_button:
            st.warning("⚠️ Please select both device name and manufacturer to assess risk.")
        else:
            st.info("👈 Please enter device information in the sidebar and click 'Assess Risk' to get started.")
    
    with col2:
        st.header("📊 Dashboard Statistics")
        
        # Display some statistics
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric("Total Devices", f"{len(device_names):,}")
            st.metric("Total Manufacturers", f"{len(manufacturers):,}")
        
        with col2_2:
            st.metric("Model Accuracy", "90%")
            st.metric("Data Points", "124,969")
        
        # Sample devices for reference
        st.markdown("### 🔍 Sample Devices")
        sample_devices = device_names[:10] if len(device_names) >= 10 else device_names
        for device in sample_devices:
            st.text(f"• {device}")
        
        if len(device_names) > 10:
            st.caption(f"... and {len(device_names) - 10} more devices")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Medical Device Risk Assessment Dashboard | Powered by XGBoost Machine Learning</p>
        <p><small>This tool provides risk assessments based on historical device data and manufacturer information.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
