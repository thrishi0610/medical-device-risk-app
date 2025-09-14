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
        margin: 2rem
