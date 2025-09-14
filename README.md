# Medical Device Risk Assessment Dashboard

A web-based dashboard for predicting medical device risk levels using machine learning.

## Features

- 🎯 **Risk Prediction**: Enter device name and manufacturer to get risk assessment
- 🔍 **Autocomplete**: Smart suggestions for device names and manufacturers
- 📊 **Visual Indicators**: Color-coded risk levels (High/Medium/Low)
- 🏥 **Medical Focus**: Specialized for medical device safety assessment

## Risk Levels

- **🔴 HIGH RISK (1)**: Requires immediate attention and careful monitoring
- **🟡 MEDIUM RISK (2)**: Should be monitored regularly  
- **🟢 LOW RISK (3)**: Minimal safety concerns

## Quick Start

### Option 1: Double-click launcher
Simply double-click `launch_dashboard.bat` to start the dashboard.

### Option 2: Command line
```bash
python -m streamlit run dashboard.py --server.headless true --server.port 8501
```

### Option 3: Regular Streamlit (if available)
```bash
streamlit run dashboard.py
```

## Access the Dashboard

Once started, open your web browser and go to:
**http://localhost:8501**

## How to Use

1. **Enter Device Information**: Use the sidebar to select device name and manufacturer
2. **Get Suggestions**: The dropdowns provide autocomplete suggestions from the database
3. **Assess Risk**: Click "🔍 Assess Risk" to get the prediction
4. **View Results**: See the risk level with color-coded indicators and descriptions

## Technical Details

- **Model**: XGBoost classifier trained on 34,744+ medical device records
- **Accuracy**: 90% validation accuracy
- **Features**: Device name and manufacturer name
- **Data Source**: Medical device recall and safety data

## Files

- `dashboard.py` - Main Streamlit dashboard application
- `xgbModel_2feat.model` - Trained XGBoost model
- `le_device.pkl` - Device name label encoder
- `le_manuf.pkl` - Manufacturer name label encoder
- `data csv/final_merged_dataset.csv` - Source dataset
- `launch_dashboard.bat` - Windows launcher script
- `test_dashboard.py` - Component testing script

## Requirements

- Python 3.11+
- Streamlit
- XGBoost
- Pandas
- Scikit-learn
- Joblib

## Troubleshooting

If you encounter issues:

1. **Streamlit not found**: Use `python -m streamlit run dashboard.py`
2. **Model errors**: Run `python test_dashboard.py` to verify components
3. **Port conflicts**: Change port with `--server.port 8502`

## Support

This dashboard provides risk assessments based on historical device data and manufacturer information. Always consult with medical professionals for critical decisions.
