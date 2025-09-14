@echo off
echo Starting Medical Device Risk Assessment Dashboard...
echo.
echo Dashboard will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.
python -m streamlit run dashboard.py --server.headless true --server.port 8501
pause
