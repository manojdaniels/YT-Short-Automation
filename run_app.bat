@echo off
cd /d "%~dp0"
python -m streamlit run app.py --server.address localhost --server.port 8501
