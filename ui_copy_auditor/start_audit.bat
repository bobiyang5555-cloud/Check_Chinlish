@echo off
cd /d D:\ui_copy_auditor
call .venv\Scripts\activate
python audit_core.py
python -m streamlit run app.py
pause