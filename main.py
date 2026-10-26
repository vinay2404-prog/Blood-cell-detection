"""Launcher: applies the shadcn skin, then runs your unchanged app.py.

Run with:  streamlit run main.py
"""
import runpy
from pathlib import Path

import streamlit as st
from streamlit.commands.page_config import set_page_config as _real_set_page_config

BASE_DIR = Path(__file__).resolve().parent

# Must be the first Streamlit call. Same settings as app.py.
_real_set_page_config(
    page_title="HemoScan AI | Blood Smear Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# app.py calls set_page_config again, which would raise; make it a no-op.
st.set_page_config = lambda *args, **kwargs: None

css = (BASE_DIR / "shadcn.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

runpy.run_path(str(BASE_DIR / "app.py"), run_name="__main__")