"""Entrée Streamlit (local et Community Cloud) : navigation entre les pages AtlasShop."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="AtlasShop · UC2", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

from demo.theme import apply_theme  # noqa: E402
from demo.nav import PAGES  # noqa: E402

apply_theme()
st.navigation(list(PAGES.values()), position="top").run()
