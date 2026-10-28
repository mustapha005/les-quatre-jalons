"""Entrée Streamlit, compatible lancement local et Community Cloud."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "demo" / "app.py"), run_name="__main__")
