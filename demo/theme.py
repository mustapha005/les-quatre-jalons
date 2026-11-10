"""Identité visuelle AtlasShop : vert sauge, titres serif, cartes arrondies, boutons pilule."""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

SAGE = "#667566"
SAGE_DARK = "#4F5D50"
SAGE_DEEP = "#2F3A33"
MIST = "#DCE1DA"
PAPER = "#F4F6F2"
INK = "#1F2A22"
GOOD = "#3E7C55"
WARN = "#B9822F"
BAD = "#A6493F"

ICONS = {
    "sparkle": '<path d="M12 2l1.8 5.6L19.5 9.5l-5.7 1.9L12 17l-1.8-5.6L4.5 9.5l5.7-1.9z"/><path d="M19 14l.9 2.6 2.6.9-2.6.9L19 21l-.9-2.6-2.6-.9 2.6-.9z"/>',
    "cart": '<path d="M3 4h2.5l2.2 10.5h10.6L20.5 7H7"/><circle cx="9.5" cy="19" r="1.4"/><circle cx="17" cy="19" r="1.4"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "tag": '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.3"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "live": '<circle cx="12" cy="12" r="3"/><path d="M5.6 5.6a9 9 0 000 12.8M18.4 5.6a9 9 0 010 12.8M8.5 8.5a4.5 4.5 0 000 7M15.5 8.5a4.5 4.5 0 010 7"/>',
}

CURVES = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 1000' preserveAspectRatio='none'>"
    "<g fill='none' stroke='white' stroke-opacity='.55' stroke-width='2'>"
    "<path d='M880 170 C980 40 1040 -20 1060 0'/>"
    "<path d='M1380 190 C1480 120 1560 40 1600 10'/>"
    "<path d='M0 700 C300 640 600 660 900 600'/>"
    "<path d='M660 1000 C650 900 660 860 690 820'/>"
    "</g></svg>"
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Jost:wght@300;400;500;600&display=swap');
:root {{ --sage:{SAGE}; --sage-dark:{SAGE_DARK}; --sage-deep:{SAGE_DEEP}; --mist:{MIST}; --paper:{PAPER}; --ink:{INK}; }}
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{ font-family:'Jost',system-ui,sans-serif; }}
.stApp {{
  background-color:var(--mist);
  background-image:url("{CURVES}"),
    radial-gradient(1200px 700px at 15% 0%, rgba(255,255,255,.55), transparent 60%),
    radial-gradient(900px 600px at 100% 100%, rgba(102,117,102,.18), transparent 60%);
  background-size:100% 100%, auto, auto; background-attachment:fixed;
}}
h1, h2, h3, h4, .serif {{ font-family:'DM Serif Display',Georgia,serif !important; font-weight:400 !important; letter-spacing:.2px; color:var(--ink); }}
.block-container {{ padding-top:4.5rem; max-width:1240px; }}
header[data-testid="stHeader"] {{ background:rgba(220,225,218,.82); backdrop-filter:blur(8px); }}
[data-testid="stSidebar"] {{ background:var(--sage-deep); }}
[data-testid="stSidebar"] * {{ color:#EEF1EC !important; }}

/* Navigation du haut */
[data-testid="stTopNavLink"] p, [data-testid="stNavigation"] a p {{ font-family:'Jost',sans-serif; letter-spacing:.4px; }}

/* Boutons pilule */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius:999px; padding:.42rem 1.15rem; font-size:.78rem; letter-spacing:.9px; text-transform:uppercase;
  border:1.5px solid var(--sage-dark); background:transparent; color:var(--sage-deep); font-weight:500;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:var(--sage-dark); color:#fff; border-color:var(--sage-dark); }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background:var(--sage-deep); color:#fff; border-color:var(--sage-deep);
}}

/* Panneau sauge comme la maquette */
[class*="st-key-panel"] {{
  background:var(--sage); border-radius:28px; padding:2rem 2rem 1.6rem; box-shadow:0 30px 60px -30px rgba(31,42,34,.45);
}}
[class*="st-key-panel"] h1, [class*="st-key-panel"] h2, [class*="st-key-panel"] h3,
[class*="st-key-panel"] p, [class*="st-key-panel"] span, [class*="st-key-panel"] label,
[class*="st-key-panel"] [data-testid="stMetricValue"], [class*="st-key-panel"] [data-testid="stMetricLabel"] p {{ color:#fff !important; }}
[class*="st-key-panel"] .stButton > button {{ border-color:#fff; color:#fff; }}
[class*="st-key-panel"] .stButton > button:hover {{ background:#fff; color:var(--sage-deep); }}
[class*="st-key-panel"] .stButton > button[kind="primary"] {{ background:#fff; color:var(--sage-deep); border-color:#fff; }}

/* Cartes blanches */
[class*="st-key-card"] {{
  background:rgba(255,255,255,.78); border-radius:22px; padding:1.25rem 1.35rem; border:1px solid rgba(255,255,255,.9);
  box-shadow:0 18px 40px -28px rgba(31,42,34,.5);
}}
[class*="st-key-pcard"] {{ text-align:center; gap:.45rem; }}
[class*="st-key-pcard"] img {{ border-radius:18px; max-width:230px !important; margin:0 auto; aspect-ratio:1/1; object-fit:cover; }}
[class*="st-key-pcard"] .stButton > button {{ width:100%; padding:.32rem .5rem; font-size:.68rem; }}

/* Composants maison */
.hero {{ display:flex; gap:1.4rem; align-items:center; margin:.4rem 0 1.6rem; }}
.badge {{ flex:0 0 auto; width:84px; height:84px; border-radius:50%; background:#F7F8F5; display:grid; place-items:center;
  box-shadow:0 14px 30px -14px rgba(31,42,34,.45); }}
.badge .icon {{ width:38px; height:38px; }}
.hero h1 {{ font-size:2.6rem; line-height:1.05; margin:0; padding:0; }}
.hero p {{ margin:.35rem 0 0; color:#43514A; font-size:1.05rem; max-width:760px; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:2px; font-size:.72rem; color:var(--sage-dark); font-weight:600; }}
.ptitle {{ font-family:'DM Serif Display',serif; font-size:1.12rem; line-height:1.2; color:var(--ink); margin-top:.4rem; }}
.psub {{ font-family:'Jost',sans-serif; font-size:.8rem; opacity:.8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.pprice {{ font-size:.95rem; color:#3A463E; margin-bottom:.2rem; }}
[class*="st-key-panel"] .ptitle {{ color:#fff; }}
.chip {{ display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.72rem; letter-spacing:.8px; text-transform:uppercase; font-weight:600; }}
.chip.good {{ background:#DCEBDD; color:{GOOD}; }} .chip.warn {{ background:#F4E7CF; color:{WARN}; }} .chip.bad {{ background:#F2D9D5; color:{BAD}; }}
.chip.neutral {{ background:#E7EBE5; color:var(--sage-deep); }}
.bigscore {{ font-family:'DM Serif Display',serif; font-size:4.2rem; line-height:1; color:var(--ink); }}
[class*="st-key-panel"] .bigscore {{ color:#fff; }}
.bar {{ height:9px; border-radius:99px; background:rgba(47,58,51,.12); overflow:hidden; }}
.bar > span {{ display:block; height:100%; border-radius:99px; background:var(--sage-dark); }}
.factor {{ display:grid; grid-template-columns: 1fr 52px; gap:.2rem .8rem; align-items:center; margin:.55rem 0; }}
.factor small {{ color:#55625A; grid-column:1 / -1; margin-top:-.15rem; }}
.note {{ font-size:.85rem; color:#55625A; }}
.feature {{ height:100%; }}
.feature h3 {{ margin:.6rem 0 .3rem; font-size:1.45rem; }}
.feature p {{ color:#4A574F; margin:0; }}
.feature .badge {{ width:58px; height:58px; }} .feature .badge .icon {{ width:26px; height:26px; }}
.eventlog {{ font-size:.86rem; line-height:1.55; max-height:260px; overflow:auto; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius:20px; background:rgba(255,255,255,.6); }}
[data-testid="stMetric"] {{ background:transparent; }}
[data-testid="stExpander"] details {{ border-radius:18px; background:rgba(255,255,255,.55); }}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{ font-family:'Material Symbols Rounded' !important; }}
[class*="st-key-panel"] .stButton > button[kind="primary"] p {{ color:var(--sage-deep) !important; }}
[class*="st-key-panel"] .psub {{ font-family:'Jost',sans-serif; font-size:.8rem; opacity:.8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.pprice {{ color:#F1F4EF; }}
[class*="st-key-panel"] [class*="st-key-card"] p, [class*="st-key-panel"] [class*="st-key-card"] span,
[class*="st-key-panel"] [class*="st-key-card"] label, [class*="st-key-panel"] [class*="st-key-card"] b {{ color:var(--ink) !important; }}
[class*="st-key-panel"] [class*="st-key-card"] small, [class*="st-key-panel"] [class*="st-key-card"] .note {{ color:#55625A !important; }}
[class*="st-key-pcard"] [data-testid="stButton"], [class*="st-key-pcard"] [data-testid="stButton"] button {{ width:100% !important; }}
[class*="st-key-panel"] .chip.good {{ color:{GOOD} !important; }} [class*="st-key-panel"] .chip.warn {{ color:{WARN} !important; }}
[class*="st-key-panel"] .chip.bad {{ color:{BAD} !important; }} [class*="st-key-panel"] .chip.neutral {{ color:var(--sage-deep) !important; }}
[data-testid="stVegaLiteChart"] {{ background:transparent; }}
@media (max-width: 640px) {{ .hero h1 {{ font-size:1.9rem; }} .badge {{ width:62px; height:62px; }} .bigscore {{ font-size:3.2rem; }} }}
</style>
"""


def apply_theme() -> None:
    st.markdown("\n".join(line for line in CSS.splitlines() if line.strip()), unsafe_allow_html=True)


def icon_svg(name: str) -> str:
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{SAGE_DARK}' stroke-width='1.6' "
           f"stroke-linejoin='round' stroke-linecap='round'>{ICONS[name]}</svg>")
    data = base64.b64encode(svg.replace('"', "'").encode()).decode("ascii")
    return f'<img class="icon" alt="" src="data:image/svg+xml;base64,{data}">'


def hero(title: str, subtitle: str, icon: str = "sparkle", eyebrow: str | None = None) -> None:
    eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.html(f'<div class="hero"><div class="badge">{icon_svg(icon)}</div><div>{eb}<h1>{title}</h1><p>{subtitle}</p></div></div>')


def chip(text: str, kind: str = "neutral") -> str:
    return f'<span class="chip {kind}">{text}</span>'


def factor_bar(label: str, value: float, help_text: str = "") -> str:
    v = float(np.clip(value, 0, 1))
    small = f"<small>{help_text}</small>" if help_text else ""
    return (f'<div class="factor"><div><div style="display:flex;justify-content:space-between"><span>{label}</span></div>'
            f'<div class="bar"><span style="width:{v*100:.0f}%"></span></div></div><b>{v*100:.0f}%</b>{small}</div>')


def verdict(score: float, high: float = .65, low: float = .45) -> tuple[str, str]:
    if score >= high:
        return "Potentiel élevé", "good"
    if score >= low:
        return "Potentiel moyen", "warn"
    return "Potentiel faible", "bad"


def product_png(pixels: np.ndarray, size: int = 280, bg: str = "#E9ECE5", fg: str = SAGE_DEEP) -> bytes:
    """Rend une image 28×28 Fashion-MNIST comme une photo produit sur fond clair."""
    arr = np.asarray(pixels)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(np.clip(arr, 0, 1) * 255)
    img = Image.fromarray(arr.astype(np.uint8), "L").resize((size, size), Image.Resampling.BICUBIC)
    img = ImageOps.colorize(img, black=bg, white=fg)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def gray_png(pixels: np.ndarray, size: int = 224) -> bytes:
    arr = np.asarray(pixels)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(np.clip(arr, 0, 1) * 255)
    img = Image.fromarray(arr.astype(np.uint8), "L").resize((size, size), Image.Resampling.NEAREST)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def img_tag(png: bytes, alt: str, width: int | None = None, radius: int = 18) -> str:
    w = f' width="{width}"' if width else ' style="width:100%"'
    data = base64.b64encode(png).decode("ascii")
    return f'<img alt="{alt}"{w} src="data:image/png;base64,{data}" style="border-radius:{radius}px;display:block;max-width:100%">'


def show_chart(chart, **_):
    """Graphique Altair sur fond transparent, intégré aux cartes."""
    st.altair_chart(chart.configure(background="#00000000", font="Jost").configure_view(stroke=None), use_container_width=True)
