"""Identité visuelle AtlasShop : tableau de bord violet et vert, cartes en verre, graphiques à dégradés."""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

VIOLET = "#6C4DF6"
VIOLET_DEEP = "#4B2FD1"
VIOLET_SOFT = "#EDE9FF"
GREEN = "#22C77E"
GREEN_SOFT = "#DDF7EA"
PINK = "#E879C9"
AMBER = "#F5A524"
RED = "#F2557A"
INK = "#1E1B3A"
MUTED = "#7C7A99"
GRID = "#ECEAF7"
TRACK = "#E9E6FA"
# alias historiques utilisés par les vues
SAGE_DARK = VIOLET
SAGE_DEEP = INK
GOOD, WARN, BAD = GREEN, AMBER, RED
PALETTE = [VIOLET, GREEN, PINK, "#5AA9FF", AMBER, "#9B8CFF"]

ICONS = {
    "sparkle": '<path d="M12 2l1.8 5.6L19.5 9.5l-5.7 1.9L12 17l-1.8-5.6L4.5 9.5l5.7-1.9z"/><path d="M19 14l.9 2.6 2.6.9-2.6.9L19 21l-.9-2.6-2.6-.9 2.6-.9z"/>',
    "cart": '<path d="M3 4h2.5l2.2 10.5h10.6L20.5 7H7"/><circle cx="9.5" cy="19" r="1.4"/><circle cx="17" cy="19" r="1.4"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "tag": '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.3"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "live": '<circle cx="12" cy="12" r="3"/><path d="M5.6 5.6a9 9 0 000 12.8M18.4 5.6a9 9 0 010 12.8M8.5 8.5a4.5 4.5 0 000 7M15.5 8.5a4.5 4.5 0 010 7"/>',
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
:root {{ --violet:{VIOLET}; --violet-deep:{VIOLET_DEEP}; --violet-soft:{VIOLET_SOFT}; --green:{GREEN}; --ink:{INK}; --muted:{MUTED}; }}
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{ font-family:'Inter',system-ui,sans-serif; }}
.stApp {{
  background: radial-gradient(900px 600px at 0% 0%, rgba(255,255,255,.55), transparent 60%),
              radial-gradient(800px 600px at 100% 100%, rgba(232,121,201,.28), transparent 60%),
              linear-gradient(135deg, #C3CEF6 0%, #D9DAF8 38%, #E7DCF7 70%, #F3D8EE 100%);
  background-attachment:fixed;
}}
h1, h2, h3, h4 {{ font-family:'Poppins',system-ui,sans-serif !important; font-weight:600 !important; letter-spacing:-.2px; color:var(--ink); }}
.block-container {{ padding-top:4.5rem; max-width:1260px; }}
header[data-testid="stHeader"] {{ background:rgba(222,224,250,.75); backdrop-filter:blur(10px); }}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,{VIOLET} 0%,{VIOLET_DEEP} 100%); }}
[data-testid="stSidebar"] * {{ color:#F3F0FF !important; }}
[data-testid="stTopNavLink"] p {{ font-weight:500; }}

/* Boutons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius:12px; padding:.45rem 1.1rem; font-size:.84rem; font-weight:600; letter-spacing:.1px;
  border:1.5px solid rgba(108,77,246,.35); background:rgba(255,255,255,.7); color:var(--violet-deep);
  transition:all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--violet); color:var(--violet); background:#fff;
  box-shadow:0 8px 20px -10px rgba(108,77,246,.6); }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background:linear-gradient(135deg,#7D60FF,{VIOLET_DEEP}); color:#fff; border:none; box-shadow:0 10px 22px -10px rgba(75,47,209,.8);
}}
.stButton > button[kind="primary"] p, .stDownloadButton > button[kind="primary"] p {{ color:#fff !important; }}

/* Panneau vif (scores, résultats) */
[class*="st-key-panel"] {{
  background:linear-gradient(135deg,#7E62FF 0%,{VIOLET} 45%,{VIOLET_DEEP} 100%); border-radius:24px; padding:1.6rem 1.8rem;
  box-shadow:0 28px 60px -28px rgba(75,47,209,.75);
}}
[class*="st-key-panel"] h1, [class*="st-key-panel"] h2, [class*="st-key-panel"] h3,
[class*="st-key-panel"] p, [class*="st-key-panel"] span, [class*="st-key-panel"] label,
[class*="st-key-panel"] [data-testid="stMetricValue"], [class*="st-key-panel"] [data-testid="stMetricLabel"] p {{ color:#fff !important; }}
[class*="st-key-panel"] .stButton > button {{ background:rgba(255,255,255,.14); border-color:rgba(255,255,255,.55); color:#fff; }}
[class*="st-key-panel"] .stButton > button[kind="primary"] {{ background:#fff; color:var(--violet-deep); }}
[class*="st-key-panel"] .stButton > button[kind="primary"] p {{ color:var(--violet-deep) !important; }}
[class*="st-key-panel"] [data-baseweb="select"] > div, [class*="st-key-panel"] input {{ background:rgba(255,255,255,.95) !important; }}
[class*="st-key-panel"] [data-baseweb="select"] span, [class*="st-key-panel"] input {{ color:var(--ink) !important; }}

/* Cartes verre */
[class*="st-key-card"], [class*="st-key-glass"] {{
  background:rgba(255,255,255,.62); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  border-radius:22px; padding:1.25rem 1.35rem; border:1px solid rgba(255,255,255,.8);
  box-shadow:0 22px 48px -30px rgba(76,61,180,.55);
}}
[class*="st-key-pcard"] {{ text-align:center; gap:.4rem; background:#fff; border-radius:20px; padding:.7rem .7rem .9rem;
  box-shadow:0 14px 30px -22px rgba(76,61,180,.6); transition:transform .15s ease, box-shadow .15s ease; }}
[class*="st-key-pcard"]:hover {{ transform:translateY(-3px); box-shadow:0 22px 40px -22px rgba(76,61,180,.7); }}
[class*="st-key-pcard"] img {{ border-radius:14px; max-width:230px !important; margin:0 auto; aspect-ratio:1/1; object-fit:cover; }}
[class*="st-key-pcard"] [data-testid="stButton"], [class*="st-key-pcard"] [data-testid="stButton"] button {{ width:100% !important; }}
[class*="st-key-pcard"] .stButton > button {{ padding:.32rem .5rem; font-size:.76rem; }}

/* Composants */
.hero {{ display:flex; gap:1.2rem; align-items:center; margin:.2rem 0 1.4rem; }}
.badge {{ flex:0 0 auto; width:72px; height:72px; border-radius:20px; background:linear-gradient(135deg,#7E62FF,{VIOLET_DEEP});
  display:grid; place-items:center; box-shadow:0 16px 30px -14px rgba(75,47,209,.8); }}
.badge .icon {{ width:34px; height:34px; }}
.hero h1 {{ font-size:2.35rem; line-height:1.1; margin:0; padding:0; }}
.hero p {{ margin:.3rem 0 0; color:#5A587A; font-size:1.02rem; max-width:780px; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:1.6px; font-size:.7rem; color:var(--violet); font-weight:700; }}
.ctitle {{ font-family:'Poppins',sans-serif; font-weight:600; font-size:1rem; color:var(--ink); display:flex; align-items:center;
  justify-content:space-between; margin-bottom:.3rem; }}
.pill {{ font-family:'Inter',sans-serif; font-size:.72rem; font-weight:600; padding:.18rem .65rem; border-radius:8px;
  background:var(--violet); color:#fff; }}
.pill.soft {{ background:var(--violet-soft); color:var(--violet-deep); }}
.ptitle {{ font-family:'Poppins',sans-serif; font-weight:600; font-size:1rem; line-height:1.25; color:var(--ink); margin-top:.35rem; }}
.psub {{ font-size:.75rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.pprice {{ font-family:'Poppins',sans-serif; font-weight:600; font-size:1.02rem; color:var(--violet-deep); margin:.1rem 0 .25rem; }}
.chip {{ display:inline-block; padding:.22rem .7rem; border-radius:8px; font-size:.74rem; font-weight:600; }}
.chip.good {{ background:{GREEN_SOFT}; color:#0F9A5C !important; }} .chip.warn {{ background:#FFF1D6; color:#B87500 !important; }}
.chip.bad {{ background:#FFE1E8; color:#D02A55 !important; }} .chip.neutral {{ background:var(--violet-soft); color:var(--violet-deep) !important; }}
.bigscore {{ font-family:'Poppins',sans-serif; font-weight:700; font-size:3.8rem; line-height:1; color:var(--ink); }}
[class*="st-key-panel"] .bigscore {{ color:#fff; }}
.bar {{ height:10px; border-radius:99px; background:{TRACK}; overflow:hidden; }}
.bar > span {{ display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,{VIOLET},{PINK}); }}
.bar.green > span {{ background:linear-gradient(90deg,#4FE0A0,{GREEN}); }}
.factor {{ display:grid; grid-template-columns: 1fr 52px; gap:.25rem .8rem; align-items:center; margin:.6rem 0; }}
.factor b {{ font-family:'Poppins',sans-serif; color:var(--ink); }}
.factor small {{ color:var(--muted); grid-column:1 / -1; margin-top:-.15rem; }}
.note {{ font-size:.84rem; color:var(--muted); }}
.feature h3 {{ margin:.7rem 0 .3rem; font-size:1.2rem; }}
.feature p {{ color:#5A587A; margin:0; font-size:.93rem; }}
.feature .badge {{ width:50px; height:50px; border-radius:14px; }} .feature .badge .icon {{ width:24px; height:24px; }}
.kpi {{ display:flex; flex-direction:column; gap:.15rem; }}
.kpi .v {{ font-family:'Poppins',sans-serif; font-weight:700; font-size:1.9rem; color:var(--ink); line-height:1.1; }}
.kpi .l {{ font-size:.8rem; color:var(--muted); }}
.kpi .d {{ font-size:.75rem; font-weight:600; }}
[class*="st-key-panel"] .kpi .v {{ color:#fff; }} [class*="st-key-panel"] .kpi .l {{ color:rgba(255,255,255,.8) !important; }}
.eventlog {{ font-size:.86rem; line-height:1.6; max-height:280px; overflow:auto; }}
.eventlog .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.45rem; }}
.svgwrap {{ display:flex; justify-content:center; }}
.svgwrap img {{ max-width:100%; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius:16px; background:rgba(255,255,255,.7); border:1.5px dashed rgba(108,77,246,.35); }}
[data-testid="stExpander"] details {{ border-radius:16px; background:rgba(255,255,255,.6); border:1px solid rgba(255,255,255,.8); }}
[data-testid="stTabs"] button[aria-selected="true"] p {{ color:var(--violet) !important; font-weight:600; }}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{ font-family:'Material Symbols Rounded' !important; }}
[class*="st-key-panel"] [class*="st-key-card"] p, [class*="st-key-panel"] [class*="st-key-card"] span,
[class*="st-key-panel"] [class*="st-key-card"] label, [class*="st-key-panel"] [class*="st-key-card"] b {{ color:var(--ink) !important; }}
[class*="st-key-panel"] [class*="st-key-card"] small, [class*="st-key-panel"] [class*="st-key-card"] .note {{ color:var(--muted) !important; }}
[class*="st-key-panel"] [class*="st-key-card"] {{ background:rgba(255,255,255,.94); }}
[data-testid="stVegaLiteChart"] {{ background:transparent; }}
@media (max-width: 640px) {{ .hero h1 {{ font-size:1.7rem; }} .badge {{ width:56px; height:56px; }} .bigscore {{ font-size:3rem; }} }}
</style>
"""


def apply_theme() -> None:
    st.markdown("\n".join(line for line in CSS.splitlines() if line.strip()), unsafe_allow_html=True)


def icon_svg(name: str) -> str:
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#FFFFFF' stroke-width='1.8' "
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
    cls = "bar green" if v >= .65 else "bar"
    return (f'<div class="factor"><div><div style="display:flex;justify-content:space-between"><span>{label}</span></div>'
            f'<div class="{cls}"><span style="width:{v*100:.0f}%"></span></div></div><b>{v*100:.0f}%</b>{small}</div>')


def verdict(score: float, high: float = .65, low: float = .45) -> tuple[str, str]:
    if score >= high:
        return "Potentiel élevé", "good"
    if score >= low:
        return "Potentiel moyen", "warn"
    return "Potentiel faible", "bad"


def product_png(pixels: np.ndarray, size: int = 280, bg: str = "#F1EEFF", fg: str = INK) -> bytes:
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
    mime = "image/jpeg" if png[:2] == b"\xff\xd8" else "image/png"
    return f'<img alt="{alt}"{w} src="data:{mime};base64,{data}" style="border-radius:{radius}px;display:block;max-width:100%">'


def _style(chart, dark: bool = False):
    label = "rgba(255,255,255,.85)" if dark else MUTED
    grid = "rgba(255,255,255,.18)" if dark else GRID
    return (chart.configure(background="#00000000", font="Inter")
            .configure_axis(labelColor=label, titleColor=label, gridColor=grid, gridDash=[3, 4], domain=False,
                            ticks=False, labelFontSize=11, titleFontSize=11, titleFontWeight=500, labelPadding=6)
            .configure_legend(labelColor=label, titleColor=label, symbolType="circle", symbolSize=90, labelFontSize=12)
            .configure_view(stroke=None)
            .configure_range(category=PALETTE))


def show_chart(chart, dark: bool = False, **_):
    """Graphique Altair au style du tableau de bord (fond transparent, grille pointillée, palette violet/vert)."""
    st.altair_chart(_style(chart, dark), theme=None, use_container_width=True)


def gradient(color: str = VIOLET, top: float = .55, bottom: float = 0.0, horizontal: bool = False):
    import altair as alt
    def rgba(a):
        h = color.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
    if horizontal:
        return alt.Gradient(gradient="linear", stops=[alt.GradientStop(color=rgba(bottom), offset=0), alt.GradientStop(color=rgba(top), offset=1)],
                            x1=0, x2=1, y1=0, y2=0)
    return alt.Gradient(gradient="linear", stops=[alt.GradientStop(color=rgba(top), offset=0), alt.GradientStop(color=rgba(bottom), offset=1)],
                        x1=1, x2=1, y1=0, y2=1)


def area_line(df, x: str, y: str, color: str = VIOLET, y_domain=None, rule: float | None = None, rule_color: str = GREEN,
              height: int = 220, dark: bool = False, x_title=None, y_title=None, y_format: str = "%"):
    """Courbe lissée avec aire en dégradé, points et ligne de seuil facultative."""
    import altair as alt
    yscale = alt.Scale(domain=y_domain) if y_domain else alt.Undefined
    line_c = "#FFFFFF" if dark else color
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=yscale, axis=alt.Axis(format=y_format, tickCount=4)))
    area = base.mark_area(interpolate="monotone", color=gradient("#FFFFFF" if dark else color, .45 if dark else .5, 0))
    line = base.mark_line(interpolate="monotone", strokeWidth=3, color=line_c)
    pts = base.mark_point(filled=True, size=70, color="#FFFFFF", stroke=line_c if not dark else VIOLET_DEEP, strokeWidth=2.5, opacity=1).encode(
        tooltip=[alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q", format=".1%" if y_format == "%" else ".3f")])
    layers = [area, line, pts]
    if rule is not None:
        import pandas as pd
        layers.append(alt.Chart(pd.DataFrame({"r": [rule]})).mark_rule(strokeDash=[6, 5], strokeWidth=2, color=rule_color).encode(y="r:Q"))
    return alt.layer(*layers).properties(height=height)


def _svg_img(svg: str, width: int) -> str:
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<div class="svgwrap"><img alt="" width="{width}" src="data:image/svg+xml;base64,{data}"></div>'


def gauge(value: float, label: str = "", dark: bool = False, threshold: float | None = None, width: int = 260) -> str:
    """Jauge demi-cercle en dégradé (SVG), comme le compteur du tableau de bord de référence."""
    import math
    v = float(np.clip(value, 0, 1))
    L = math.pi * 100
    track = "rgba(255,255,255,.22)" if dark else TRACK
    txt = "#FFFFFF" if dark else INK
    sub = "rgba(255,255,255,.8)" if dark else MUTED
    ang = math.pi * (1 - v)
    nx, ny = 120 + 100 * math.cos(ang), 120 - 100 * math.sin(ang)
    tick = ""
    if threshold is not None:
        ta = math.pi * (1 - float(np.clip(threshold, 0, 1)))
        x1, y1 = 120 + 86 * math.cos(ta), 120 - 86 * math.sin(ta)
        x2, y2 = 120 + 116 * math.cos(ta), 120 - 116 * math.sin(ta)
        tick = f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{GREEN}" stroke-width="3" stroke-linecap="round"/>'
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 240 150' font-family='Inter,Segoe UI,Helvetica,Arial,sans-serif'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='0'><stop offset='0' stop-color='#9D86FF'/><stop offset='.6' stop-color='{VIOLET}'/>
<stop offset='1' stop-color='{GREEN}'/></linearGradient></defs>
<path d='M20 120 A100 100 0 0 1 220 120' fill='none' stroke='{track}' stroke-width='18' stroke-linecap='round'/>
<path d='M20 120 A100 100 0 0 1 220 120' fill='none' stroke='url(#g)' stroke-width='18' stroke-linecap='round'
 stroke-dasharray='{v * L:.1f} {L:.1f}'/>
{tick}
<circle cx='{nx:.1f}' cy='{ny:.1f}' r='11' fill='#FFFFFF' stroke='{VIOLET_DEEP}' stroke-width='4'/>
<text x='120' y='112' text-anchor='middle' font-size='34' font-weight='700' fill='{txt}'>{v*100:.0f}%</text>
<text x='22' y='144' text-anchor='middle' font-size='11' fill='{sub}'>0%</text>
<text x='218' y='144' text-anchor='middle' font-size='11' fill='{sub}'>100%</text>
<text x='120' y='145' text-anchor='middle' font-size='11' fill='{sub}'>{label}</text>
</svg>"""
    return _svg_img(svg, width)


def donut(parts: list[tuple[str, float, str]], center: str, caption: str = "", dark: bool = False, width: int = 200) -> str:
    """Anneau multi-segments (SVG) : parts = [(nom, valeur, couleur)], avec un chiffre au centre."""
    import math
    total = sum(max(p[1], 0) for p in parts) or 1.0
    C = 2 * math.pi * 70
    track = "rgba(255,255,255,.22)" if dark else TRACK
    txt = "#FFFFFF" if dark else INK
    sub = "rgba(255,255,255,.8)" if dark else MUTED
    segs, off = [], 0.0
    gap = 4 if len([p for p in parts if p[1] > 0]) > 1 else 0
    for _, val, col in parts:
        frac = max(val, 0) / total
        ln = max(frac * C - gap, 0)
        if ln > 0:
            segs.append(f"<circle cx='100' cy='100' r='70' fill='none' stroke='{col}' stroke-width='24' stroke-linecap='butt' "
                        f"stroke-dasharray='{ln:.1f} {C:.1f}' stroke-dashoffset='{-off:.1f}' transform='rotate(-90 100 100)'/>")
        off += frac * C
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' font-family='Inter,Segoe UI,Helvetica,Arial,sans-serif'>
<circle cx='100' cy='100' r='70' fill='none' stroke='{track}' stroke-width='24'/>{''.join(segs)}
<circle cx='100' cy='100' r='52' fill='{"rgba(255,255,255,.08)" if dark else "#FFFFFF"}'/>
<text x='100' y='{104 if not caption else 100}' text-anchor='middle' font-size='30' font-weight='700' fill='{txt}'>{center}</text>
<text x='100' y='122' text-anchor='middle' font-size='11' fill='{sub}'>{caption}</text></svg>"""
    return _svg_img(svg, width)


def legend_rows(parts: list[tuple[str, str, str]]) -> str:
    """Lignes de légende : (couleur, libellé, valeur) — style « Active Statistics »."""
    rows = "".join(f'<div style="display:flex;align-items:center;justify-content:space-between;margin:.35rem 0">'
                   f'<span style="display:flex;align-items:center;gap:.5rem"><span style="width:10px;height:10px;border-radius:3px;'
                   f'background:{c};display:inline-block"></span>{l}</span><b>{v}</b></div>' for c, l, v in parts)
    return f'<div style="font-size:.88rem">{rows}</div>'


def kpi(value: str, label: str, delta: str = "", good: bool = True) -> str:
    d = f'<span class="d" style="color:{GREEN if good else RED}">{delta}</span>' if delta else ""
    return f'<div class="kpi"><span class="l">{label}</span><span class="v">{value}</span>{d}</div>'


def ctitle(text: str, pill: str | None = None, soft: bool = False) -> str:
    p = f'<span class="pill{" soft" if soft else ""}">{pill}</span>' if pill else ""
    return f'<div class="ctitle"><span>{text}</span>{p}</div>'
