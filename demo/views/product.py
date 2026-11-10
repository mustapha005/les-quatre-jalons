"""Fiche produit : photo + titre + description + prix → catégorie, prix estimé et indice de potentiel."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from demo import core
from demo.theme import chip, ctitle, donut, factor_bar, gray_png, hero, img_tag, verdict, show_chart


def weights_editor(prefix: str) -> dict:
    with st.expander("Pondération de l’indice"):
        st.caption("L’indice est une moyenne pondérée de signaux issus des modèles. Ajustez l’importance de chaque signal.")
        w = {}
        cols = st.columns(len(core.DEFAULT_WEIGHTS))
        for col, (k, v) in zip(cols, core.DEFAULT_WEIGHTS.items()):
            w[k] = col.slider(core.FACTOR_LABELS[k][0], 0.0, 1.0, v, 0.05, key=f"{prefix}_w_{k}")
        if sum(w.values()) == 0:
            w = dict(core.DEFAULT_WEIGHTS)
    return w


def render():
    hero("Potentiel produit",
         "Ajoutez la photo, le titre, la description et le prix d’un article. Les modèles prédisent sa catégorie, estiment son prix "
         "et calculent un indice de potentiel de vente.", icon="tag", eyebrow="Analyse d’annonce")

    r = core.real_photos()
    left, right = st.columns([1, 1.25], gap="large")
    with left, st.container(key="card_product_form"):
        source = st.segmented_control("Image", ["Photo exemple", "Ma photo"], default="Photo exemple", key="prod_source") or "Photo exemple"
        image28, original, cut = None, None, None
        if source == "Ma photo":
            up = st.file_uploader("Photo du produit (fond uni de préférence)", type=["png", "jpg", "jpeg", "webp"], key="prod_img")
            if up is not None:
                with st.spinner("Détourage de la photo…"):
                    image28, cut = _convert(up.getvalue())
                original = Image.open(up)
            default_title, default_desc, default_price = "", "", 250.0
            tag = "upload"
        else:
            idx = st.slider("Photo réelle (70 exemples, domaine public)", 0, len(r["name"]) - 1, 30, key="prod_idx")
            name = str(r["name"][idx])
            st.html(img_tag(core.square_jpeg(name, 280), name, width=200))
            image28 = r["X"][idx]
            original = core.real_photo(name)
            cut = core.cutout_png(original, 280)
            default_title, default_desc = str(r["title"][idx]), ""
            default_price = float(core.category_median_price()[int(r["categorie"][idx])])
            tag = str(idx)
        title = st.text_input("Titre", default_title, key=f"prod_title_{tag}")
        desc = st.text_area("Description", default_desc, height=90, key=f"prod_desc_{tag}",
                            placeholder="ex. robe noire élégante coton neuf avec étiquette livraison partout au maroc")
        price = st.number_input("Prix de vente (DH)", 1.0, 30000.0, default_price, 5.0, key=f"prod_price_{tag}")
        mkey = st.selectbox("Modèle de catégorie", list(core.PRODUCT_MODELS), index=0, key="prod_model",
                            format_func=lambda k: core.PRODUCT_MODELS[k]["label"])
        st.caption(f'{core.PRODUCT_MODELS[mkey]["acc"]} · {core.PRODUCT_MODELS[mkey]["real"]}')
        pkey = st.selectbox("Modèle de prix", list(core.PRICE_MODELS), key="prod_price_model", format_func=core.PRICE_MODELS.get)
        weights = weights_editor("prod")

    text = f"{title} {desc}".strip()
    has_image = image28 is not None
    x = (image28.astype("float32") / 255.0)[None] if has_image else None
    probs = core.category_probs(mkey, x, [text])[0]
    tprobs = core.text_probs([text])[0]
    cat = int(probs.argmax())
    if has_image:
        est = float(core.price_estimates(pkey, x, [text])[0])
        recon, mse = core.reconstruction(x)
        mse = float(mse[0])
    else:
        est = float(core.category_median_price()[cat])
        recon, mse = None, None
    factors = core.potential_factors([probs.max()], [tprobs[cat]], [price / max(est, 1)], core.known_words([text]),
                                     [mse] if has_image else None)
    score = float(core.potential_score(factors, weights)[0])
    label, kind = verdict(score)

    with right, st.container(key="panel_product"):
        a, b = st.columns([1, 1.1], vertical_alignment="center")
        with a:
            st.html('<div class="eyebrow" style="color:rgba(255,255,255,.85)">Indice de potentiel</div>'
                    + donut([("indice", score, "#22C77E" if kind == "good" else ("#F5A524" if kind == "warn" else "#F2557A")),
                             ("reste", 1 - score, "rgba(255,255,255,0)")], f"{score*100:.0f}", "sur 100", dark=True, width=165)
                    + f'<div style="text-align:center;margin:.3rem 0">{chip(label, kind)}</div>')
        with b:
            st.html(f'<h3 style="margin:0">{core.CLASSES[cat]}</h3><p style="margin:.2rem 0">confiance {probs.max()*100:.0f} % · '
                    f'{core.PRODUCT_MODELS[mkey]["short"]}</p>'
                    f'<p style="margin:.2rem 0">Prix estimé : <b>{est:.0f} DH</b> · votre prix : <b>{price:.0f} DH</b> '
                    f'({(price/est-1)*100:+.0f} %)</p>'
                    + ("" if has_image else '<p class="note" style="color:#E6ECE4 !important">Sans image : catégorie par le texte seul, '
                       'prix = médiane de la catégorie.</p>'))
        with st.container(key="card_factors"):
            st.html("".join(factor_bar(core.FACTOR_LABELS[k][0], v[0], core.FACTOR_LABELS[k][1]) for k, v in factors.items()))

    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1, st.container(key="card_probs"):
        st.html(ctitle("Probabilités par catégorie"))
        dfp = pd.DataFrame({"catégorie": core.CLASSES, "modèle choisi": probs, "texte seul (J4)": tprobs}).melt(
            "catégorie", var_name="source", value_name="probabilité")
        show_chart(alt.Chart(dfp).mark_bar(cornerRadiusEnd=5, height=9).encode(
            y=alt.Y("catégorie:N", sort=core.CLASSES, title=None), x=alt.X("probabilité:Q", axis=alt.Axis(format="%")),
            yOffset="source:N", color=alt.Color("source:N", scale=alt.Scale(range=["#6C4DF6", "#22C77E"]),
                                                legend=alt.Legend(orient="top", title=None))).properties(height=330),
            use_container_width=True)
    with c2, st.container(key="card_image_check"):
        st.html(ctitle("Ce que voit le modèle"))
        if has_image:
            cols = st.columns(4)
            if original is not None:
                cols[0].html(img_tag(core.pil_jpeg(original, 200), "photo", radius=10) + '<p class="note">Photo</p>')
            if cut is not None:
                cols[1].html(img_tag(cut, "détourage", radius=10) + '<p class="note">Détourage</p>')
            cols[2].html(img_tag(gray_png(image28, 140), "entrée 28×28", radius=10) + '<p class="note">Entrée 28×28</p>')
            cols[3].html(img_tag(gray_png(recon[0], 140), "reconstruction", radius=10) + '<p class="note">Reconstruction AE</p>')
            th = core.ae_threshold()
            ok = mse <= th
            st.html(chip("Silhouette proche de l’entraînement" if ok else "Silhouette inhabituelle", "good" if ok else "warn")
                    + f'<p class="note">Erreur de reconstruction {mse:.4f} · seuil figé {th:.4f}.</p>')
            st.caption("La photo est détourée (U²-Net-p), puis convertie comme Fashion-MNIST : gris, rognage, 28 px, centrage, inversion. "
                       "Les modèles voient donc une silhouette : formes nettes (pantalon, robe) bien reconnues ; hauts proches entre eux "
                       "et chaussures en paire vue de dessus plus difficiles. Le titre améliore nettement la prédiction.")
        else:
            st.caption("Ajoutez une photo pour activer les modèles image et le contrôle de l’auto-encodeur.")

    st.caption("Indice heuristique et transparent, construit à partir des modèles des jalons. Les données UC2 ne contiennent ni ventes "
               "par produit ni étiquette « viral » : ce n’est pas une probabilité de vente calibrée. Pour apprendre une vraie "
               "probabilité de vente, importez un historique étiqueté dans « Scorer un catalogue ».")


@st.cache_data(show_spinner=False, max_entries=32)
def _convert(raw: bytes):
    from io import BytesIO
    img = Image.open(BytesIO(raw))
    return core.to_fashion_mnist(img), core.cutout_png(img, 280)
