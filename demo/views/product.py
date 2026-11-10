"""Fiche produit : photo + titre + description + prix → catégorie, prix estimé et indice de potentiel."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from demo import core
from demo.theme import chip, factor_bar, gray_png, hero, img_tag, verdict, show_chart


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

    d = core.data("uc2_catalogueA.npz")
    left, right = st.columns([1, 1.25], gap="large")
    with left, st.container(key="card_product_form"):
        source = st.segmented_control("Image", ["Ma photo", "Annonce exemple"], default="Annonce exemple", key="prod_source") or "Annonce exemple"
        image28, original = None, None
        if source == "Ma photo":
            up = st.file_uploader("Photo du produit", type=["png", "jpg", "jpeg", "webp"], key="prod_img")
            if up is not None:
                original = Image.open(up)
                image28 = core.to_fashion_mnist(original)
            default_title, default_desc, default_price = "", "", 250.0
        else:
            idx = st.slider("Annonce de validation", 0, len(d["X_val"]) - 1, 12, key="prod_idx")
            image28 = d["X_val"][idx]
            default_title = core.CLASSES[int(d["categorie_val"][idx])]
            default_desc = str(d["description_val"][idx])
            default_price = float(d["prix_val"][idx])
        title = st.text_input("Titre", default_title, key=f"prod_title_{source}_{st.session_state.get('prod_idx')}")
        desc = st.text_area("Description", default_desc, height=90, key=f"prod_desc_{source}_{st.session_state.get('prod_idx')}",
                            placeholder="ex. robe noire élégante coton neuf avec étiquette livraison partout au maroc")
        price = st.number_input("Prix de vente (DH)", 1.0, 30000.0, default_price, 5.0, key=f"prod_price_{source}_{st.session_state.get('prod_idx')}")
        mkey = st.selectbox("Modèle de catégorie", list(core.PRODUCT_MODELS), key="prod_model",
                            format_func=lambda k: f'{core.PRODUCT_MODELS[k]["label"]} — exactitude test {core.PRODUCT_MODELS[k]["acc"]}')
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
            st.html(f'<div class="eyebrow" style="color:#E6ECE4">Indice de potentiel</div><div class="bigscore">{score*100:.0f}</div>'
                    f'<div style="margin:.4rem 0">{chip(label, kind)}</div>')
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
        st.html('<div class="eyebrow">Probabilités par catégorie</div>')
        dfp = pd.DataFrame({"catégorie": core.CLASSES, "modèle choisi": probs, "texte seul (J4)": tprobs}).melt(
            "catégorie", var_name="source", value_name="probabilité")
        show_chart(alt.Chart(dfp).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("catégorie:N", sort=core.CLASSES, title=None), x=alt.X("probabilité:Q", axis=alt.Axis(format="%")),
            yOffset="source:N", color=alt.Color("source:N", scale=alt.Scale(range=["#2F3A33", "#AEBBAA"]),
                                                legend=alt.Legend(orient="top", title=None))).properties(height=330),
            use_container_width=True)
    with c2, st.container(key="card_image_check"):
        st.html('<div class="eyebrow">Ce que voit le modèle</div>')
        if has_image:
            i1, i2, i3 = st.columns(3)
            if original is not None:
                i1.image(original, caption="Photo importée", use_container_width=True)
            else:
                i1.html(img_tag(gray_png(image28, 140), "annonce") + '<p class="note">Annonce</p>')
            i2.html(img_tag(gray_png(image28, 140), "entrée 28×28") + '<p class="note">Entrée 28×28</p>')
            i3.html(img_tag(gray_png(recon[0], 140), "reconstruction") + '<p class="note">Reconstruction AE</p>')
            th = core.ae_threshold()
            ok = mse <= th
            st.html(chip("Image dans la distribution" if ok else "Image hors distribution", "good" if ok else "warn")
                    + f'<p class="note">Erreur {mse:.4f} · seuil figé {th:.4f}.</p>')
            if not ok:
                st.caption("Les modèles ont appris sur des vignettes Fashion-MNIST (vêtement centré, fond uni). Une photo très différente "
                           "rend la catégorie et le prix moins fiables : l’indice baisse en conséquence.")
        else:
            st.caption("Ajoutez une photo pour activer les modèles image et le contrôle de l’auto-encodeur.")

    st.caption("Indice heuristique et transparent, construit à partir des modèles des jalons. Les données UC2 ne contiennent ni ventes "
               "par produit ni étiquette « viral » : ce n’est pas une probabilité de vente calibrée. Pour apprendre une vraie "
               "probabilité de vente, importez un historique étiqueté dans « Scorer un catalogue ».")
