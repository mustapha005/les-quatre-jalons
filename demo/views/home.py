import numpy as np
import streamlit as st

from demo import core
from demo.theme import hero, icon_svg, img_tag, product_png

FEATURES = [
    ("live", "Boutique en direct", "Naviguez dans une boutique démo : chaque clic met à jour la probabilité d’achat.", "boutique"),
    ("upload", "Prédire les achats", "Importez vos parcours clients (CSV) et classez les sessions les plus proches d’un achat.", "achats"),
    ("tag", "Potentiel produit", "Photo, titre, description et prix : catégorie, prix estimé et indice de potentiel de vente.", "produit"),
    ("chart", "Scorer un catalogue", "Classez tout un catalogue, puis entraînez un modèle sur vos propres étiquettes de ventes.", "catalogue"),
]

SHOWCASE = [12, 3, 7]


def render():
    hero("AtlasShop Intelligence",
         "Les modèles des quatre jalons UC2 appliqués à une marketplace : parcours clients, fiches produit et catalogues.",
         icon="sparkle", eyebrow="Groupe 13 · Catalogue multimodal")

    cols = st.columns(4, gap="medium")
    for col, (icon, title, text, url) in zip(cols, FEATURES):
        with col:
            with st.container(key=f"card_feat_{url}", height="stretch"):
                st.html(f'<div class="feature"><div class="badge">{icon_svg(icon)}</div><h3>{title}</h3><p>{text}</p></div>')
                st.page_link(_page(url), label="Ouvrir  →")

    st.write("")
    d = core.data("uc2_catalogueA.npz")
    idx = np.array(SHOWCASE)
    images = d["X_val"][idx].astype("float32") / 255.0
    texts = [str(t) for t in d["description_val"][idx]]
    probs = core.category_probs("cnn_a", images, texts)
    prices = core.price_estimates("huber_a", images, texts)

    with st.container(key="panel_showcase"):
        st.html('<h2 style="text-align:center;font-size:2.6rem;margin:.2rem 0 1.2rem">Vous pourriez aussi aimer</h2>')
        pc = st.columns(3, gap="large")
        for k, col in enumerate(pc):
            i = int(idx[k])
            with col:
                with st.container(key=f"pcard_home_{k}"):
                    st.html(img_tag(product_png(d["X_val"][i]), CLASSES_LABEL(probs[k])))
                    st.html(f'<div class="ptitle">{core.CLASSES[int(probs[k].argmax())]} · {texts[k][:34]}</div>'
                            f'<div class="pprice">{float(d["prix_val"][i]):.2f} DH · estimé {float(prices[k]):.0f} DH</div>')
                    b1, b2 = st.columns(2)
                    b1.page_link(_page("produit"), label="Analyser")
                    b2.page_link(_page("boutique"), label="Acheter")
        st.html('<p style="text-align:center;opacity:.85;margin-top:1rem">Catégorie (CNN, jalon 2) et prix (MLP log-Huber, jalon 1) '
                'calculés à l’instant sur trois annonces de validation.</p>')

    with st.expander("Données, modèles et limites"):
        st.markdown(
            "- **Données** : images Fashion-MNIST (28×28, niveaux de gris) ; descriptions, prix et sessions simulés pour le cours. "
            "Seules 200 lignes de validation par jeu sont distribuées avec l’application.\n"
            "- **Modèles** : les huit checkpoints `.keras` des notebooks, lus sans réentraînement.\n"
            "- **Photos réelles** : elles sont converties au format 28×28 des jalons ; l’auto-encodeur signale si l’image sort de la distribution apprise.\n"
            "- **Indice de potentiel** : combinaison transparente de sorties de modèles (prix, confiance, cohérence, annonce, image). "
            "Aucun label « viral » ou « vendu » n’existe dans les données UC2 ; pour un vrai modèle de ventes, utilisez "
            "« Entraîner sur vos étiquettes » dans *Scorer un catalogue*.\n"
            "- **Sessions** : les codes d’événement 1–5 sont fournis sans dictionnaire. La boutique démo leur associe des actions d’interface (hypothèse affichée).")


def CLASSES_LABEL(p):
    return core.CLASSES[int(np.argmax(p))]


def _page(url):
    from demo.nav import PAGES
    return PAGES[url]
