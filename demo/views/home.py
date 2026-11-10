import numpy as np
import streamlit as st

from demo import core
from demo.theme import hero, icon_svg, img_tag, kpi

FEATURES = [
    ("live", "Boutique en direct", "Naviguez dans une boutique démo : chaque clic met à jour la probabilité d’achat.", "boutique"),
    ("upload", "Prédire les achats", "Importez vos parcours clients (CSV) et classez les sessions les plus proches d’un achat.", "achats"),
    ("tag", "Potentiel produit", "Photo, titre, description et prix : catégorie, prix estimé et indice de potentiel de vente.", "produit"),
    ("chart", "Scorer un catalogue", "Classez tout un catalogue, puis entraînez un modèle sur vos propres étiquettes de ventes.", "catalogue"),
]

SHOWCASE = [30, 40, 10]  # robe, veste, pantalon parmi les photos réelles embarquées


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
    r = core.real_photos()
    idx = np.array(SHOWCASE)
    images = r["X"][idx].astype("float32") / 255.0
    titles = [str(t) for t in r["title"][idx]]
    probs = core.category_probs("fusion", images, titles)
    prices = core.price_estimates("huber_a", images, titles)

    with st.container(key="card_kpis"):
        k = st.columns(5)
        for col, (v, l) in zip(k, [("8", "modèles entraînés"), ("4", "jalons UC2"), ("70", "photos réelles"),
                                   ("200", "sessions de validation"), ("81 %", "catégorie juste · photo + titre")]):
            col.html(kpi(v, l))
    st.write("")
    with st.container(key="glass_showcase"):
        st.html('<div class="ctitle" style="font-size:1.35rem"><span>Vous pourriez aussi aimer</span>'
                '<span class="pill">Prédictions en direct</span></div>')
        pc = st.columns(3, gap="large")
        for k, col in enumerate(pc):
            name = str(r["name"][idx[k]])
            with col:
                with st.container(key=f"pcard_home_{k}"):
                    st.html(img_tag(core.square_jpeg(name), titles[k]))
                    st.html(f'<div class="ptitle">{titles[k]}</div>'
                            f'<div class="pprice">{float(prices[k]):.0f} DH · prédit : {core.CLASSES[int(probs[k].argmax())]}</div>')
                    b1, b2 = st.columns(2)
                    b1.page_link(_page("produit"), label="Analyser")
                    b2.page_link(_page("boutique"), label="Acheter")
        st.html('<p class="note" style="text-align:center;margin-top:1rem">Photos réelles (domaine public). Chaque photo est détourée puis '
                'convertie en 28×28 ; catégorie par la fusion photo + texte (jalons 3 et 4) et prix par le MLP log-Huber (jalon 1), calculés à l’instant.</p>')

    with st.expander("Données, modèles et limites"):
        st.markdown(
            "- **Données d’entraînement** : images Fashion-MNIST (28×28, niveaux de gris) ; descriptions, prix et sessions simulés pour le cours. "
            "Seules 200 lignes de validation par jeu sont distribuées avec l’application (page *Les quatre jalons*).\n"
            "- **Photos réelles** : 70 photos du jeu *clothing-dataset-small* (Alexey Grigorev, CC0 domaine public) servent d’exemples "
            "dans la boutique, la fiche produit et le catalogue.\n"
            "- **Modèles** : les huit checkpoints `.keras` des notebooks, lus sans réentraînement.\n"
            "- **Conversion des photos** : détourage par U²-Net-p (Apache 2.0), puis la chaîne de Fashion-MNIST (gris, rognage, 28 px, centrage, inversion). "
            "Sur 318 photos réelles test, la catégorie est correcte 51 % du temps avec l’image seule (multitâche J3) et 81 % avec un titre simple (fusion J3 + J4) ; "
            "les chaussures photographiées par paire, vue de dessus, sont mal reconnues.\n"
            "- **Indice de potentiel** : combinaison transparente de sorties de modèles (prix, confiance, cohérence, annonce, image). "
            "Aucun label « viral » ou « vendu » n’existe dans les données UC2 ; pour un vrai modèle de ventes, utilisez "
            "« Entraîner sur vos étiquettes » dans *Scorer un catalogue*.\n"
            "- **Sessions** : les codes d’événement 1–5 sont fournis sans dictionnaire. La boutique démo leur associe des actions d’interface (hypothèse affichée).")


def CLASSES_LABEL(p):
    return core.CLASSES[int(np.argmax(p))]


def _page(url):
    from demo.nav import PAGES
    return PAGES[url]
