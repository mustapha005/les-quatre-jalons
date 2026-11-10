"""Pages de l'application, partagées entre l'entrée et les liens internes."""
import streamlit as st

from demo.views import catalogue, home, jalons, live_shop, product, sessions

PAGES = {
    "accueil": st.Page(home.render, title="Accueil", url_path="accueil", default=True),
    "boutique": st.Page(live_shop.render, title="Boutique en direct", url_path="boutique"),
    "achats": st.Page(sessions.render, title="Prédire les achats", url_path="achats"),
    "produit": st.Page(product.render, title="Potentiel produit", url_path="produit"),
    "catalogue": st.Page(catalogue.render, title="Scorer un catalogue", url_path="catalogue"),
    "jalons": st.Page(jalons.render, title="Les quatre jalons", url_path="jalons"),
}
