"""Les quatre vues d'origine (validation seulement, aucune formation), dans le nouveau style."""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demo import core
from demo.theme import gray_png, hero, img_tag, show_chart

VIEWS = ["1 · MLP et prix", "2 · CNN et auto-encodeur", "3 · Boutique B et multitâche", "4 · Sessions GRU"]


def show_class(prob, true=None):
    k = int(np.argmax(prob))
    st.html(f'<h3 style="margin:.2rem 0">Catégorie prédite : {core.CLASSES[k]}</h3>'
            + (f'<p class="note">Étiquette validation : {core.CLASSES[int(true)]}</p>' if true is not None else ""))
    df = pd.DataFrame({"catégorie": core.CLASSES, "score": np.asarray(prob).ravel()})
    show_chart(alt.Chart(df).mark_bar(color="#4F5D50", cornerRadiusEnd=3).encode(
        y=alt.Y("catégorie:N", sort=core.CLASSES, title=None), x=alt.X("score:Q", axis=alt.Axis(format="%"))).properties(height=250),
        use_container_width=True)


def render():
    hero("Les quatre jalons", "Les modèles retenus dans chaque notebook, appliqués aux exemples de validation inclus.",
         icon="layers", eyebrow="Démonstrateur d’origine")
    page = st.segmented_control("Jalon", VIEWS, default=VIEWS[0], key="jalon_view") or VIEWS[0]
    j = int(page[0])
    if j in (1, 2, 3):
        _catalogue_view(j)
    else:
        _sessions_view()
    with st.expander("Résultats enregistrés du jalon et limites"):
        st.json(core.artifact(j, "test_results.json"))
        st.caption("Résultats du run des notebooks, sur les partitions de test complètes. Ils ne sont pas recalculés sur les 200 exemples "
                   "de validation inclus. Prix, descriptions et sessions pédagogiques simulés ; aucune garantie sur une vraie marketplace.")


def _catalogue_view(j):
    d = core.data("uc2_catalogueB.npz" if j == 3 else "uc2_catalogueA.npz")
    left, right = st.columns([1, 1.4], gap="large")
    with left, st.container(key=f"card_j{j}_input"):
        index = st.slider("Annonce de validation", 0, len(d["X_val"]) - 1, 0, key=f"j{j}_idx")
        image = d["X_val"][index]
        st.html(img_tag(gray_png(image, 224), "Annonce de validation", width=224))
        text = st.text_input("Description (modifiable)", str(d["description_val"][index]), key=f"j{j}_text_{index}")
        st.caption("Modifier la description produit une prédiction exploratoire, sans nouvelle évaluation.")
    x = image[None, ..., None].astype("float32") / 255.0
    with right, st.container(key=f"card_j{j}_out"):
        if j == 1:
            inp = {"image": x.reshape(1, 784), "mots": core.bow(1, text)}
            show_class(core.infer(1, "MLP_reference", inp)[0], d["categorie_val"][index])
            upper = core.price_upper("uc2_catalogueA.npz")
            est = float(np.exp(np.clip(core.infer(1, "price_log_huber", inp).ravel()[0], 0, np.log(upper))))
            a, b = st.columns(2)
            a.metric("Prix proposé (DH)", f"{est:.2f}")
            b.metric("Prix fourni validation (DH)", f"{float(d['prix_val'][index]):.2f}")
        elif j == 2:
            show_class(core.infer(2, "CNN_base", {"image": x, "mots": core.bow(2, text)})[0], d["categorie_val"][index])
        else:
            sc = core.artifact(3, "log_price_scaler.json")
            out = core.infer(3, "MT_warm_lambda_1p0", {"image": x, "mots": core.bow(3, text)})
            show_class(out["categorie"][0], d["categorie_val"][index])
            lp = float(out["logprix"].ravel()[0]) * sc["std"] + sc["mean"]
            est = float(np.exp(np.clip(lp, np.log(1), np.log(core.price_upper("uc2_catalogueB.npz")))))
            a, b = st.columns(2)
            a.metric("Prix multitâche (DH)", f"{est:.2f}")
            b.metric("Prix fourni validation (DH)", f"{float(d['prix_val'][index]):.2f}")
            st.caption("Modèle réservé à B. L’adaptation dégrade la performance sur A : le CNN A reste versionné séparément.")

    if j == 1:
        with st.container(key="card_j1_session"):
            st.subheader("Troisième tâche : achat par MLP")
            s = core.data("uc2_sessions.npz")
            si = st.slider("Session pour le MLP", 0, len(s["achat_val"]) - 1, 0, key="j1_sess")
            arr = {"type": s["type_val"][si:si + 1].astype(int), "categorie": s["categorie_val"][si:si + 1].astype(int),
                   "prix": s["prix_val"][si:si + 1].astype("float32"), "duree": s["duree_val"][si:si + 1].astype("float32"),
                   "longueur": s["longueur_val"][si:si + 1]}
            score = float(core.predict_purchase(arr, "mlp")[0])
            st.metric("Score d’achat MLP", f"{score:.3f}")
            st.caption(f"Étiquette de validation : {int(s['achat_val'][si])}. Score calculé sur la session complète.")
    if j == 2:
        with st.container(key="card_j2_ae"):
            st.subheader("Auto-encodeur : contrôle des images")
            probe = st.selectbox("Image à reconstruire", ["Annonce ci-dessus", "Rectangle synthétique de démonstration",
                                                          "Bruit synthétique de démonstration"], key="j2_probe")
            img = x[..., 0]
            if probe.startswith("Rectangle"):
                img = np.zeros((1, 28, 28), dtype="float32")
                img[:, 9:19, 8:20] = .4
            elif probe.startswith("Bruit"):
                img = np.random.default_rng(42).random((1, 28, 28), dtype=np.float32)
            recon, mse = core.reconstruction(img)
            th = core.ae_threshold()
            a, b, c = st.columns([1, 1, 1.4])
            a.html(img_tag(gray_png(img[0], 200), "Original", width=200) + '<p class="note">Original</p>')
            b.html(img_tag(gray_png(recon[0], 200), "Reconstruction", width=200) + '<p class="note">Reconstruction</p>')
            c.metric("Erreur MSE", f"{float(mse[0]):.6f}")
            c.write(f"Seuil figé : {th:.6f} → **{'rejet' if mse[0] > th else 'acceptation'}**")
            st.caption("Une acceptation ne certifie pas une annonce valide. Les proxies du démonstrateur ne constituent pas un score OOD officiel.")


def _sessions_view():
    d = core.data("uc2_sessions.npz")
    left, right = st.columns([1.3, 1], gap="large")
    with left, st.container(key="card_j4_sess"):
        index = st.slider("Session de validation", 0, len(d["achat_val"]) - 1, 0, key="j4_idx")
        L = int(d["longueur_val"][index])
        arr = {"type": d["type_val"][index:index + 1].astype("int32"), "categorie": d["categorie_val"][index:index + 1].astype("int32"),
               "prix": d["prix_val"][index:index + 1].astype("float32"), "duree": d["duree_val"][index:index + 1].astype("float32"),
               "longueur": d["longueur_val"][index:index + 1]}
        p = float(core.predict_purchase(arr, "gru")[0])
        th = core.coupon_threshold()
        a, b = st.columns(2)
        a.metric("Score d’achat GRU (session complète)", f"{p:.3f}")
        b.metric("Étiquette validation", int(d["achat_val"][index]))
        st.write(f"{L} événements · seuil val complet {th:.6f} · au-dessus du seuil : **{p >= th}**")
        st.dataframe(pd.DataFrame({"type": d["type_val"][index, :L], "categorie": d["categorie_val"][index, :L],
                                   "gamme_prix": d["prix_val"][index, :L], "duree": d["duree_val"][index, :L].astype(float)}),
                     hide_index=True, use_container_width=True, height=240)
        st.caption("Pour un lot, le budget exact est le top 10 % par rang. Le GRU retenu sans masque dépend du padding fixe à 60.")
    with right, st.container(key="card_j4_text"):
        st.subheader("Description → catégorie (BoW)")
        desc = st.text_input("Description à classifier", "robe noire élégante", key="j4_text")
        show_class(core.text_probs([desc])[0])
        st.caption("Le BoW a remporté la sélection validation face aux LSTM de texte (notebook du jalon 4).")
