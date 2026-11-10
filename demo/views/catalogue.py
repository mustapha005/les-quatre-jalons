"""Catalogue complet : indice de potentiel par produit, puis modèle de vente entraîné sur les étiquettes de l'utilisateur."""
from io import BytesIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demo import core
from demo.theme import ctitle, donut, gradient, hero, kpi, legend_rows, show_chart
from demo.views.product import weights_editor

FORMAT = pd.DataFrame([
    ["product_id", "Identifiant", "non"],
    ["title", "Titre de l’annonce", "title ou description"],
    ["description", "Texte de l’annonce", "title ou description"],
    ["price", "Prix en DH", "oui"],
    ["image", "Nom du fichier image présent dans le ZIP", "non"],
    ["vendu, ventes, viral…", "Toute colonne cible 0/1 pour l’entraînement", "non"],
], columns=["colonne", "contenu", "obligatoire"])

TRAINERS = {
    "logreg": "Régression logistique",
    "rf": "Forêt aléatoire",
    "gb": "Gradient boosting",
}


@st.cache_data(show_spinner=False)
def _images(raw: bytes):
    return core.read_zip_images(raw)


def render():
    hero("Scorer un catalogue",
         "Importez vos produits : chaque ligne reçoit catégorie, prix estimé et indice de potentiel. Si vous avez un historique "
         "de ventes, entraînez un modèle de probabilité de vente sur vos propres étiquettes.", icon="chart", eyebrow="Catalogue")

    left, right = st.columns([1.05, 1], gap="large")
    with left, st.container(key="card_cat_upload"):
        csv = st.file_uploader("Catalogue CSV", type=["csv"], key="cat_csv")
        zipf = st.file_uploader("Images (ZIP, facultatif)", type=["zip"], key="cat_zip")
        use_sample = st.toggle("Utiliser l’exemple : 70 photos réelles", value=True, key="cat_sample") if csv is None else False
        mkey = st.selectbox("Modèle de catégorie", list(core.PRODUCT_MODELS), index=0, key="cat_model",
                            format_func=lambda k: core.PRODUCT_MODELS[k]["label"])
        pkey = st.selectbox("Modèle de prix", list(core.PRICE_MODELS), key="cat_price_model", format_func=core.PRICE_MODELS.get)
        sample_csv, sample_zip = core.sample_catalogue()
        d1, d2 = st.columns(2)
        d1.download_button("Modèle CSV", sample_csv, "catalogue_exemple.csv", "text/csv", key="cat_tpl", use_container_width=True)
        d2.download_button("Images exemple", sample_zip, "images_exemple.zip", "application/zip", key="cat_tpl_zip", use_container_width=True)
    with right, st.container(key="card_cat_format"):
        st.html(ctitle("Format attendu"))
        st.dataframe(FORMAT, hide_index=True, use_container_width=True)
        st.caption("Noms acceptés aussi : titre, prix, nom, photo… Sans image, la catégorie vient du texte seul et le prix estimé "
                   "est la médiane de la catégorie. Chaque photo est détourée (U²-Net) puis convertie au format 28×28 des jalons, comme Fashion-MNIST.")
        weights = weights_editor("cat")

    if csv is not None:
        try:
            raw = pd.read_csv(csv)
        except Exception as e:
            st.error(f"Lecture du CSV impossible : {e}")
            return
        with st.spinner("Détourage et conversion des photos…"):
            images = _images(zipf.getvalue()) if zipf is not None else {}
    elif use_sample:
        raw = pd.read_csv(BytesIO(sample_csv))
        images = core.sample_catalogue_images()
    else:
        st.info("Importez un catalogue ou activez l’exemple.")
        return

    try:
        with st.spinner("Analyse du catalogue…"):
            res = core.score_products(raw, mkey, pkey, images, weights)
    except ValueError as e:
        st.error(str(e))
        return

    n = len(res)
    q_hi, q_lo = np.quantile(res["potentiel"], [.8, .2]) if n >= 5 else (res["potentiel"].max(), res["potentiel"].min())
    res["groupe"] = np.where(res["potentiel"] >= q_hi, "Fort potentiel", np.where(res["potentiel"] <= q_lo, "Faible potentiel", "Intermédiaire"))

    with st.container(key="panel_cat_results"):
        body, side, leg = st.columns([2.4, .9, 1], vertical_alignment="center")
        with body:
            st.html('<h2 style="margin:0 0 .9rem">Vue d’ensemble du catalogue</h2>')
            m = st.columns(4)
            m[0].html(kpi(f"{n}", "Produits"))
            m[1].html(kpi(f"{int((res['image'] == 'oui').sum())}", "Avec photo"))
            m[2].html(kpi(f"{res['potentiel'].mean()*100:.0f}", "Indice moyen /100"))
            m[3].html(kpi(f"{(res['prix (DH)'] > res['prix estimé (DH)']).mean()*100:.0f} %", "Prix > estimation"))
        g = res["groupe"].value_counts()
        parts = [("Fort potentiel", int(g.get("Fort potentiel", 0)), "#22C77E"),
                 ("Intermédiaire", int(g.get("Intermédiaire", 0)), "#B9A9FF"),
                 ("Faible potentiel", int(g.get("Faible potentiel", 0)), "#F2557A")]
        side.html(donut(parts, f"{n}", "produits", dark=True, width=160))
        leg.html(legend_rows([(c, l, str(v)) for l, v, c in parts]))

    cols_show = ["product_id", "titre", "catégorie prédite", "prix (DH)", "prix estimé (DH)", "potentiel"]
    cfg = {"potentiel": st.column_config.ProgressColumn("indice", format="percent", min_value=0, max_value=1)}
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(key="card_cat_high"):
        st.html(ctitle("Fort potentiel · top 20 %"))
        st.dataframe(res.sort_values("potentiel", ascending=False).head(8)[cols_show], hide_index=True,
                     use_container_width=True, column_config=cfg)
    with c2, st.container(key="card_cat_low"):
        st.html(ctitle("Faible potentiel · bas 20 %"))
        st.dataframe(res.sort_values("potentiel").head(8)[cols_show], hide_index=True, use_container_width=True, column_config=cfg)

    c3, c4 = st.columns([1.2, 1], gap="large")
    with c3, st.container(key="card_cat_cat"):
        st.html(ctitle("Indice moyen par catégorie"))
        agg = res.groupby("catégorie prédite").agg(indice=("potentiel", "mean"), produits=("potentiel", "size")).reset_index()
        show_chart(alt.Chart(agg).mark_bar(color=gradient("#6C4DF6", .95, .45, horizontal=True), cornerRadiusEnd=6, height=16).encode(
            y=alt.Y("catégorie prédite:N", sort="-x", title=None), x=alt.X("indice:Q", axis=alt.Axis(format="%")),
            tooltip=["catégorie prédite", alt.Tooltip("indice:Q", format=".0%"), "produits"]).properties(height=300),
            use_container_width=True)
    with c4, st.container(key="card_cat_price"):
        st.html(ctitle("Prix saisi vs prix estimé"))
        show_chart(alt.Chart(res).mark_circle(size=70, opacity=.75, stroke="#FFFFFF", strokeWidth=1).encode(
            x=alt.X("prix estimé (DH):Q", scale=alt.Scale(type="log")), y=alt.Y("prix (DH):Q", scale=alt.Scale(type="log")),
            color=alt.Color("groupe:N", scale=alt.Scale(domain=["Fort potentiel", "Intermédiaire", "Faible potentiel"],
                                                        range=["#22C77E", "#A99BFF", "#F2557A"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["titre", "prix (DH)", "prix estimé (DH)", alt.Tooltip("potentiel:Q", format=".0%")]).properties(height=300),
            use_container_width=True)

    st.subheader("Catalogue complet")
    st.dataframe(res.sort_values("potentiel", ascending=False), hide_index=True, use_container_width=True, column_config=cfg)
    st.download_button("Télécharger les scores (CSV)", res.to_csv(index=False).encode("utf-8"), "scores_catalogue.csv", "text/csv",
                       type="primary", key="cat_dl")

    _trainer(res)


def _features(res: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    probs = res.attrs["probs"]
    words = core.bow(4, res.attrs["features_text"], "bow_vocabulary.json")
    vocab = list(core.artifact(4, "bow_vocabulary.json"))
    num = np.column_stack([np.log1p(res["prix (DH)"]), np.log1p(res["prix estimé (DH)"]),
                           np.log(res["prix (DH)"] / res["prix estimé (DH)"].clip(lower=1)),
                           res[[c for c in res.columns if c.startswith("f_")]].fillna(0.5).to_numpy(),
                           res["image"].eq("oui").astype(float)])
    names = ["log prix", "log prix estimé", "écart au prix estimé"] + [c[2:] for c in res.columns if c.startswith("f_")] + ["a une image"]
    names += [f"p({c})" for c in core.CLASSES] + [f"mot « {w} »" for w in vocab]
    return np.column_stack([num, probs, words]).astype("float32"), names


def _binary(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    if s.dtype == bool:
        return series.astype("float")
    if pd.api.types.is_numeric_dtype(s):
        u = set(np.unique(s))
        if u <= {0, 1}:
            return series.astype("float")
        return (series > s.median()).astype("float").where(series.notna())
    m = {"oui": 1, "yes": 1, "true": 1, "vendu": 1, "viral": 1, "non": 0, "no": 0, "false": 0, "invendu": 0}
    mapped = s.astype(str).str.lower().str.strip().map(m)
    return series.astype(str).str.lower().str.strip().map(m) if mapped.notna().all() else None


def _trainer(res: pd.DataFrame):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    st.write("")
    with st.container(key="panel_trainer"):
        st.html('<h2 style="margin:0">Entraîner sur vos étiquettes</h2><p>Avec une colonne cible (vendu, ventes, viral…), l’application apprend '
                'une vraie probabilité de vente à partir des sorties des modèles, du prix et des mots de l’annonce, évaluée par validation croisée.</p>')
        base = {"product_id", "titre", "catégorie prédite", "confiance", "prix (DH)", "prix estimé (DH)", "source estimation",
                "potentiel", "image", "groupe"}
        candidates = [c for c in res.columns if c not in base and not c.startswith("f_")]
        if not candidates:
            st.html('<p>Aucune colonne cible détectée. Ajoutez par exemple <b>vendu</b> (0/1) ou <b>ventes</b> (nombre, coupé à la médiane) '
                    'à votre CSV pour activer cette section.</p>')
            return
        c1, c2 = st.columns(2)
        target = c1.selectbox("Colonne cible", candidates, key="train_target")
        algo = c2.selectbox("Algorithme", list(TRAINERS), format_func=TRAINERS.get, key="train_algo")
        y = _binary(res[target])
        if y is None:
            st.html('<p>Cette colonne n’est pas binaire (0/1, oui/non) ni numérique.</p>')
            return
        keep = y.notna().to_numpy()
        yv = y[keep].astype(int).to_numpy()
        counts = np.bincount(yv, minlength=2)
        if len(yv) < 20 or counts.min() < 5:
            st.html(f'<p>Pas assez d’exemples : il faut au moins 20 lignes et 5 par classe (actuellement {counts[0]} / {counts[1]}).</p>')
            return
        go = st.button("Entraîner et évaluer", type="primary", key="train_go")
    if not go and "train_result" not in st.session_state:
        return
    X, names = _features(res)
    X = X[keep]
    model = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
             "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
             "gb": GradientBoostingClassifier(random_state=42)}[algo]
    if go:
        folds = StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True, random_state=42)
        with st.spinner("Validation croisée…"):
            oof = cross_val_predict(model, X, yv, cv=folds, method="predict_proba")[:, 1]
            model.fit(X, yv)
        if algo == "logreg":
            imp = np.abs(model[-1].coef_[0])
        else:
            imp = model.feature_importances_
        st.session_state.train_result = {"oof": oof, "imp": imp, "auc": roc_auc_score(yv, oof), "target": target, "algo": algo,
                                         "ids": res.loc[keep, "product_id"].to_numpy(), "titles": res.loc[keep, "titre"].to_numpy(),
                                         "y": yv, "heur": res.loc[keep, "potentiel"].to_numpy()}
    r = st.session_state.train_result
    if r["target"] != target or r["algo"] != algo or len(r["y"]) != len(yv):
        st.info("Paramètres modifiés : relancez l’entraînement.")
        return
    heur_auc = roc_auc_score(r["y"], r["heur"])
    with st.container(key="card_train_kpi"):
        m = st.columns(3)
        m[0].html(kpi(f"{r['auc']:.3f}", "AUC validation croisée", "modèle entraîné", good=r["auc"] >= heur_auc))
        m[1].html(kpi(f"{heur_auc:.3f}", "AUC de l’indice heuristique", "référence", good=True))
        m[2].html(kpi(f"{r['y'].mean()*100:.1f} %", "Taux positif"))
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1, st.container(key="card_train_imp"):
        st.html(ctitle("Signaux les plus utiles"))
        top = np.argsort(-r["imp"])[:12]
        show_chart(alt.Chart(pd.DataFrame({"signal": np.array(names)[top], "importance": r["imp"][top]})).mark_bar(
            color=gradient("#6C4DF6", .95, .45, horizontal=True), cornerRadiusEnd=5, height=14).encode(y=alt.Y("signal:N", sort="-x", title=None), x="importance:Q").properties(height=320),
            use_container_width=True)
    with c2, st.container(key="card_train_pred"):
        st.html(ctitle("Probabilité de vente (hors échantillon)"))
        out = pd.DataFrame({"product_id": r["ids"], "titre": r["titles"], "probabilité de vente": r["oof"].round(3), r["target"]: r["y"]})
        out = out.sort_values("probabilité de vente", ascending=False)
        st.dataframe(out, hide_index=True, use_container_width=True, height=320,
                     column_config={"probabilité de vente": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.download_button("Télécharger les probabilités", out.to_csv(index=False).encode("utf-8"), "probabilites_vente.csv",
                           "text/csv", key="train_dl")
    st.caption("Chaque probabilité est prédite par un modèle qui n’a pas vu la ligne (validation croisée). Avec peu de lignes, l’AUC "
               "varie fortement : comparez-la à l’indice heuristique avant de vous y fier.")
