"""Import d'une base de parcours clients et prédiction de l'achat par session."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demo import core
from demo.theme import ctitle, donut, hero, kpi, show_chart

FORMAT = pd.DataFrame([
    ["session_id", "Identifiant du visiteur ou de la session", "S001", "oui"],
    ["event_type", "Code d’événement 1–5 (format UC2)", "1", "oui"],
    ["category", "Catégorie du produit consulté, 1–10", "7", "oui"],
    ["price_range", "Gamme de prix, 1–5", "3", "oui"],
    ["duration", "Durée de l’événement (même échelle que UC2, ≈ 1,5–4,5)", "3.1", "oui"],
    ["step", "Ordre de l’événement dans la session", "4", "non"],
    ["achat", "Étiquette 0/1 si connue : active l’évaluation", "1", "non"],
], columns=["colonne", "contenu", "exemple", "obligatoire"])


def render():
    hero("Prédire les achats",
         "Importez votre base de parcours clients : une ligne par événement. Chaque session reçoit une probabilité d’achat et un rang de ciblage.",
         icon="upload", eyebrow="Base de sessions")

    left, right = st.columns([1.05, 1], gap="large")
    with left, st.container(key="card_upload_sessions"):
        key = st.selectbox("Modèle", list(core.PURCHASE_MODELS), format_func=lambda k: core.PURCHASE_MODELS[k]["label"], key="sess_model")
        st.caption(core.PURCHASE_MODELS[key]["info"])
        compare = st.toggle("Comparer avec l’autre modèle", key="sess_compare")
        up = st.file_uploader("Fichier CSV des événements", type=["csv"], key="sess_file")
        use_sample = st.toggle("Utiliser l’exemple : 200 sessions de validation", value=True, key="sess_sample") if up is None else False
        st.download_button("Télécharger le modèle de fichier", core.sample_sessions_csv(), "sessions_exemple.csv", "text/csv", key="sess_tpl")
    with right, st.container(key="card_format_sessions"):
        st.html(ctitle("Format attendu"))
        st.dataframe(FORMAT, hide_index=True, use_container_width=True)
        st.caption("Noms acceptés aussi : session, type, categorie, gamme_prix, duree, purchase… Les sessions de plus de 60 événements "
                   "sont tronquées aux 60 derniers.")

    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Lecture du CSV impossible : {e}")
            return
    elif use_sample:
        from io import BytesIO
        raw = pd.read_csv(BytesIO(core.sample_sessions_csv()))
    else:
        st.info("Importez un fichier ou activez l’exemple.")
        return

    try:
        df, warnings = core.validate_sessions(raw)
    except ValueError as e:
        st.error(str(e))
        return
    for w in warnings:
        st.warning(w)

    with st.spinner("Prédiction des sessions…"):
        ids, arr = core.sessions_to_arrays(df)
        p = core.predict_purchase(arr, key)
        other = [k for k in core.PURCHASE_MODELS if k != key][0]
        q = core.predict_purchase(arr, other) if compare else None

    n = len(p)
    k = max(1, int(np.ceil(core.COUPON_FRACTION * n)))
    order = np.argsort(-p, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    threshold = core.coupon_threshold()
    res = pd.DataFrame({"session_id": ids, "événements": arr["longueur"], "probabilité": p.round(4), "rang": rank,
                        "top 10 %": rank <= k})
    if key == "gru":
        res["au-dessus du seuil figé"] = p >= threshold
    if compare:
        res[f"probabilité {core.PURCHASE_MODELS[other]['short']}"] = q.round(4)
    labels = None
    if "achat" in df.columns:
        labels = df.groupby("session_id", sort=False)["achat"].first().reindex(ids).to_numpy()
        if pd.isna(labels).any():
            labels = None
        else:
            res["achat réel"] = labels.astype(int)
    res = res.sort_values("rang")

    st.write("")
    with st.container(key="panel_sess_results"):
        body, side = st.columns([3, 1], vertical_alignment="center")
        with body:
            st.html(f'<h2 style="margin:0 0 .9rem">Résultats · {core.PURCHASE_MODELS[key]["label"]}</h2>')
            m = st.columns(4)
            m[0].html(kpi(f"{n}", "Sessions analysées"))
            m[1].html(kpi(f"{p.mean()*100:.1f} %", "Probabilité moyenne"))
            m[2].html(kpi(f"{k}", "Ciblées (top 10 %)"))
            m[3].html(kpi(f"{p[order[k-1]]*100:.1f} %", "Seuil du top 10 %"))
            if labels is not None:
                met = core.ranking_metrics(labels.astype(int), p)
                st.write("")
                e = st.columns(4)
                for col, (name, val) in zip(e, met.items()):
                    col.html(kpi(f"{val:.3f}" if name in ("AUC", "Précision moyenne") else f"{val*100:.1f} %", name))
        with side:
            hot = int((p >= 0.5).sum())
            st.html(donut([("Probables", hot, "#22C77E"), ("Peu probables", n - hot, "#FFFFFF")], f"{hot/n*100:.0f}%",
                          "sessions ≥ 50 %", dark=True, width=170))
            if compare:
                met2 = core.ranking_metrics(labels.astype(int), q)
                st.html(f'<p>{core.PURCHASE_MODELS[other]["short"]} sur le même fichier : AUC {met2.get("AUC", float("nan")):.3f} · '
                        f'P@10 % {met2["P@10 %"]*100:.1f} %</p>')

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1, st.container(key="card_sess_hist"):
        st.html(ctitle("Distribution des probabilités"))
        hist = pd.DataFrame({"probabilité": p, "groupe": np.where(rank <= k, "Top 10 % ciblé", "Autres")})
        if labels is not None:
            hist["groupe"] = np.where(labels.astype(int) == 1, "Achat réel", "Pas d’achat")
        dom = ["Achat réel", "Pas d’achat"] if labels is not None else ["Top 10 % ciblé", "Autres"]
        chart = alt.Chart(hist).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, binSpacing=3).encode(
            x=alt.X("probabilité:Q", bin=alt.Bin(maxbins=20), axis=alt.Axis(format="%", grid=False), title="probabilité d’achat"),
            y=alt.Y("count():Q", title="sessions", stack=True),
            color=alt.Color("groupe:N", scale=alt.Scale(domain=dom, range=["#22C77E", "#8F78FF"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["groupe", alt.Tooltip("count():Q", title="sessions")])
        show_chart(chart.properties(height=260), use_container_width=True)
    with c2, st.container(key="card_sess_top"):
        st.html(ctitle("Sessions les plus proches d’un achat"))
        st.dataframe(res.head(10)[["session_id", "événements", "probabilité"] + (["achat réel"] if labels is not None else [])],
                     hide_index=True, use_container_width=True,
                     column_config={"probabilité": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    if compare:
        with st.container(key="card_sess_compare"):
            st.html(ctitle("Accord entre les deux modèles"))
            sc = pd.DataFrame({core.PURCHASE_MODELS[key]["short"]: p, core.PURCHASE_MODELS[other]["short"]: q})
            show_chart(alt.Chart(sc).mark_circle(size=60, color="#6C4DF6", opacity=.55, stroke="#FFFFFF", strokeWidth=1).encode(
                x=alt.X(f"{core.PURCHASE_MODELS[key]['short']}:Q", axis=alt.Axis(format="%")),
                y=alt.Y(f"{core.PURCHASE_MODELS[other]['short']}:Q", axis=alt.Axis(format="%"))).properties(height=280),
                use_container_width=True)

    st.subheader("Toutes les sessions")
    st.dataframe(res, hide_index=True, use_container_width=True,
                 column_config={"probabilité": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.download_button("Télécharger les prédictions (CSV)", res.to_csv(index=False).encode("utf-8"), "predictions_achat.csv",
                       "text/csv", type="primary", key="sess_dl")
    st.caption("Ciblage budgétaire : top 10 % par rang, comme au jalon 4. Le seuil figé (GRU) vient de la validation et ne garantit "
               "pas exactement 10 % sur un autre lot. Les modèles ont été entraînés sur des sessions simulées UC2 : sur vos données, "
               "évaluez-les avec la colonne achat avant toute décision.")
