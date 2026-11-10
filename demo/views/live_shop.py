"""Boutique démo : chaque action du visiteur devient un événement et la probabilité d'achat est recalculée."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demo import core
from demo.theme import chip, hero, img_tag, product_png, show_chart

SHOP_ITEMS = [0, 1, 2, 5, 8, 11, 14, 21]
ACTIONS = {1: "Voir le produit", 2: "Voir les détails", 3: "Ajouter au panier", 4: "Retirer du panier", 5: "Rechercher"}
PACE = {"Rapide": 0.2, "Normal": 0.5, "Attentif": 0.8}


@st.cache_data(show_spinner=False)
def _shop():
    d = core.data("uc2_catalogueA.npz")
    edges = np.quantile(d["prix_val"], [.2, .4, .6, .8])
    durs = core.data("uc2_sessions.npz")
    real = durs["duree_val"][durs["type_val"] != 0].astype("float32")
    items = []
    for i in SHOP_ITEMS:
        c = int(d["categorie_val"][i])
        desc = str(d["description_val"][i]) or "pièce sélectionnée"
        items.append({"i": i, "cat": c, "title": f"{core.CLASSES[c]}", "sub": desc[:40],
                      "price": float(d["prix_val"][i]), "range": int(np.searchsorted(edges, d["prix_val"][i]) + 1),
                      "png": product_png(d["X_val"][i], size=240)})
    return items, {k: float(np.quantile(real, q)) for k, q in PACE.items()}


def _state():
    ss = st.session_state
    ss.setdefault("shop_events", [])
    ss.setdefault("shop_cart", [])
    ss.setdefault("replay_step", 0)
    ss.setdefault("replay_play", False)


def _add(event_type, item, durations, pace):
    st.session_state.shop_events.append({
        "event_type": event_type, "category": item["cat"] + 1, "price_range": item["range"],
        "duration": durations[pace], "label": f"{ACTIONS[event_type]} · {item['title']}"})


def _history(events, key):
    """Score après chaque événement (préfixes calculés en un seul lot)."""
    if not events:
        return np.array([])
    df = pd.DataFrame([{**e, "session_id": n} for n in range(1, len(events) + 1) for e in events[:n]])
    _, arr = core.sessions_to_arrays(df)
    return core.predict_purchase(arr, key)


def _score_panel(history, threshold, labels, key_suffix, truth=None):
    p = float(history[-1]) if len(history) else None
    with st.container(key=f"panel_score_{key_suffix}"):
        st.html('<div class="eyebrow" style="color:#E6ECE4">Probabilité d’achat · en direct</div>')
        if p is None:
            st.html('<div class="bigscore">—</div><p>Interagissez avec la boutique pour démarrer une session.</p>')
            return
        delta = (p - float(history[-2])) * 100 if len(history) > 1 else 0.0
        arrow = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "•")
        st.html(f'<div class="bigscore">{p*100:.0f}%</div><p style="margin:.2rem 0 .6rem">{arrow} {delta:+.1f} pts après « {labels[-1]} »</p>')
        hot = p >= threshold
        st.html(chip("Client à cibler (coupon)" if hot else "Sous le seuil coupon", "good" if hot else "neutral")
                + (f' &nbsp;{chip("Achat réel" if truth else "Pas d’achat réel", "good" if truth else "bad")}' if truth is not None else ""))
        hist = pd.DataFrame({"événement": np.arange(1, len(history) + 1), "probabilité": history})
        line = alt.Chart(hist).mark_line(point=alt.OverlayMarkDef(color="#FFFFFF", size=40), color="#FFFFFF", strokeWidth=2.5).encode(
            x=alt.X("événement:Q", axis=alt.Axis(tickMinStep=1, labelColor="#E6ECE4", titleColor="#E6ECE4", grid=False)),
            y=alt.Y("probabilité:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", labelColor="#E6ECE4", titleColor="#E6ECE4", gridColor="#7D8B7C")),
            tooltip=["événement", alt.Tooltip("probabilité:Q", format=".1%")])
        rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(strokeDash=[5, 4], color="#F4E7CF").encode(y=alt.Y("y:Q", title=None))
        show_chart((line + rule).properties(height=190, background="transparent").configure_view(stroke=None),
                        use_container_width=True)
        st.html('<p class="note" style="color:#E6ECE4 !important">Pointillés : seuil coupon figé sur la validation (top 10 %). '
                'Score exploratoire sur une session en cours ; les modèles ont été validés sur des sessions complètes.</p>')


def render():
    _state()
    items, durations = _shop()
    threshold = core.coupon_threshold()
    hero("Boutique en direct",
         "Chaque clic devient un événement de session. Le modèle choisi recalcule la probabilité que ce visiteur achète.",
         icon="cart", eyebrow="Simulation temps réel")

    tab_manual, tab_auto = st.tabs(["Je navigue moi-même", "Client automatique (replay)"])

    with tab_manual:
        c1, c2, c3 = st.columns([1.4, 1, 1])
        key = c1.selectbox("Modèle de prédiction", list(core.PURCHASE_MODELS), key="shop_model",
                           format_func=lambda k: core.PURCHASE_MODELS[k]["label"])
        pace = c2.segmented_control("Temps passé par action", list(PACE), default="Normal", key="shop_pace") or "Normal"
        with c3:
            st.write("")
            if st.button("Nouvelle session", key="shop_reset", use_container_width=True):
                st.session_state.shop_events, st.session_state.shop_cart = [], []
                st.rerun()

        left, right = st.columns([2.15, 1], gap="large")
        with left:
            with st.container(key="panel_shop"):
                s1, s2 = st.columns([3, 1], vertical_alignment="bottom")
                query = s1.text_input("Rechercher dans la boutique", placeholder="robe noire, basket, sac…", key="shop_query")
                if s2.button("Rechercher", key="shop_search", use_container_width=True):
                    hits = core.text_probs([query or "article"])[0].argmax()
                    target = next((it for it in items if it["cat"] == hits), items[0])
                    st.session_state.shop_events.append({"event_type": 5, "category": int(hits) + 1, "price_range": target["range"],
                                                         "duration": durations[pace], "label": f"Rechercher « {query or 'article'} »"})
                for row in range(0, len(items), 4):
                    cols = st.columns(4, gap="medium")
                    for col, it in zip(cols, items[row:row + 4]):
                        with col, st.container(key=f"pcard_shop_{it['i']}"):
                            st.html(img_tag(it["png"], it["title"]))
                            st.html(f'<div class="ptitle">{it["title"]}<div class="psub">{it["sub"]}</div></div>'
                                    f'<div class="pprice">{it["price"]:.2f} DH</div>')
                            if st.button("Détails", key=f"d_{it['i']}"):
                                _add(1, it, durations, pace)
                                _add(2, it, durations, pace)
                            if st.button("Ajouter", key=f"c_{it['i']}", type="primary"):
                                _add(3, it, durations, pace)
                                st.session_state.shop_cart.append(it["i"])
                cart = st.session_state.shop_cart
                if cart:
                    k1, k2 = st.columns([3, 1], vertical_alignment="center")
                    if k2.button("Retirer le dernier", key="shop_remove", use_container_width=True) and cart:
                        removed = cart.pop()
                        it = next(x for x in items if x["i"] == removed)
                        _add(4, it, durations, pace)
                    names = ", ".join(next(x["title"] for x in items if x["i"] == i) for i in cart) or "vide"
                    k1.html(f'<p style="margin:0">Panier ({len(cart)}) : {names}</p>')
        events = st.session_state.shop_events
        history = _history(events, key)
        with right:
            _score_panel(history, threshold, [e["label"] for e in events], "manual")
            with st.container(key="card_log"):
                st.html('<div class="eyebrow">Journal de session</div>')
                if events:
                    rows = "".join(f'<div>{n}. {e["label"]} <span style="opacity:.6">· type {e["event_type"]} · '
                                   f'{history[n-1]*100:.0f}%</span></div>' for n, e in enumerate(events, 1))
                    st.html(f'<div class="eventlog">{rows}</div>')
                else:
                    st.caption("Aucun événement pour l’instant.")
                if len(events) > core.MAX_EVENTS:
                    st.caption(f"Plus de {core.MAX_EVENTS} événements : seuls les derniers sont lus par le modèle.")
        with st.expander("Comment les clics deviennent des événements"):
            st.markdown(
                "Les jeux UC2 fournissent des codes d’événement 1–5 **sans dictionnaire officiel**. Pour la démo, l’interface leur associe : "
                "Voir le produit = 1, Voir les détails = 2, Ajouter au panier = 3, Retirer du panier = 4, Rechercher = 5. "
                "Dans la validation, le code 3 en dernier événement précède un achat dans 13 sessions sur 14, d’où ce choix pour le panier.\n\n"
                "Catégorie de l’événement = classe du produit + 1 ; gamme de prix = quintile du prix dans le catalogue A ; "
                "durée = quantile 20 / 50 / 80 % des durées réelles selon le rythme choisi.")

    with tab_auto:
        _replay(threshold)


def _replay(threshold):
    d = core.data("uc2_sessions.npz")
    c1, c2, c3, c4 = st.columns([1.2, 1.3, .8, .8], vertical_alignment="bottom")
    sid = c1.number_input("Session de validation", 0, len(d["achat_val"]) - 1, 4, key="replay_sid")
    key = c2.selectbox("Modèle", list(core.PURCHASE_MODELS), key="replay_model",
                       format_func=lambda k: core.PURCHASE_MODELS[k]["label"])
    if st.session_state.get("replay_last_sid") != sid:
        st.session_state.replay_last_sid, st.session_state.replay_step, st.session_state.replay_play = sid, 0, False
    if c3.button("Pause" if st.session_state.replay_play else "Lecture", key="replay_toggle", type="primary", use_container_width=True):
        st.session_state.replay_play = not st.session_state.replay_play
        st.rerun()
    if c4.button("Rembobiner", key="replay_reset", use_container_width=True):
        st.session_state.replay_step, st.session_state.replay_play = 0, False
        st.rerun()

    length = int(d["longueur_val"][sid])
    events = [{"event_type": int(d["type_val"][sid, t]), "category": int(d["categorie_val"][sid, t]),
               "price_range": int(d["prix_val"][sid, t]), "duration": float(d["duree_val"][sid, t]),
               "label": f"{ACTIONS[int(d['type_val'][sid, t])]} · catégorie {int(d['categorie_val'][sid, t])}"} for t in range(length)]
    full = _history(events, key)

    @st.fragment(run_every=0.7 if st.session_state.replay_play else None)
    def live():
        ss = st.session_state
        if ss.replay_play and ss.replay_step < length:
            ss.replay_step += 1
        elif ss.replay_play:
            ss.replay_play = False
        step = max(ss.replay_step, 0)
        left, right = st.columns([1.4, 1], gap="large")
        with right:
            truth = bool(d["achat_val"][sid]) if step == length else None
            _score_panel(full[:step], threshold, [e["label"] for e in events[:step]], "auto", truth)
        with left, st.container(key="card_replay"):
            st.html(f'<div class="eyebrow">Parcours rejoué · {step}/{length} événements</div>')
            st.progress(step / length)
            if step:
                tail = events[max(0, step - 12):step]
                rows = "".join(f'<div>{max(0, step - 12) + n}. {e["label"]} <span style="opacity:.6">· type {e["event_type"]} · '
                               f'{full[max(0, step - 12) + n - 1]*100:.0f}%</span></div>' for n, e in enumerate(tail, 1))
                st.html(f'<div class="eventlog">{rows}</div>')
            else:
                st.caption("Appuyez sur Lecture : les événements réels de la session arrivent un par un.")
            if step == length:
                st.caption("Fin de session : l’étiquette réelle de validation est affichée à droite.")
    live()
