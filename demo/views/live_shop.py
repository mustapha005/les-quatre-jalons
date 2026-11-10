"""Boutique démo : chaque action du visiteur devient un événement et la probabilité d'achat est recalculée."""
import numpy as np
import pandas as pd
import streamlit as st

from demo import core
from demo.theme import AMBER, GREEN, RED, VIOLET, area_line, chip, gauge, gray_png, hero, img_tag, kpi, show_chart

SHOP_ITEMS = [0, 10, 20, 30, 40, 50, 60, 31]  # une photo réelle par classe + une robe
ACTIONS = {1: "Voir le produit", 2: "Voir les détails", 3: "Ajouter au panier", 4: "Retirer du panier", 5: "Rechercher"}
EVENT_COLORS = {1: VIOLET, 2: "#5AA9FF", 3: GREEN, 4: RED, 5: AMBER}
PACE = {"Rapide": 0.2, "Normal": 0.5, "Attentif": 0.8}
SIZES = {"Chaussures": ["39", "40", "41", "42", "43"]}
DEFAULT_SIZES = ["S", "M", "L", "XL"]
BLURB = {
    "T-shirt": "T-shirt du quotidien, coupe droite et col rond.",
    "Pantalon": "Pantalon confortable, coupe régulière, idéal pour tous les jours.",
    "Pull": "Pull à manches longues, maille douce pour la mi-saison.",
    "Robe": "Robe légère, coupe fluide, se porte en toute saison.",
    "Veste": "Veste zippée, pratique pour les journées fraîches.",
    "Chemise": "Chemise à manches courtes, tissu respirant.",
    "Chaussures": "Paire de chaussures pour un usage quotidien.",
}


@st.cache_data(show_spinner=False)
def _shop():
    d = core.data("uc2_catalogueA.npz")
    edges = np.quantile(d["prix_val"], [.2, .4, .6, .8])
    durs = core.data("uc2_sessions.npz")
    real = durs["duree_val"][durs["type_val"] != 0].astype("float32")
    r = core.real_photos()
    idx = np.array(SHOP_ITEMS)
    titles = [str(t) for t in r["title"][idx]]
    x = r["X"][idx].astype("float32") / 255.0
    prices = core.price_estimates("huber_a", x, titles)
    probs = core.category_probs("fusion", x, titles)
    items = []
    for k, i in enumerate(SHOP_ITEMS):
        c = int(r["categorie"][i])
        price = float(np.round(prices[k] / 5) * 5)
        name = str(r["name"][i])
        items.append({"i": i, "cat": c, "title": titles[k], "price": price, "range": int(np.searchsorted(edges, price) + 1),
                      "png": core.square_jpeg(name), "big": core.square_jpeg(name, 520), "x28": r["X"][i],
                      "pred": core.CLASSES[int(probs[k].argmax())], "conf": float(probs[k].max()), "raw_price": float(prices[k])})
    return items, {k: float(np.quantile(real, q)) for k, q in PACE.items()}


def _state():
    ss = st.session_state
    ss.setdefault("shop_events", [])
    ss.setdefault("shop_cart", [])
    ss.setdefault("shop_view", "grid")
    ss.setdefault("replay_step", 0)
    ss.setdefault("replay_play", False)


def _add(event_type, item, durations, pace, label=None):
    st.session_state.shop_events.append({
        "event_type": event_type, "category": item["cat"] + 1, "price_range": item["range"],
        "duration": durations[pace], "label": label or f"{ACTIONS[event_type]} · {item['title']}"})


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
        st.html('<div class="eyebrow" style="color:rgba(255,255,255,.85)">Probabilité d’achat · en direct</div>')
        if p is None:
            st.html(gauge(0, "en attente", dark=True, threshold=threshold)
                    + '<p style="text-align:center">Interagissez avec la boutique pour démarrer une session.</p>')
            return
        st.html(gauge(p, "probabilité d’achat", dark=True, threshold=threshold))
        delta = (p - float(history[-2])) * 100 if len(history) > 1 else 0.0
        arrow = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "•")
        hot = p >= threshold
        st.html(f'<p style="text-align:center;margin:.1rem 0 .5rem">{arrow} {delta:+.1f} pts après « {labels[-1]} »</p>'
                f'<div style="text-align:center">{chip("Client à cibler · coupon" if hot else "Sous le seuil coupon", "good" if hot else "neutral")}'
                + (f' {chip("Achat réel" if truth else "Pas d’achat réel", "good" if truth else "bad")}' if truth is not None else "")
                + "</div>")
        hist = pd.DataFrame({"événement": np.arange(1, len(history) + 1), "probabilité": history})
        show_chart(area_line(hist, "événement", "probabilité", y_domain=[0, 1], rule=threshold, height=170, dark=True), dark=True)
        st.html('<p class="note" style="color:rgba(255,255,255,.8) !important;margin:0">Ligne verte : seuil coupon figé sur la validation (top 10 %). '
                'Score exploratoire sur une session en cours ; les modèles ont été validés sur des sessions complètes.</p>')


def _event_log(events, history, start=0):
    rows = "".join(f'<div><span class="dot" style="background:{EVENT_COLORS[e["event_type"]]}"></span>{start + n}. {e["label"]} '
                   f'<span style="color:#7C7A99">· type {e["event_type"]} · {history[start + n - 1]*100:.0f}%</span></div>'
                   for n, e in enumerate(events, 1))
    return f'<div class="eventlog">{rows}</div>'


def _shop_header(items, durations, pace):
    cart = st.session_state.shop_cart
    h1, h2, h3, h4 = st.columns([1.05, 2.1, 1.15, 1.05], vertical_alignment="bottom")
    h1.html('<div style="font-family:Poppins;font-weight:700;font-size:1.35rem;color:#4B2FD1">Atlas<span style="color:#22C77E">Shop</span></div>')
    query = h2.text_input("Rechercher", placeholder="robe, pantalon, veste…", key="shop_query", label_visibility="collapsed")
    if h3.button("Rechercher", key="shop_search", use_container_width=True):
        hits = int(core.text_probs([query or "article"])[0].argmax())
        target = next((it for it in items if it["cat"] == hits), items[0])
        st.session_state.shop_events.append({"event_type": 5, "category": hits + 1, "price_range": target["range"],
                                             "duration": durations[pace], "label": f"Rechercher « {query or 'article'} »"})
    if h4.button(f"Panier ({len(cart)})", key="shop_cart_open", use_container_width=True,
                 type="primary" if cart else "secondary"):
        st.session_state.shop_view = "cart"
        st.rerun()


def _product_card(it, durations, pace, prefix="g"):
    with st.container(key=f"pcard_{prefix}_{it['i']}"):
        st.html(img_tag(it["png"], it["title"]))
        st.html(f'<div class="ptitle">{it["title"]}</div><div class="pprice">{it["price"]:.0f} DH</div>')
        if st.button("Voir le produit", key=f"open_{prefix}_{it['i']}"):
            _add(1, it, durations, pace)
            st.session_state.shop_view = it["i"]
            st.rerun()
        if prefix == "g" and st.button("Ajouter", key=f"add_{it['i']}", type="primary"):
            _add(3, it, durations, pace)
            st.session_state.shop_cart.append(it["i"])
            st.rerun()


def _grid(items, durations, pace):
    st.html('<div class="ctitle" style="margin-top:.4rem"><span>Nouveautés</span><span class="pill soft">8 articles · photos réelles</span></div>')
    for row in range(0, len(items), 4):
        cols = st.columns(4, gap="small")
        for col, it in zip(cols, items[row:row + 4]):
            with col:
                _product_card(it, durations, pace)


def _detail(items, it, durations, pace):
    if st.button("← Retour à la boutique", key="shop_back"):
        st.session_state.shop_view = "grid"
        st.rerun()
    a, b = st.columns([1.05, 1], gap="large")
    with a:
        st.html(img_tag(it["big"], it["title"], radius=18))
    with b:
        st.html(f'<div class="eyebrow">{it["title"]} · nouvelle collection</div>'
                f'<h2 style="margin:.2rem 0 .1rem;font-size:1.9rem">{it["title"]}</h2>'
                f'<div style="font-family:Poppins;font-weight:700;font-size:1.6rem;color:#4B2FD1;margin:.2rem 0 .6rem">{it["price"]:.0f} DH</div>'
                f'<p style="color:#5A587A;margin:0 0 .6rem">{BLURB.get(it["title"], "")}</p>')
        sizes = SIZES.get(it["title"], DEFAULT_SIZES)
        size = st.pills("Taille", sizes, key=f"size_{it['i']}")
        if size and st.session_state.get(f"size_logged_{it['i']}") != size:
            st.session_state[f"size_logged_{it['i']}"] = size
            _add(2, it, durations, pace, label=f"Choisir la taille {size} · {it['title']}")
        q1, q2 = st.columns([1, 2], vertical_alignment="bottom")
        qty = q1.number_input("Quantité", 1, 5, 1, key=f"qty_{it['i']}")
        if q2.button("Ajouter au panier", key=f"padd_{it['i']}", type="primary", use_container_width=True):
            for _ in range(int(qty)):
                st.session_state.shop_cart.append(it["i"])
            _add(3, it, durations, pace, label=f"Ajouter au panier ×{int(qty)} · {it['title']}")
            st.toast(f"{it['title']} ajouté au panier")
            st.rerun()
        if st.button("Voir la fiche détaillée et le guide des tailles", key=f"more_{it['i']}", use_container_width=True):
            _add(2, it, durations, pace)
            st.session_state[f"more_open_{it['i']}"] = True
        if st.session_state.get(f"more_open_{it['i']}"):
            st.html('<div style="font-size:.88rem;color:#5A587A;line-height:1.6">Photo réelle (domaine public). '
                    'Tailles standard ; en cas de doute, prendre la taille au-dessus. Livraison partout au Maroc · retour sous 7 jours '
                    '(boutique de démonstration).</div>')
        with st.container(key="card_model_view"):
            st.html('<div class="ctitle"><span>Ce que voient les modèles</span><span class="pill soft">IA</span></div>')
            m1, m2 = st.columns([1, 2.2], vertical_alignment="center")
            m1.html(img_tag(gray_png(it["x28"], 96), "28×28", width=84, radius=10))
            m2.html(f'<div style="font-size:.86rem;line-height:1.55">Catégorie prédite : <b>{it["pred"]}</b> '
                    f'({it["conf"]*100:.0f} %, fusion J3+J4)<br>Prix estimé : <b>{it["raw_price"]:.0f} DH</b> (MLP J1)<br>'
                    f'Gamme de prix envoyée au modèle de session : {it["range"]}/5</div>')
    others = [x for x in items if x["i"] != it["i"]]
    same = [x for x in others if x["cat"] == it["cat"]]
    recs = (same + [x for x in others if x not in same])[:3]
    st.html('<div class="ctitle" style="margin-top:1rem"><span>Vous pourriez aussi aimer</span></div>')
    cols = st.columns(3, gap="small")
    for col, x in zip(cols, recs):
        with col:
            _product_card(x, durations, pace, prefix=f"r{it['i']}")


def _cart(items, durations, pace):
    if st.button("← Continuer mes achats", key="shop_back_cart"):
        st.session_state.shop_view = "grid"
        st.rerun()
    cart = st.session_state.shop_cart
    st.html(f'<h3 style="margin:.3rem 0 .6rem">Mon panier · {len(cart)} article(s)</h3>')
    if not cart:
        st.caption("Votre panier est vide.")
        return
    total = 0.0
    for pos, i in enumerate(list(cart)):
        it = next(x for x in items if x["i"] == i)
        total += it["price"]
        c1, c2, c3 = st.columns([.6, 2, 1], vertical_alignment="center")
        c1.html(img_tag(it["png"], it["title"], width=64, radius=10))
        c2.html(f'<b>{it["title"]}</b><br><span style="color:#4B2FD1;font-weight:600">{it["price"]:.0f} DH</span>')
        if c3.button("Retirer", key=f"rm_{pos}_{i}", use_container_width=True):
            cart.pop(pos)
            _add(4, it, durations, pace)
            st.rerun()
    st.html(f'<div style="display:flex;justify-content:space-between;font-family:Poppins;font-weight:600;font-size:1.1rem;'
            f'border-top:1px solid #E5E2F7;padding-top:.6rem;margin-top:.4rem"><span>Total</span><span>{total:.0f} DH</span></div>')
    if st.button("Passer commande", key="shop_checkout", type="primary", use_container_width=True):
        st.session_state.shop_done = True
    if st.session_state.get("shop_done"):
        st.success("Commande simulée. Comparez avec la probabilité affichée à droite juste avant la commande.")


def render():
    _state()
    items, durations = _shop()
    threshold = core.coupon_threshold()
    hero("Boutique en direct",
         "Naviguez comme sur un vrai site : chaque clic devient un événement de session et le modèle recalcule la probabilité d’achat.",
         icon="cart", eyebrow="Simulation temps réel")

    tab_manual, tab_auto = st.tabs(["Je navigue moi-même", "Client automatique (replay)"])

    with tab_manual:
        c1, c2, c3 = st.columns([1.4, 1, 1], vertical_alignment="bottom")
        key = c1.selectbox("Modèle de prédiction", list(core.PURCHASE_MODELS), key="shop_model",
                           format_func=lambda k: core.PURCHASE_MODELS[k]["label"])
        pace = c2.segmented_control("Temps passé par action", list(PACE), default="Normal", key="shop_pace") or "Normal"
        if c3.button("Nouvelle session", key="shop_reset", use_container_width=True):
            for k in [k for k in st.session_state if str(k).startswith(("size_", "qty_", "more_open_"))]:
                del st.session_state[k]
            st.session_state.shop_events, st.session_state.shop_cart = [], []
            st.session_state.shop_view, st.session_state.shop_done = "grid", False
            st.rerun()

        left, right = st.columns([2.1, 1], gap="medium")
        with left, st.container(key="glass_shop"):
            _shop_header(items, durations, pace)
            view = st.session_state.shop_view
            if view == "cart":
                _cart(items, durations, pace)
            elif view == "grid":
                _grid(items, durations, pace)
            else:
                _detail(items, next(x for x in items if x["i"] == view), durations, pace)
        events = st.session_state.shop_events
        history = _history(events, key)
        with right:
            _score_panel(history, threshold, [e["label"] for e in events], "manual")
            with st.container(key="card_stats"):
                k1, k2, k3 = st.columns(3)
                k1.html(kpi(str(len(events)), "événements"))
                k2.html(kpi(str(len(st.session_state.shop_cart)), "au panier"))
                k3.html(kpi(f"{max(history)*100:.0f}%" if len(history) else "—", "pic"))
            with st.container(key="card_log"):
                st.html('<div class="ctitle"><span>Journal de session</span></div>')
                if events:
                    st.html(_event_log(events, history))
                else:
                    st.caption("Aucun événement pour l’instant.")
                if len(events) > core.MAX_EVENTS:
                    st.caption(f"Plus de {core.MAX_EVENTS} événements : seuls les derniers sont lus par le modèle.")
        with st.expander("Comment les clics deviennent des événements"):
            st.markdown(
                "Les jeux UC2 fournissent des codes d’événement 1–5 **sans dictionnaire officiel**. Pour la démo, l’interface leur associe : "
                "ouvrir une fiche produit = 1, choisir une taille ou ouvrir la fiche détaillée = 2, ajouter au panier = 3, "
                "retirer du panier = 4, rechercher = 5. Dans la validation, le code 3 en dernier événement précède un achat dans "
                "13 sessions sur 14, d’où ce choix pour le panier.\n\n"
                "Les produits sont des photos réelles (domaine public) ; leur prix est celui estimé par le MLP log-Huber (J1) à partir de "
                "la photo convertie et du titre. Catégorie de l’événement = classe du produit + 1 ; gamme de prix = quintile du prix dans "
                "le catalogue A ; durée = quantile 20 / 50 / 80 % des durées réelles selon le rythme choisi.")

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
               "label": f"Événement type {int(d['type_val'][sid, t])} · catégorie {int(d['categorie_val'][sid, t])}"} for t in range(length)]
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
            st.html(f'<div class="ctitle"><span>Parcours rejoué</span><span class="pill">{step}/{length} événements</span></div>')
            st.progress(step / length)
            if step:
                s0 = max(0, step - 12)
                st.html(_event_log(events[s0:step], full, start=s0))
            else:
                st.caption("Appuyez sur Lecture : les événements réels de la session arrivent un par un.")
            if step == length:
                st.caption("Fin de session : l’étiquette réelle de validation est affichée à droite.")
    live()
