"""Smoke tests des pages avec les vrais modèles, sans entraînement des jalons."""
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 180


def page(name):
    script = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
              "from demo.theme import apply_theme; apply_theme()\n"
              f"from demo.views import {name}; {name}.render()\n")
    return AppTest.from_string(script, default_timeout=TIMEOUT).run()


def check(a, label):
    assert not a.exception, [x.message for x in a.exception]
    print("PASS", label, flush=True)


def button(a, key):
    return next(b for b in a.button if b.key == key)


# Entrée complète + accueil
a = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=TIMEOUT).run()
check(a, "accueil")

# Boutique en direct
a = page("live_shop")
check(a, "boutique vide")
for k in ["open_g_0", "padd_0", "more_0", "shop_back", "add_10", "shop_cart_open", "rm_0_0", "shop_back_cart", "shop_search"]:
    button(a, k).click().run()
    check(a, f"boutique clic {k}")
a.selectbox(key="shop_model").select("mlp").run()
check(a, "boutique modèle MLP")
button(a, "replay_toggle").click().run()
check(a, "replay lecture")

# Sessions : exemple, comparaison, fichier invalide
a = page("sessions")
check(a, "sessions exemple GRU")
a.toggle(key="sess_compare").set_value(True).run()
check(a, "sessions comparaison")
a.selectbox(key="sess_model").select("mlp").run()
check(a, "sessions MLP")

# Produit : exemple et chaque modèle, texte vide
a = page("product")
check(a, "produit exemple")
for m in ["mlp_a", "mt_b", "bow", "cnn_a", "fusion"]:
    a.selectbox(key="prod_model").select(m).run()
    check(a, f"produit modèle {m}")
a.selectbox(key="prod_price_model").select("mt_b").run()
check(a, "produit prix multitâche")
a.slider(key="prod_idx").set_value(69).run()
check(a, "produit dernière photo réelle")

# Catalogue : exemple
a = page("catalogue")
check(a, "catalogue exemple")

# Jalons d'origine
a = page("jalons")
check(a, "jalon 1")
a.slider(key="j1_idx").set_value(199).run()
check(a, "jalon 1 dernier exemple")
for v in ["2 · CNN et auto-encodeur", "3 · Boutique B et multitâche", "4 · Sessions GRU"]:
    a = page("jalons")
    a.session_state["jalon_view"] = v
    a.run()
    check(a, v)
a = page("jalons")
a.session_state["jalon_view"] = "2 · CNN et auto-encodeur"
a.run()
for probe in ["Rectangle synthétique de démonstration", "Bruit synthétique de démonstration"]:
    a.selectbox(key="j2_probe").select(probe).run()
    check(a, probe)

# Fonctions cœur : fichiers utilisateurs
from demo import core  # noqa: E402

raw = pd.read_csv(BytesIO(core.sample_sessions_csv()))
df, w = core.validate_sessions(raw.rename(columns={"event_type": "type", "duration": "duree"}))
ids, arr = core.sessions_to_arrays(df)
p = core.predict_purchase(arr, "gru")
assert len(p) == 200 and np.all((p >= 0) & (p <= 1))
m = core.ranking_metrics(df.groupby("session_id", sort=False)["achat"].first().to_numpy(), p)
assert m["AUC"] > 0.6, m
print("PASS sessions CSV : AUC exemple", round(m["AUC"], 3))
long = pd.DataFrame({"session_id": 1, "event_type": 1, "category": 2, "price_range": 3, "duration": 3.0}, index=range(75))
df, w = core.validate_sessions(long)
assert w and core.sessions_to_arrays(df)[1]["longueur"][0] == 60
print("PASS session > 60 événements tronquée")
try:
    core.validate_sessions(pd.DataFrame({"a": [1]}))
    raise AssertionError("colonnes manquantes non détectées")
except ValueError:
    print("PASS colonnes manquantes")

csv, zipb = core.sample_catalogue()
cat = pd.read_csv(BytesIO(csv))
imgs = core.read_zip_images(zipb)
res = core.score_products(cat, "cnn_a", "huber_a", imgs, core.DEFAULT_WEIGHTS)
assert len(res) == 70 and res["potentiel"].between(0, 1).all()
res2 = core.score_products(cat.drop(columns=["image"]), "mlp_a", "mt_b", {}, core.DEFAULT_WEIGHTS)
assert (res2["source estimation"] == "médiane de la catégorie").all()
print("PASS catalogue avec et sans images")

from PIL import Image  # noqa: E402
photo = Image.new("RGB", (300, 400), "white")
photo.paste((40, 40, 40), (80, 60, 220, 340))
x = core.to_fashion_mnist(photo)
assert x.shape == (28, 28) and x.max() > 150 and x[0, 0] < 30
print("PASS conversion photo synthétique → 28×28")
r = core.real_photos()
assert np.array_equal(core.to_fashion_mnist(core.real_photo(str(r["name"][0]))), r["X"][0])
print("PASS photos réelles : conversion embarquée identique à l’import")

# Entraînement sur étiquettes : cible synthétique de test uniquement
from demo.views.catalogue import _binary, _features  # noqa: E402
X, names = _features(res)
assert X.shape[0] == 70 and X.shape[1] == len(names)
assert _binary(pd.Series(["oui", "non", "oui"])).tolist() == [1, 0, 1]
print("PASS caractéristiques d’entraînement")
print("Toutes les vues passent sans exception.")
