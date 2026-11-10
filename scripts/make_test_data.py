"""Génère les fichiers de test manuel des pages d'import (dossier test_data/).

- sessions_test.csv : 600 parcours clients SIMULÉS (une ligne par événement) avec l'étiquette `achat`.
- sessions_test_erreurs.csv : petit fichier piégé (alias français, codes invalides, valeurs manquantes, session > 60 événements).
- catalogue_test.csv + catalogue_test_images.zip : 245 annonces avec de VRAIES photos (clothing-dataset-small, partition
  train, CC0 ; les exemples embarqués viennent de la partition test), titres et descriptions de style marketplace, prix en DH et étiquette `vendu` SIMULÉE.
- catalogue_test_sans_images.csv : les mêmes annonces sans colonne image (chemin texte seul).

Usage : python scripts/make_test_data.py /chemin/vers/clothing-dataset-small
"""
from pathlib import Path
import io
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_data"
rng = np.random.default_rng(2026)

# ------------------------------------------------------------------ sessions
def simulate_session(sid: int, buyer: bool) -> list[dict]:
    n = int(np.clip(rng.poisson(21 if buyer else 14) + 4, 4, 60))
    interests = rng.choice(np.arange(1, 11), size=rng.integers(1, 4), replace=False)
    pref_price = int(rng.integers(1, 6))
    types = rng.choice([1, 2, 5], size=n, p=[.82, .10, .08])
    if buyer:
        if rng.random() < .85:  # 15 % achètent directement, sans événement panier visible
            carts = rng.integers(1, 3)
            pos = rng.choice(np.arange(n // 3, n), size=min(carts, n - n // 3), replace=False)
            types[pos] = 3
            if rng.random() < .35:
                types[-1] = 3
    else:
        if rng.random() < .5:  # paniers abandonnés, fréquents en e-commerce
            for p in rng.choice(np.arange(1, n), size=min(int(rng.integers(1, 3)), n - 1), replace=False):
                types[p] = 3
                if rng.random() < .3 and p + 1 < n:
                    types[p + 1] = 4
    rows = []
    t0 = pd.Timestamp("2026-10-01") + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 30)))
    for k in range(n):
        cat = int(rng.choice(interests)) if rng.random() < .8 else int(rng.integers(1, 11))
        price = int(np.clip(pref_price + rng.integers(-1, 2), 1, 5))
        dur = float(np.clip(rng.normal(3.0 + (.25 if types[k] in (2, 3) else 0), .45), 1.5, 4.5))
        t0 += pd.Timedelta(seconds=float(np.exp(dur)))
        rows.append({"session_id": f"S-2026-{sid:05d}", "step": k + 1, "event_time": t0.isoformat(timespec="seconds"),
                     "event_type": int(types[k]), "category": cat, "price_range": price, "duration": round(dur, 3),
                     "achat": int(buyer)})
    return rows


def sessions():
    rows = []
    for sid in range(1, 601):
        rows += simulate_session(sid, rng.random() < .30)
    pd.DataFrame(rows).to_csv(OUT / "sessions_test.csv", index=False)

    bad = []
    for sid in range(1, 31):
        bad += simulate_session(9000 + sid, rng.random() < .3)
    df = pd.DataFrame(bad).drop(columns=["event_time"])
    df = df.rename(columns={"session_id": "session", "event_type": "type", "category": "categorie",
                            "price_range": "gamme_prix", "duration": "duree", "achat": "purchase"})
    df.loc[3, "type"] = 7          # code d'événement inconnu
    df.loc[8, "categorie"] = 0     # catégorie hors 1–10
    df.loc[12, "duree"] = np.nan   # durée manquante
    long = pd.DataFrame([{"session": "S-2026-LONG", "step": k + 1, "type": 1 if k % 9 else 3, "categorie": 4,
                          "gamme_prix": 3, "duree": 3.1, "purchase": 1} for k in range(75)])
    df = pd.concat([df, long], ignore_index=True).sample(frac=1, random_state=1)  # lignes mélangées : tri par step
    df.to_csv(OUT / "sessions_test_erreurs.csv", index=False)


# ------------------------------------------------------------------ catalogue
CLASSES = {
    "t-shirt": (["T-shirt coton", "Tee shirt col rond", "T-shirt over size", "Top manches courtes"], 120),
    "pants": (["Pantalon chino", "Jean slim", "Pantalon coupe droite", "Jogging sport"], 220),
    "longsleeve": (["Pull manches longues", "Sweat hoodie", "Pull col roulé", "Pull en laine"], 200),
    "dress": (["Robe midi", "Robe longue", "Robe de soirée", "Robe légère été"], 280),
    "outwear": (["Veste zip", "Blouson", "Doudoune hiver", "Parka imperméable"], 450),
    "shirt": (["Chemise manches courtes", "Chemise en lin", "Surchemise", "Chemisier"], 180),
    "shoes": (["Basket running", "Sneaker blanc", "Bottine cuir", "Sandale"], 350),
}
STATES = [("neuf avec étiquette", 1.0, .8), ("très bon état", .75, .3), ("porté quelques fois", .65, .1), ("occasion", .55, -.2)]
MATERIALS = {"shoes": ["cuir", "daim", "toile", "synthétique"], "outwear": ["polyester", "laine", "cuir", "coton"],
             "longsleeve": ["laine", "coton", "tricot"], "default": ["coton", "lin", "polyester", "viscose"]}
COLORS = ["noir", "blanc", "bleu", "marine", "gris", "beige", "camel", "rouge", "rose", "vert", "bordeaux"]
EXTRAS = ["livraison partout au maroc", "paiement à la livraison", "prix négociable", "photos réelles", "échange possible",
          "stock limité", "qualité premium", "urgent", "original"]


def catalogue(src: Path):
    rows, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for cls, (titles, ref) in CLASSES.items():
            files = sorted((src / "train" / cls).glob("*.jpg"))
            for f in rng.choice(files, size=min(35, len(files)), replace=False):
                n += 1
                img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                img.thumbnail((640, 640))
                name = f"produit_{n:03d}.jpg"
                b = io.BytesIO()
                img.save(b, format="JPEG", quality=85)
                z.writestr(name, b.getvalue())
                title = str(rng.choice(titles))
                state, mult, bonus = STATES[int(rng.choice(4, p=[.35, .3, .2, .15]))]
                k_extra = int(rng.integers(0, 4))
                extras = list(rng.choice(EXTRAS, size=k_extra, replace=False))
                size = f"pointure {rng.integers(38, 45)}" if cls == "shoes" else f"taille {rng.choice(['S', 'M', 'L', 'XL'])}"
                parts = [state, str(rng.choice(MATERIALS.get(cls, MATERIALS["default"]))), str(rng.choice(COLORS)), size] + extras
                desc = " ".join(parts) if rng.random() > .08 else ""
                fair = ref * mult
                price = float(np.round(fair * np.exp(rng.normal(0, .35)) / 5) * 5)
                # étiquette SIMULÉE : vend mieux si moins cher que le prix juste, neuf, annonce riche et livraison proposée
                logit = (-2.2 * np.log(price / fair) + bonus + .35 * k_extra
                         + (.6 if "livraison partout au maroc" in extras else 0) - .9 + rng.normal(0, .6))
                rows.append({"product_id": f"P{n:04d}", "title": title, "description": desc, "price": price,
                             "image": name, "vendu": int(rng.random() < 1 / (1 + np.exp(-logit)))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "catalogue_test.csv", index=False)
    df.drop(columns=["image"]).to_csv(OUT / "catalogue_test_sans_images.csv", index=False)
    (OUT / "catalogue_test_images.zip").write_bytes(buf.getvalue())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sessions()
    catalogue(Path(sys.argv[1]))
    print("Fichiers écrits dans", OUT)
