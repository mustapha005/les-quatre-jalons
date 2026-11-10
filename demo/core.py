"""Chargement des artefacts figés et fonctions de prédiction partagées par toutes les pages.

Aucun entraînement des modèles des jalons n'a lieu ici : les poids `.keras` sont lus tels quels.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import CLASSES, SmartphoneBrightness, keras, session_features  # noqa: E402,F401
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

MAX_EVENTS = 60
COUPON_FRACTION = 0.10

# ---------------------------------------------------------------- artefacts
@st.cache_data(show_spinner=False)
def data(name: str) -> dict:
    with np.load(ROOT / "data_UC2" / name, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k.endswith("_val")}


@st.cache_resource(show_spinner="Chargement du modèle…")
def model(j: int, name: str):
    return keras.models.load_model(ROOT / f"artifacts/jalon{j}/models/{name}/best.keras", compile=False)


@st.cache_resource(show_spinner=False)
def saved(j: int, name: str):
    return joblib.load(ROOT / f"artifacts/jalon{j}" / name)


@st.cache_data(show_spinner=False)
def artifact(j: int, name: str):
    return json.loads((ROOT / f"artifacts/jalon{j}" / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def price_upper(catalogue: str) -> float:
    return float(json.loads((ROOT / "data_UC2/metadata.json").read_text())[catalogue]["price_clip_upper"])


@st.cache_resource(show_spinner=False)
def vectorizer(j: int, filename: str = "vocabulary.json") -> CountVectorizer:
    return CountVectorizer(binary=True, vocabulary=artifact(j, filename))


def bow(j: int, texts, filename: str = "vocabulary.json") -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    return vectorizer(j, filename).transform(list(texts)).toarray().astype("float32")


def infer(j: int, name: str, x, batch: int = 512):
    """Inférence par lots ; renvoie un ndarray ou un dict de ndarray."""
    m = model(j, name)
    n = len(next(iter(x.values()))) if isinstance(x, dict) else len(x)
    outs = []
    for s in range(0, n, batch):
        part = {k: v[s:s + batch] for k, v in x.items()} if isinstance(x, dict) else x[s:s + batch]
        value = m(part, training=False)
        outs.append({k: v.numpy() for k, v in value.items()} if isinstance(value, dict) else value.numpy())
    if isinstance(outs[0], dict):
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    return np.concatenate(outs)


# ---------------------------------------------------------------- achats (sessions)
PURCHASE_MODELS = {
    "gru": {"label": "GRU séquentiel · jalon 4", "short": "GRU (J4)", "j": 4, "name": "GRU_unmasked",
            "info": "Lit l’ordre des 60 derniers événements. Modèle retenu au jalon 4 (AUC test 0,836 ; P@10 % 0,798)."},
    "mlp": {"label": "MLP de caractéristiques · jalon 1", "short": "MLP riche (J1)", "j": 1, "name": "session_rich_mlp",
            "info": "Comptages, durées, prix et cinq derniers événements. Jalon 1 (AUC test 0,834 ; P@10 % 0,778)."},
}

EVENT_TYPES = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5"}

SESSION_ALIASES = {
    "session_id": ["session_id", "session", "id_session", "user_session", "visitor_id", "user_id"],
    "event_type": ["event_type", "type", "type_evenement", "action"],
    "category": ["category", "categorie", "catégorie", "category_id"],
    "price_range": ["price_range", "gamme_prix", "prix", "price_bucket", "gamme"],
    "duration": ["duration", "duree", "durée", "dwell", "dwell_time"],
    "achat": ["achat", "purchase", "purchased", "label", "converted", "y"],
    "step": ["step", "etape", "ordre", "position", "timestamp", "event_time"],
}


def normalise_columns(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for target, names in aliases.items():
        for n in names:
            if n in lower:
                rename[lower[n]] = target
                break
    return df.rename(columns=rename)


def validate_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Nettoie un fichier long (une ligne = un événement). Renvoie (df, avertissements)."""
    df = normalise_columns(df, SESSION_ALIASES)
    required = ["session_id", "event_type", "category", "price_range", "duration"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(missing))
    warnings = []
    df = df.copy()
    for c in ["event_type", "category", "price_range"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    bad = df[required].isna().any(axis=1) | ~df["event_type"].between(1, 5) | ~df["category"].between(1, 10) \
        | ~df["price_range"].between(1, 5)
    if bad.any():
        warnings.append(f"{int(bad.sum())} événement(s) hors format ignoré(s) (type 1–5, catégorie 1–10, gamme 1–5).")
        df = df[~bad]
    if df.empty:
        raise ValueError("Aucun événement valide après contrôle du format.")
    if "step" in df.columns:
        df = df.sort_values(["session_id", "step"], kind="stable")
    long = df.groupby("session_id", sort=False).size()
    if (long > MAX_EVENTS).any():
        warnings.append(f"{int((long > MAX_EVENTS).sum())} session(s) de plus de {MAX_EVENTS} événements : "
                        f"seuls les {MAX_EVENTS} derniers sont gardés, comme le padding fixe des modèles.")
    return df, warnings


def sessions_to_arrays(df: pd.DataFrame) -> tuple[list, dict]:
    """Événements → tableaux (n, 60) au format des jalons, padding à droite."""
    ids, groups = [], []
    for sid, g in df.groupby("session_id", sort=False):
        ids.append(sid)
        groups.append(g.tail(MAX_EVENTS))
    n = len(groups)
    arr = {k: np.zeros((n, MAX_EVENTS), dtype=dt) for k, dt in
           [("type", "int32"), ("categorie", "int32"), ("prix", "float32"), ("duree", "float32")]}
    length = np.zeros(n, dtype="int32")
    for i, g in enumerate(groups):
        L = len(g)
        length[i] = L
        arr["type"][i, :L] = g["event_type"].to_numpy()
        arr["categorie"][i, :L] = g["category"].to_numpy()
        arr["prix"][i, :L] = g["price_range"].to_numpy()
        arr["duree"][i, :L] = g["duration"].to_numpy()
    arr["longueur"] = length
    return ids, arr


def events_to_arrays(events: list[dict]) -> dict:
    df = pd.DataFrame(events or [], columns=["event_type", "category", "price_range", "duration"])
    df["session_id"] = 0
    _, arr = sessions_to_arrays(df)
    return arr


def predict_purchase(arr: dict, key: str) -> np.ndarray:
    spec = PURCHASE_MODELS[key]
    if key == "gru":
        types = arr["type"].astype("int32")
        numeric = np.stack([arr["prix"] / 5.0, arr["duree"]], axis=-1).astype("float32")
        numeric = saved(4, "event_numeric_scaler.joblib").transform(numeric.reshape(-1, 2)).reshape(numeric.shape).astype("float32")
        numeric[types == 0] = 0.0
        fields = {"type": types, "categorie": arr["categorie"].astype("int32"), "numerique": numeric}
        return infer(spec["j"], spec["name"], fields).ravel()
    split = {f"{k}_x": v for k, v in arr.items()}
    split["type_x"] = arr["type"].astype(int)
    feats = session_features(split, "x", rich=True)
    feats = saved(1, "session_rich_mlp_scaler.joblib").transform(feats).astype("float32")
    return infer(spec["j"], spec["name"], feats).ravel()


def coupon_threshold() -> float:
    return float(artifact(4, "frozen_protocol.json")["selection"]["coupon_threshold_val"])


@st.cache_data(show_spinner=False)
def sample_sessions_csv() -> bytes:
    """Les 200 sessions de validation incluses, au format long attendu."""
    d = data("uc2_sessions.npz")
    rows = []
    for i in range(len(d["achat_val"])):
        for t in range(int(d["longueur_val"][i])):
            rows.append({"session_id": f"S{i:03d}", "step": t + 1, "event_type": int(d["type_val"][i, t]),
                         "category": int(d["categorie_val"][i, t]), "price_range": int(d["prix_val"][i, t]),
                         "duration": round(float(d["duree_val"][i, t]), 3), "achat": int(d["achat_val"][i])})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    k = max(1, int(np.ceil(COUPON_FRACTION * len(p))))
    top = np.argsort(-p, kind="stable")[:k]
    out = {"P@10 %": float(y[top].mean()), "Taux d’achat": float(y.mean())}
    if 0 < y.sum() < len(y):
        out["AUC"] = float(roc_auc_score(y, p))
        out["Précision moyenne"] = float(average_precision_score(y, p))
    return out


# ---------------------------------------------------------------- produits
PRODUCT_MODELS = {
    "fusion": {"label": "Fusion photo + texte · J3 + J4 (recommandé)", "short": "Fusion J3+J4", "j": 3, "name": "MT_warm_lambda_1p0",
               "vocab": "vocabulary.json", "image": "4d", "acc": "moyenne des probabilités du multitâche et du BoW",
               "real": "photos réelles : 81 % avec un titre simple, 51 % sans titre reconnu"},
    "cnn_a": {"label": "CNN image + texte · jalon 2 (boutique A)", "short": "CNN (J2)", "j": 2, "name": "CNN_base",
              "vocab": "vocabulary.json", "image": "4d", "acc": "95,2 %", "real": "photos réelles : 40 % image seule, 65 % avec un titre simple"},
    "mlp_a": {"label": "MLP image + texte · jalon 1 (boutique A)", "short": "MLP (J1)", "j": 1, "name": "MLP_reference",
              "vocab": "vocabulary.json", "image": "flat", "acc": "93,1 %", "real": "photos réelles : 36 % image seule, 59 % avec un titre simple"},
    "mt_b": {"label": "Multitâche adapté · jalon 3 (boutique B)", "short": "Multitâche (J3)", "j": 3, "name": "MT_warm_lambda_1p0",
             "vocab": "vocabulary.json", "image": "4d", "acc": "80,4 % sur B", "real": "photos réelles : 51 % image seule, 67 % avec un titre simple"},
    "bow": {"label": "Texte seul (BoW + MLP) · jalon 4", "short": "Texte BoW (J4)", "j": 4, "name": "BoW_MLP",
            "vocab": "bow_vocabulary.json", "image": None, "acc": "74,0 %", "real": "photos réelles : 77 % sur le titre seul"},
}

PRICE_MODELS = {
    "huber_a": "MLP prix log-Huber · jalon 1 (boutique A)",
    "mt_b": "Tête prix du multitâche · jalon 3 (boutique B)",
}


def text_probs(texts) -> np.ndarray:
    return infer(4, "BoW_MLP", bow(4, texts, "bow_vocabulary.json"))


def category_probs(key: str, images: np.ndarray | None, texts) -> np.ndarray:
    """images : (n,28,28) dans [0,1] ou None pour un modèle texte seul."""
    spec = PRODUCT_MODELS[key]
    if spec["image"] is None or images is None:
        return text_probs(texts)
    if key == "fusion":  # fusion tardive : moyenne image+texte (J3) et texte seul (J4) quand le titre contient des mots connus
        texts = list(texts)
        p = category_probs("mt_b", images, texts)
        has_words = known_words(texts) > 0
        if has_words.any():
            p[has_words] = (p[has_words] + text_probs([t for t, h in zip(texts, has_words) if h])) / 2
        return p
    words = bow(spec["j"], texts, spec["vocab"])
    x = images.reshape(len(images), 784) if spec["image"] == "flat" else images[..., None]
    out = infer(spec["j"], spec["name"], {"image": x.astype("float32"), "mots": words})
    return out["categorie"] if isinstance(out, dict) else out


def price_estimates(key: str, images: np.ndarray, texts) -> np.ndarray:
    if key == "mt_b":
        sc = artifact(3, "log_price_scaler.json")
        out = infer(3, "MT_warm_lambda_1p0", {"image": images[..., None].astype("float32"), "mots": bow(3, texts)})
        lp = out["logprix"].ravel() * sc["std"] + sc["mean"]
        return np.exp(np.clip(lp, 0, np.log(price_upper("uc2_catalogueB.npz"))))
    lp = infer(1, "price_log_huber", {"image": images.reshape(len(images), 784).astype("float32"), "mots": bow(1, texts)}).ravel()
    return np.exp(np.clip(lp, 0, np.log(price_upper("uc2_catalogueA.npz"))))


@st.cache_data(show_spinner=False)
def category_median_price() -> np.ndarray:
    d = data("uc2_catalogueA.npz")
    return np.array([np.median(d["prix_val"][d["categorie_val"] == k]) for k in range(10)], dtype="float32")


def ae_threshold() -> float:
    return float(artifact(2, "frozen_protocol.json")["selection"]["threshold"])


def reconstruction(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = images[..., None].astype("float32")
    recon = infer(2, "AE_128", x)
    return recon[..., 0], np.mean((x - recon) ** 2, axis=(1, 2, 3))


SEGMENTER = ROOT / "artifacts/segmentation/u2netp.onnx"


@st.cache_resource(show_spinner=False)
def _segmenter():
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not SEGMENTER.is_file():
        return None
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(SEGMENTER), opts, providers=["CPUExecutionProvider"])


def garment_mask(img: Image.Image) -> np.ndarray | None:
    """Masque [0,1] de l'objet principal (U²-Net-p, détourage), à la taille de l'image ; None si indisponible."""
    sess = _segmenter()
    if sess is None:
        return None
    rgb = img.convert("RGB")
    a = np.asarray(rgb.resize((320, 320), Image.Resampling.LANCZOS), dtype="float32")
    a = a / max(float(a.max()), 1.0)
    a = (a - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    pred = sess.run(None, {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype("float32")})[0][0, 0]
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    m = Image.fromarray((pred * 255).astype(np.uint8)).resize(rgb.size, Image.Resampling.LANCZOS)
    return np.asarray(m, dtype="float32") / 255.0


def to_fashion_mnist(img: Image.Image) -> np.ndarray:
    """Photo réelle → 28×28 au format Fashion-MNIST.

    Reprend la chaîne de conversion publiée de Fashion-MNIST (photos Zalando) : détourage du vêtement
    (fond remplacé par du blanc), niveaux de gris, rognage, plus grand côté à 28 px, centrage, inversion.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((640, 640))
    mask = garment_mask(img)
    if mask is None:
        return _to_fashion_mnist_heuristic(img)
    g = np.asarray(img.convert("L"), dtype="float32")
    obj = mask > 0.5
    if obj.sum() < 50:
        return _to_fashion_mnist_heuristic(img)
    ys, xs = np.where(obj)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ink = (255.0 - g) * mask  # vêtement sur fond blanc, puis inversion : fond noir, vêtement clair
    ink = ink[y0:y1, x0:x1]
    ink = np.maximum(ink, 40.0 * mask[y0:y1, x0:x1])  # un vêtement blanc reste visible, comme la silhouette Fashion-MNIST
    h, w = ink.shape
    scale = 28.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray(np.clip(ink, 0, 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - nw) // 2, (28 - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def pil_jpeg(img: Image.Image, size: int = 240) -> bytes:
    out = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def cutout_png(img: Image.Image, size: int = 320, bg: str = "#EEF1EC") -> bytes | None:
    """Vêtement détouré sur fond uni (aperçu de ce que voit la segmentation)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size, size))
    mask = garment_mask(img)
    if mask is None:
        return None
    back = Image.new("RGB", img.size, bg)
    out = Image.composite(img, back, Image.fromarray((mask * 255).astype(np.uint8)))
    out = ImageOps.pad(out, (size, size), color=bg)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _to_fashion_mnist_heuristic(img: Image.Image) -> np.ndarray:
    """Repli sans segmentation : seuil sur la couleur du bord."""
    g = img.convert("L")
    g.thumbnail((512, 512))
    a = np.asarray(g, dtype="float32")
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if np.median(border) > 127:  # fond clair : on inverse comme Fashion-MNIST
        a = 255 - a
        border = 255 - border
    bg = np.median(border)
    mask = a > bg + 25
    if mask.sum() > 20:
        ys, xs = np.where(mask)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = np.clip(a - bg, 0, None)
    h, w = a.shape
    side = int(max(h, w) * 1.08) + 1
    canvas = np.zeros((side, side), dtype="float32")
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = a
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS)
    o = np.asarray(out, dtype="float32")
    if o.max() > 0:
        o = o / o.max() * 255
    return o.astype(np.uint8)


def read_zip_images(raw: bytes) -> dict[str, np.ndarray]:
    found = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX") or not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                continue
            try:
                with z.open(info) as f:
                    found[Path(name).name.lower()] = to_fashion_mnist(Image.open(io.BytesIO(f.read())))
            except Exception:
                continue
    return found


# ---------------------------------------------------------------- indice de potentiel
DEFAULT_WEIGHTS = {"prix": 0.30, "confiance": 0.20, "coherence": 0.20, "annonce": 0.15, "image": 0.15}
FACTOR_LABELS = {
    "prix": ("Prix attractif", "Prix saisi comparé au prix estimé par le modèle : en dessous = attractif."),
    "confiance": ("Catégorie claire", "Confiance du modèle choisi dans la catégorie prédite."),
    "coherence": ("Texte et image cohérents", "Le modèle texte seul (J4) confirme-t-il la catégorie ?"),
    "annonce": ("Annonce riche", "Nombre de mots de la description reconnus par le vocabulaire appris."),
    "image": ("Image conforme", "Erreur de reconstruction de l’auto-encodeur (J2) comparée au seuil figé."),
}


def known_words(texts) -> np.ndarray:
    return bow(4, texts, "bow_vocabulary.json").sum(axis=1)


def potential_factors(conf, coherence, price_ratio, words, mse=None) -> dict[str, np.ndarray]:
    f = {
        "prix": 1.0 / (1.0 + np.exp(4.0 * (np.asarray(price_ratio, dtype="float64") - 1.0))),
        "confiance": np.asarray(conf, dtype="float64"),
        "coherence": np.asarray(coherence, dtype="float64"),
        "annonce": np.clip(np.asarray(words, dtype="float64") / 6.0, 0, 1),
    }
    if mse is not None:
        f["image"] = np.clip(1.0 - 0.5 * np.asarray(mse, dtype="float64") / ae_threshold(), 0, 1)
    return f


def potential_score(factors: dict, weights: dict, has_image=None) -> np.ndarray:
    keys = [k for k in weights if k in factors]
    total = 0.0
    wsum = 0.0
    for k in keys:
        w = weights[k]
        v = factors[k]
        if k == "image" and has_image is not None:
            w = w * np.asarray(has_image, dtype="float64")
            v = np.nan_to_num(v)
        total = total + w * v
        wsum = wsum + w
    return np.asarray(total / np.maximum(wsum, 1e-9))


REAL_DIR = ROOT / "assets" / "real_photos"


@st.cache_data(show_spinner=False)
def real_photos() -> dict:
    """Photos réelles embarquées (CC0) et leur conversion 28×28 précalculée par la même fonction que l'import."""
    with np.load(REAL_DIR / "real_photos.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def real_photo(name: str) -> Image.Image:
    return Image.open(REAL_DIR / str(name))


@st.cache_data(show_spinner=False)
def square_jpeg(name: str, size: int = 360) -> bytes:
    """Vignette carrée d'une photo réelle (recadrage centré) pour les cartes produit."""
    img = ImageOps.fit(real_photo(name).convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def sample_catalogue() -> tuple[bytes, bytes]:
    """CSV + ZIP de 70 photos réelles. Prix d'exemple = prix médian UC2 de la catégorie (démonstration)."""
    r = real_photos()
    med = category_median_price()
    rows = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, name in enumerate(r["name"]):
            z.write(REAL_DIR / str(name), arcname=str(name))
            rows.append({"product_id": f"R{i:03d}", "title": str(r["title"][i]), "description": "",
                         "price": float(med[int(r["categorie"][i])]), "image": str(name)})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8"), buf.getvalue()


def sample_catalogue_images() -> dict[str, np.ndarray]:
    r = real_photos()
    return {str(n).lower(): x for n, x in zip(r["name"], r["X"])}


CATALOGUE_ALIASES = {
    "product_id": ["product_id", "id", "sku", "id_produit", "reference"],
    "title": ["title", "titre", "name", "nom", "product_name"],
    "description": ["description", "desc", "texte", "text"],
    "price": ["price", "prix", "prix_dh", "price_dh"],
    "image": ["image", "image_file", "fichier_image", "photo", "image_name"],
}


def score_products(df: pd.DataFrame, model_key: str, price_key: str, images: dict[str, np.ndarray], weights: dict) -> pd.DataFrame:
    """Score de potentiel pour chaque ligne d'un catalogue (avec ou sans image)."""
    df = normalise_columns(df, CATALOGUE_ALIASES).copy()
    if "description" not in df.columns and "title" not in df.columns:
        raise ValueError("Il faut au moins une colonne description ou title.")
    if "price" not in df.columns:
        raise ValueError("Colonne price (prix en DH) manquante.")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna() & (df["price"] > 0)].reset_index(drop=True)
    if df.empty:
        raise ValueError("Aucune ligne avec un prix positif.")
    text = (df.get("title", pd.Series([""] * len(df))).fillna("").astype(str) + " "
            + df.get("description", pd.Series([""] * len(df))).fillna("").astype(str)).str.strip().tolist()
    n = len(df)
    img = np.zeros((n, 28, 28), dtype="float32")
    has = np.zeros(n, dtype=bool)
    if "image" in df.columns and images:
        for i, name in enumerate(df["image"].fillna("").astype(str)):
            arr = images.get(Path(name).name.lower())
            if arr is not None:
                img[i] = arr / 255.0
                has[i] = True
    probs = np.zeros((n, 10), dtype="float32")
    tprobs = text_probs(text)
    if PRODUCT_MODELS[model_key]["image"] is not None and has.any():
        idx = np.where(has)[0]
        probs[idx] = category_probs(model_key, img[idx], [text[i] for i in idx])
    probs[~has] = tprobs[~has]
    cat = probs.argmax(1)
    est = category_median_price()[cat].astype("float64")
    if has.any():
        idx = np.where(has)[0]
        est[idx] = price_estimates(price_key, img[idx], [text[i] for i in idx])
    mse = np.full(n, np.nan)
    if has.any():
        _, mse[has] = reconstruction(img[has])
    ratio = df["price"].to_numpy() / np.maximum(est, 1.0)
    factors = potential_factors(probs.max(1), tprobs[np.arange(n), cat], ratio, known_words(text), mse if has.any() else None)
    score = potential_score(factors, weights, has_image=has if "image" in factors else None)
    out = pd.DataFrame({
        "product_id": df["product_id"] if "product_id" in df.columns else np.arange(n),
        "titre": df["title"] if "title" in df.columns else [t[:60] for t in text],
        "catégorie prédite": [CLASSES[k] for k in cat],
        "confiance": probs.max(1).round(3),
        "prix (DH)": df["price"].round(2),
        "prix estimé (DH)": est.round(2),
        "source estimation": np.where(has, "modèle image + texte", "médiane de la catégorie"),
        "potentiel": score.round(3),
    })
    for k, v in factors.items():
        out["f_" + k] = np.round(v, 3)
    out["image"] = np.where(has, "oui", "non")
    out.attrs["probs"] = probs
    out.attrs["features_text"] = text
    out.attrs["images"] = img
    out.attrs["has_image"] = has
    extra = [c for c in df.columns if c not in CATALOGUE_ALIASES]
    for c in extra:
        out[c] = df[c].values
    return out
