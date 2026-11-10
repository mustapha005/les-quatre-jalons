"""Construit les exemples de photos réelles embarqués dans l'application.

Source : clothing-dataset-small d'Alexey Grigorev (CC0 1.0, domaine public)
https://github.com/alexeygrigorev/clothing-dataset-small

Usage : python scripts/build_real_examples.py /chemin/vers/clothing-dataset-small
Sélection : 10 photos par classe, tirées au hasard (graine 42) dans la partition test, sans tri sur les prédictions.
"""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import core  # noqa: E402

# classe du jeu réel → (classe Fashion-MNIST, titre d'annonce)
CLASSES = {
    "t-shirt": (0, "T-shirt"),
    "pants": (1, "Pantalon"),
    "longsleeve": (2, "Pull"),
    "dress": (3, "Robe"),
    "outwear": (4, "Veste"),
    "shirt": (6, "Chemise"),
    "shoes": (7, "Chaussures"),
}
PER_CLASS = 10


def main(src: Path):
    out = ROOT / "assets" / "real_photos"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    names, labels, titles, origin, x28 = [], [], [], [], []
    for cls, (fm, title) in CLASSES.items():
        files = sorted((src / "test" / cls).glob("*.jpg"))
        for f in rng.choice(files, size=PER_CLASS, replace=False):
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            img.thumbnail((480, 480))
            name = f"{cls}_{Path(f).stem[:8]}.jpg"
            img.save(out / name, quality=84, optimize=True)
            names.append(name)
            labels.append(fm)
            titles.append(title)
            origin.append(cls)
            x28.append(core.to_fashion_mnist(Image.open(out / name)))  # même fichier que celui distribué
    np.savez_compressed(out / "real_photos.npz", name=np.array(names), categorie=np.array(labels, dtype="int8"),
                        title=np.array(titles), source_class=np.array(origin), X=np.stack(x28).astype("uint8"))
    (out / "SOURCE.json").write_text(json.dumps({
        "dataset": "clothing-dataset-small (Alexey Grigorev)",
        "url": "https://github.com/alexeygrigorev/clothing-dataset-small",
        "license": "CC0 1.0 Universal (domaine public)",
        "split": "test", "per_class": PER_CLASS, "seed": 42,
        "class_mapping": {k: core.CLASSES[v[0]] for k, v in CLASSES.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(names)} photos → {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
