# Les quatre jalons — démonstration Streamlit UC2

**Groupe 13 — Catalogue multimodal d’une marketplace**  
**Mustapha Aarab · Aya Agrigah · Toubani Badr Eddine**

Application bonus du travail UC2. Elle exécute les modèles sauvegardés des quatre jalons, sans relancer l’entraînement. Les huit modèles sélectionnés, les vocabulaires, les transformations et 200 exemples de validation par jeu sont inclus. Aucun téléchargement Kaggle ni clé API n’est nécessaire au lancement.

## Ce que l’application démontre

| Vue | Entrées et sorties |
| --- | --- |
| Jalon 1 — trois MLP | Image + description → catégorie et prix ; caractéristiques de session → score d’achat. |
| Jalon 2 — CNN et auto-encodeur | Catégorie, reconstruction d’image, erreur MSE et rejet selon le seuil validation figé. Deux anomalies synthétiques sont proposées. |
| Jalon 3 — nouvelle boutique | Image + texte du domaine B → catégorie et prix avec le réseau multitâche adapté. |
| Jalon 4 — sessions | Session complète → score d’achat GRU ; description → catégorie avec le BoW retenu après comparaison aux LSTM. |

Chaque vue permet de consulter les résultats enregistrés du notebook correspondant. Le démonstrateur présente les modèles retenus ; les architectures comparées, entraînements et ablations sont dans les quatre notebooks remis séparément.

## Installation Windows / VS Code

Installer Python **3.12** et Git. Ouvrir PowerShell dans le dossier où placer le projet :

```powershell
git clone https://github.com/mustapha005/les-quatre-jalons.git
cd les-quatre-jalons
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Ouvrir l’adresse locale affichée, généralement http://localhost:8501. Fermer le serveur avec `Ctrl+C`. Ces commandes utilisent directement Python de l’environnement : l’activation PowerShell n’est pas nécessaire. Dans VS Code : ouvrir ce dossier, puis **Python: Select Interpreter** et sélectionner `.venv\Scripts\python.exe`.

Le premier téléchargement de TensorFlow peut prendre plusieurs minutes. Ensuite, les poids sont lus depuis le dépôt et les prédictions sont calculées sur CPU, sans entraînement et sans GPU obligatoire. Garder les dossiers `artifacts/` et `data_UC2/` à leur place.

## Linux / macOS

```bash
git clone https://github.com/mustapha005/les-quatre-jalons.git
cd les-quatre-jalons
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run streamlit_app.py
```

## Publication facultative sur Streamlit Community Cloud

1. Ouvrir https://share.streamlit.io et se connecter avec le compte GitHub autorisé.
2. Créer une application depuis `mustapha005/les-quatre-jalons`.
3. Choisir la branche **main** et le fichier **streamlit_app.py**.
4. Dans les paramètres avancés, choisir **Python 3.12**, puis déployer.
5. Copier l’URL réellement attribuée dans le commentaire de remise et la tester dans une fenêtre privée.

Aucun secret n’est requis. L’hébergement n’est pas nécessaire pour le bonus demandé : le code et sa procédure de lancement sont fournis. La disponibilité et les limites mémoire du service peuvent varier ; la procédure locale reste disponible.

Documentation officielle : https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Vérification

Depuis la racine, avec le Python de l’environnement :

```powershell
.\.venv\Scripts\python.exe verify_assets.py
.\.venv\Scripts\python.exe tests/test_app.py
```

Le premier vérifie les empreintes des fichiers et les poids sélectionnés. Le second exécute les quatre vues, les derniers exemples, les descriptions vides et les anomalies synthétiques avec `streamlit.testing.v1.AppTest`. Aucun entraînement n’est effectué.

## Données, traçabilité et limites

- Images Fashion-MNIST issues des jeux pédagogiques UC2 fournis pour le cours ; descriptions, prix et sessions simulés. Source des images : https://github.com/zalandoresearch/fashion-mnist (licence MIT, voir `DATA_NOTICE.md`).
- Le démonstrateur contient uniquement les 200 premières lignes de **validation** de chacun des trois jeux : catalogue A, catalogue B, sessions. Aucun exemple de test ni d’entraînement n’est distribué dans cette application.
- `data_UC2/metadata.json` conserve les plafonds de prix calculés sur l’entraînement, comme dans le pipeline initial. Aucun réajustement sur la validation.
- Les fichiers `.keras` sont les checkpoints originaux, copiés sans modification. `ASSET_MANIFEST.json` enregistre les SHA-256 ; `frozen_protocol.json` conserve la sélection du run initial. Ses empreintes de données désignent les jeux complets des notebooks, et non les sous-ensembles du démonstrateur.
- Les résultats de `test_results.json` proviennent des évaluations déjà exécutées sur les jeux complets. L’application ne recalcule aucune métrique de test et ses 200 exemples ne constituent pas une nouvelle évaluation.
- Le BoW texte du jalon 4 a gagné la sélection validation contre les LSTM : il est identifié comme tel, et non présenté comme un réseau séquentiel.
- La détection OOD reste imparfaite ; un exemple accepté n’est pas garanti valide. Les anomalies de démonstration ne remplacent pas l’évaluation OOD officielle.
- Le modèle B ne remplace pas le modèle A. Le GRU retenu utilise des sessions complètes et un padding fixe à 60 ; une utilisation pendant la navigation nécessite une validation prospective.
- Le seuil numérique affiché ne garantit pas de cibler exactement 10 % d’un lot. La règle budgétaire est un classement des scores. Aucun coupon n’est envoyé.

## Remise du bonus

Joindre le code de ce dépôt (GitHub → **Code → Download ZIP**) et ce README à l’archive `Groupe13_UC2_Travaux.zip`, par exemple dans `bonus_streamlit/`. Ajouter le lien du dépôt au README général. Conserver aussi les quatre notebooks exécutés, la présentation, les résultats et les éléments de reproduction de la remise principale : ce dépôt est le bonus applicatif, pas leur remplacement.

## Envoyer cette archive dans votre dépôt GitHub vide

Extraire le ZIP et ouvrir PowerShell dans `les-quatre-jalons` (le dossier contenant ce README). Avec Git installé et votre compte GitHub connecté :

```powershell
git init -b main
git add .
git commit -m "Bonus Streamlit UC2 - Groupe 13"
git remote add origin https://github.com/mustapha005/les-quatre-jalons.git
git push -u origin main
```

Git peut ouvrir une fenêtre de connexion GitHub. Si Git demande votre identité pour le commit, configurer votre nom et votre adresse GitHub avec `git config user.name` et `git config user.email`, puis relancer le commit. Ne pas utiliser de push forcé : si le dépôt contient déjà des modifications, les récupérer et les intégrer avant de pousser.
