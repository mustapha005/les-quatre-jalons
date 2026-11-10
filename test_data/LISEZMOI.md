# Fichiers de test manuel

Générés par `python scripts/make_test_data.py <dossier clothing-dataset-small>` (graine fixe : mêmes fichiers à chaque exécution).

| Fichier | Page | Contenu |
| --- | --- | --- |
| `sessions_test.csv` | Prédire les achats | 600 parcours clients **simulés**, 12 000 événements environ, colonne `achat` (35 % d’acheteurs). Paniers abandonnés et achats sans panier inclus. |
| `sessions_test_erreurs.csv` | Prédire les achats | 31 sessions avec noms de colonnes français (`session`, `type`, `categorie`, `gamme_prix`, `duree`, `purchase`), lignes mélangées, un code type 7, une catégorie 0, une durée vide et une session de 75 événements. |
| `catalogue_test.csv` | Scorer un catalogue | 245 annonces (35 par catégorie) : titre, description de style marketplace, prix en DH, nom de photo, colonne `vendu` **simulée**. |
| `catalogue_test_images.zip` | Scorer un catalogue | Les 245 **vraies photos** correspondantes (clothing-dataset-small, CC0, partition train). |
| `catalogue_test_sans_images.csv` | Scorer un catalogue | Les mêmes annonces sans colonne image (catégorie par le texte, prix estimé = médiane). |

## Ce qui est réel et ce qui est simulé

- **Réel** : les photos de vêtements (domaine public).
- **Simulé** : les parcours clients, les prix et l’étiquette `vendu`. Un client achète plus souvent s’il ajoute au panier et navigue plus longtemps ; une annonce se vend plus souvent si son prix est sous le prix juste de sa catégorie, si l’article est neuf, si la description est riche et si la livraison est proposée. Ces fichiers servent à tester l’application, pas à mesurer la performance sur une vraie boutique.

## Résultats attendus (mesurés avant livraison)

- `sessions_test.csv` : GRU AUC 0,80 · P@10 % 87 % ; MLP riche AUC 0,78 · P@10 % 78 %.
- `sessions_test_erreurs.csv` : deux avertissements (3 événements ignorés ; 1 session tronquée à 60), puis prédiction normale.
- `catalogue_test.csv` + ZIP : 245 produits avec photo ; AUC de l’indice heuristique contre `vendu` ≈ 0,60.
- « Entraîner sur vos étiquettes » (cible `vendu`) : AUC en validation croisée ≈ 0,67 (régression logistique), 0,65 (forêt), 0,66 (boosting).

## Pas à pas

1. **Prédire les achats** : importer `sessions_test.csv`, activer « Comparer avec l’autre modèle », télécharger les prédictions. Puis importer `sessions_test_erreurs.csv` pour voir les contrôles de format.
2. **Scorer un catalogue** : importer `catalogue_test.csv` et `catalogue_test_images.zip` (détourage des 245 photos : environ une minute au premier import). Descendre à « Entraîner sur vos étiquettes », choisir `vendu`, puis « Entraîner et évaluer ».
3. Recommencer avec `catalogue_test_sans_images.csv` seul pour comparer.
4. **Potentiel produit** : « Ma photo », puis une image extraite du ZIP, avec le titre et la description de la même ligne du CSV.
