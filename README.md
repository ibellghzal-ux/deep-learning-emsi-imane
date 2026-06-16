# Projet Deep Learning — EMSI Casablanca 2025-2026

**Module** : Deep Learning | **Filière** : Informatique / Data Science | **Auteur** : Imane Bellghzal
**Établissement** : EMSI Casablanca

---

## Idée générale du projet

Ce projet de fin de module explore trois familles architecturales fondamentales du deep
learning, chacune adaptée à une modalité de données différente, sur le thème de
**l'environnement et du climat** :

| Partie | Architecture | Données | Tâche |
|--------|--------------|---------|-------|
| I | MLP (Perceptron Multicouche) | **Air Quality and Pollution Assessment** (Kaggle) — 5000 mesures environnementales, 9 variables (PM2.5, PM10, NO2, SO2, CO, température, humidité, proximité industrielle, densité de population) | Classification de la qualité de l'air (Good / Moderate / Poor / Hazardous) |
| II | CNN (Réseau Convolutionnel) | **Intel Image Classification** (Kaggle) — images de paysages 64x64 en niveaux de gris | Classification de paysages (buildings, forest, glacier, mountain, sea, street) |
| III | RNN / LSTM / GRU + Seq2Seq | **Climate Change News Articles** (Kaggle) — titres et descriptions d'actualités climatiques | Classification thématique + résumé automatique (Seq2Seq) |

L'ensemble des expériences est conduit sous **PyTorch** avec une graine aléatoire fixée
(`SEED=42`) pour garantir la reproductibilité. Chaque partie articule fondements théoriques,
implémentation commentée, expérimentation comparative et analyse critique, conformément au
cahier des charges du module.

> **Note sur les données** : un échantillon de chaque dataset est fourni directement dans le
> dossier `data/` du dépôt (générés avec `SEED=42`). Chaque notebook charge ces fichiers en
> priorité. À défaut, il tente un téléchargement automatique du dataset Kaggle correspondant
> via `kagglehub`, puis, en dernier recours, génère des données synthétiques calibrées sur
> la même structure statistique — garantissant que les notebooks restent intégralement
> exécutables et reproductibles de bout en bout.

---

## Structure du dépôt

```
deep-learning-emsi-imane/
│
├── notebooks/
│   ├── 01_MLP_AirQuality.ipynb                       # Partie I : MLP + Air Quality
│   ├── 02_CNN_LandscapeClassification.ipynb          # Partie II : CNN + Intel Image Classification
│   └── 03_RNN_LSTM_GRU_Seq2Seq_ClimateNews.ipynb      # Partie III : RNN/LSTM/GRU + Seq2Seq
│
├── data/
│   ├── air_quality.csv                              # 5000 lignes, 9 variables + cible (Partie I)
│   └── climate_news.csv                             # titres/descriptions + catégorie (Partie III)
│
├── rapport/
│   └── Rapport_DeepLearning_Imane.pdf                # Rapport scientifique complet (20 pages)
│
├── figures/                                          # Figures générées par les notebooks
│
├── requirements.txt                                  # Dépendances Python
└── README.md
```

---

## Données

Chaque notebook charge en priorité le jeu de données fourni localement dans `data/`
(échantillon prêt à l'emploi, généré avec `SEED=42`). En l'absence de ces fichiers, le
notebook tente un téléchargement Kaggle via `kagglehub`, puis, en dernier recours, génère un
jeu de données synthétique calibré sur la même structure statistique — garantissant ainsi
l'exécutabilité intégrale des notebooks dans tous les environnements.

| Partie | Fichier(s) local(aux) | Dataset Kaggle de référence |
|--------|------------------------|------------------------------|
| I | `data/air_quality.csv` (5000 lignes) | Air Quality and Pollution Assessment (`mujtabamatin/air-quality-and-pollution-assessment`) |
| II | — (généré automatiquement par le notebook, 600 images, 100/classe, 64x64, SEED=42) | Intel Image Classification (`puneet6060/intel-image-classification`) |
| III | `data/climate_news.csv` (2000 lignes) | Climate Change News Articles (`amritpal333/climate-news-articles`) |

> Pour la Partie II, les images de paysages ne sont pas stockées dans le dépôt (volume trop
> important pour un upload pratique) : elles sont générées automatiquement par le notebook
> au lancement (génération procédurale par classe, reproductible grâce à `SEED=42`), ou
> téléchargées depuis Kaggle si `kagglehub` est disponible.

---

## Contenu détaillé des notebooks

### 01_MLP_AirQuality.ipynb — Partie I : MLP sur Air Quality

- Configuration de la reproductibilité (`SEED=42`, détection device GPU/CPU)
- Rappel théorique : `nn.Module`, paramètres, gradient, `state_dict`, device, forward/backward
- Chargement du dataset Air Quality (téléchargement Kaggle ou génération synthétique calibrée)
- Préparation des données : nettoyage, encodage, split stratifié 70/15/15, `StandardScaler`
  (fit sur train uniquement)
- Implémentation MLP en deux variantes : `nn.Sequential` et classe personnalisée `AirQualityMLP`
- Inspection des paramètres (`named_parameters()`, `state_dict()`)
- Comparaison de 3 stratégies d'initialisation : Gaussienne, Constante, Xavier
- Sauvegarde / rechargement du meilleur modèle (`best_air_quality_mlp.pt`)
- Évaluation : accuracy, precision, recall, F1, matrice de confusion, courbes ROC one-vs-rest
- Analyse critique et question de synthèse

### 02_CNN_LandscapeClassification.ipynb — Partie II : CNN sur Intel Image Classification

- Pourquoi un MLP est peu adapté aux images ; idées fondatrices des CNN (localité, partage des
  poids, hiérarchie)
- Calculs manuels de dimensions de sortie (convolution, padding, stride, pooling)
- Implémentations manuelles validées numériquement : `corr2d_manual()`, `max_pool_manual()`,
  `avg_pool_manual()` (comparées à `nn.Conv2d` / `F.max_pool2d` / `F.avg_pool2d`)
- Chargement du dataset Intel Image Classification (Kaggle ou génération synthétique de
  paysages par classe)
- Architecture `LandscapeCNN` : 3 blocs Conv-BatchNorm-ReLU-Pool, convolution 1x1, classifieur
  FC avec dropout
- Entraînement avec `CosineAnnealingLR` et early stopping
- Étude d'ablation sur 5 configurations (1x1, avg/max pooling, nombre de filtres)
- Visualisation des cartes de caractéristiques et des filtres appris (`get_feature_maps()`)
- Comparaison MLP vs CNN : accuracy, nombre de paramètres, courbes ROC superposées
- Analyse critique et question de synthèse

### 03_RNN_LSTM_GRU_Seq2Seq_ClimateNews.ipynb — Partie III : RNN/LSTM/GRU + Seq2Seq

- Modèle de langage probabiliste, factorisation par règle de chaîne, perplexité
- Chargement du dataset Climate Change News Articles (Kaggle ou génération synthétique de
  titres/descriptions par catégorie)
- Construction du vocabulaire avec tokens spéciaux (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`)
- Implémentation et comparaison de `RecurrentClassifier` (RNN / LSTM / GRU) sur la
  classification thématique (5 catégories climatiques)
- Illustration expérimentale du BPTT et de l'effet du gradient clipping (sans clipping,
  clip=1.0, clip=5.0)
- Architecture Seq2Seq : `Encoder` LSTM + `Decoder` LSTM avec teacher forcing (ratio=0.5)
- Décodage glouton (`greedy_decode`) et beam search (`beam_search_decode`, k=2, 3, 5)
- Évaluation BLEU-1 sur le jeu de test, exemples qualitatifs de résumés générés
- Analyse critique et question de synthèse

---

## Installation et utilisation

### Prérequis

- Python 3.10 ou supérieur
- GPU recommandé (CUDA) mais CPU supporté

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancer les notebooks

```bash
jupyter notebook notebooks/
```

Ouvrir dans l'ordre :

1. `01_MLP_AirQuality.ipynb` — Partie I
2. `02_CNN_LandscapeClassification.ipynb` — Partie II
3. `03_RNN_LSTM_GRU_Seq2Seq_ClimateNews.ipynb` — Partie III

Chaque notebook est autonome et peut être exécuté indépendamment sur Google Colab ou Kaggle
Notebooks (GPU T4 recommandé pour les Parties II et III).

---

## Dépendances principales

| Librairie | Usage |
|-----------|-------|
| torch | Modèles, boucles d'entraînement, autograd |
| torchvision | Transforms et utilitaires image |
| scikit-learn | Métriques, StandardScaler, train_test_split |
| matplotlib, seaborn | Courbes, matrices de confusion, feature maps |
| numpy, pandas | Manipulation des données |
| nltk | Calcul du score BLEU |
| kagglehub | Téléchargement des datasets Kaggle |

---

## Références

- Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press.
- LeCun et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
- Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*.
- Cho et al. (2014). Learning phrase representations using RNN encoder-decoder. *EMNLP*.
- Glorot & Bengio (2010). Understanding the difficulty of training deep feedforward networks. *AISTATS*.
- Paszke et al. (2019). PyTorch: An imperative style high-performance deep learning library. *NeurIPS*.

---

Année universitaire 2025–2026 — EMSI Casablanca
