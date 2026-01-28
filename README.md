# 🧠 Classification d'Images via Deep Learning (VGG16)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## 📝 Description

Ce projet implémente un modèle de **Réseau de Neurones Convolutif (CNN)** pour la classification d'images de singes. Il s'appuie sur l'architecture **VGG16** (Visual Geometry Group) et utilise la technique du **Transfer Learning** pour optimiser les performances sur un jeu de données spécifique.

Le projet a été réalisé dans le cadre du module de Deep Learning, supervisé par [M. TONDJI](https://scholar.google.com/citations?user=5xyOE3QAAAAJ&hl=en). Il démontre la capacité à pré-traiter des données visuelles, à adapter une architecture pré-entraînée et à évaluer les performances du modèle.

**Points clés :**
* Utilisation de l'architecture **VGG-16**.
* Pipeline de prétraitement d'images (Redimensionnement, Rognage, Normalisation).
* Implémentation modulaire avec scripts utilitaires Python.

## 📊 Données (Dataset)

Le modèle est entraîné sur le jeu de données **10 Monkey Species**, accessible publiquement sur Kaggle.

📥 **Source :** [Télécharger le dataset sur Kaggle](https://www.kaggle.com/datasets/slothkong/10-monkey-species)

### Organisation
Pour que le script fonctionne correctement, les données doivent être extraites et organisées comme suit :

* **Training Set :** ~1098 images réparties en 10 dossiers (n0-n9).
* **Validation Set :** ~272 images réparties en 10 dossiers (n0-n9).
* **Métadonnées :** Le fichier `monkey_labels.txt` fournit les détails taxonomiques.

### Classes Cibles
Le modèle classifie les images selon les 10 étiquettes suivantes :

| ID | Nom Commun | Nom Scientifique |
| :---: | :--- | :--- |
| **n0** | Mantled Howler | *Alouatta palliata* |
| **n1** | Patas Monkey | *Erythrocebus patas* |
| **n2** | Bald Uakari | *Cacajao calvus* |
| **n3** | Japanese Macaque | *Macaca fuscata* |
| **n4** | Pygmy Marmoset | *Cebuella pygmaea* |
| **n5** | White Headed Capuchin | *Cebus capucinus* |
| **n6** | Silvery Marmoset | *Mico argentatus* |
| **n7** | Common Squirrel Monkey | *Saimiri sciureus* |
| **n8** | Black Headed Night Monkey | *Aotus nigriceps* |
| **n9** | Nilgiri Langur | *Trachypithecus johnii* |

## 📂 Structure du Projet

L'organisation des fichiers est la suivante :

```bash
.
├── Vgg16.ipynb          # Notebook principal : Entraînement et évaluation du modèle
├── tools.py             # Script utilitaire : Fonctions de prétraitement (scaling, cropping, normalization)
├── presentation.pdf     # Support de présentation du projet (contexte théorique et résultats)
└── README.md            # Documentation du projet

```

## 🛠 Installation

Pour exécuter ce projet localement, assurez-vous d'avoir Python installé. Il est recommandé d'utiliser un environnement virtuel (venv ou conda).

1. **Cloner le dépôt :**

2. **Installer les dépendances :**
Les principales bibliothèques nécessaires sont PyTorch, NumPy et Pillow (PIL).
```bash
pip install torch torchvision numpy pillow jupyter matplotlib

```

## 🚀 Usage

Le cœur du projet réside dans le notebook Jupyter, qui fait appel au script `tools.py` pour le traitement des données.

1. **Lancer Jupyter Notebook :**
```bash
jupyter notebook
```

2. **Ouvrir le fichier `Vgg16.ipynb**`.
3. **Exécuter les cellules** séquentiellement pour :
* Charger les données.
* Importer les fonctions de `tools.py` (notamment `process_image`).
* Instancier et entraîner (ou fine-tuner) le modèle VGG16.
* Visualiser les prédictions.


*Note : L'entraînement de VGG16 étant intensif en calcul, l'utilisation d'un GPU (via Google Colab ou une machine locale compatible CUDA) est fortement recommandée.*

## ⚙️ Détails Techniques

Le fichier `tools.py` gère la préparation des images pour les rendre compatibles avec les tenseurs PyTorch attendus par VGG16 :

* **Redimensionnement :** Maintien du ratio d'aspect avec une dimension minimale de 256px.
* **Center Crop :** Découpage central de 224x224 pixels.
* **Normalisation :** Application des moyennes et écarts-types standards d'ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## 👥 Auteurs

Projet réalisé par l'équipe d'ingénieurs :

* **Steve SANOGO**
* **Habib AIDARA**
* **René Lothaire BAZIE**
---
*Projet académique - Février 2023*
```
