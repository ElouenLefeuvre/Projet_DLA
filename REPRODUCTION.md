# Guide de Reproduction (Commandes)

Ce fichier liste toutes les étapes et commandes nécessaires pour reproduire nos résultats de A à Z.

## 1. Préparation de l'environnement

### Sur votre machine locale (pour les notebooks)
```bash
# Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Sur le Mésocentre (pour le training/inférence)
```bash
ssh <login>@mesocentre.fr
cd ~/dla_projet/dla
```

## 2. Disponibilité des Données

Si vous ne souhaitez pas tout régénérer, voici où trouver les données déjà existantes :

### Datasets (Images + Captions)
Les datasets déjà légendés sont disponibles dans le dossier `datasets/` :
*   `datasets/oxford_flowers102_blip/` : Dataset avec captions BLIP
*   `datasets/oxford_flowers102_llava/` : Dataset avec captions LLaVA
*   `datasets/oxford_flowers102_gemini/` : Dataset avec captions Gemini
*   `datasets/oxford_flowers102_blip_test/` : Dataset avec captions Base pour les tests
*   `datasets/oxford_flowers102_llava_test/` : Dataset avec captions Base pour les tests
*   `datasets/oxford_flowers102_gemini_test/` : Dataset avec captions Base pour les tests

### Images Générées
Les images générées par nos modèles fine-tunés sont dans `generated_images/` :
*   `generated_images/blip/` : Images générées par le modèle BLIP
*   `generated_images/llava/` : Images générées par le modèle LLaVA
*   `generated_images/gemini/` : Images générées par le modèle Gemini
*   `generated_images/base/` : Images générées par le modèle de base (non fine-tuné)
*   `generated_images/fidelity_check/` : Images générées pour vérifier la fidélité

## 3. Génération des Datasets

Cette étape génère les fichiers `data_info.json` nécessaires au training. Exécutez ces notebooks localement ou sur un serveur Jupyter :

1.  **`dataset_gen.ipynb`** : Génère `datasets/oxford_flowers102/partition/data_info.json`
2.  **`dataset_gen_blip.ipynb`** : Génère `datasets/oxford_flowers102_blip/partition/data_info.json`
3.  **`dataset_gen_llava.ipynb`** : Génère `datasets/oxford_flowers102_llava/partition/data_info.json`
4.  **`dataset_gen_gemini.ipynb`** : Génère `datasets/oxford_flowers102_gemini/partition/data_info.json`

Note : Assurez-vous d'avoir votre clé API Gemini dans un fichier `.env` pour le 4ème notebook.

## 3. Setup PixArt (Une seule fois)

Sur le Mésocentre :
```bash
cd ~/dla_projet/dla/finetune
chmod +x setup_pixart.sh
./setup_pixart.sh
```
*Cela clone PixArt-alpha, installe l'env conda `diffusion`, et télécharge les poids pré-entraînés.*

## 4. Extraction des Features (Avant Training)

Il faut extraire les features VAE et T5 pour chaque dataset.

```bash
cd ~/dla_projet/dla/finetune

# Lancer les jobs d'extraction
sbatch extract_features.sh blip
sbatch extract_features.sh llava
sbatch extract_features.sh gemini
```

*Vérifiez avec `squeue` que les jobs sont finis avant de passer à la suite.*

## 5. Entraînement (Fine-tuning)

Lancer l'entraînement pour les 3 modèles.

```bash
cd ~/dla_projet/dla/finetune

# Lancer les jobs d'entraînement
sbatch train_slurm_official.sh blip
sbatch train_slurm_official.sh llava
sbatch train_slurm_official.sh gemini
```

## 6. Conversion des Checkpoints

Une fois le training fini (~20h/modèle), convertissez les checkpoints `.pth` au format Diffusers pour la génération.

```bash
cd ~/dla_projet/dla/finetune

# Convertir le dernier checkpoint de chaque modèle
bash convert_checkpoint_to_diffusers.sh blip
bash convert_checkpoint_to_diffusers.sh llava
bash convert_checkpoint_to_diffusers.sh gemini
```

*Cela crée les dossiers dans `outputs/pixart_oxford_flowers102_<model>/diffusers`.*

## 7. Génération d'Images (Inférence)

Générer les images pour le test set afin de calculer les scores.

```bash
cd ~/dla_projet/dla/generation

# Lancer la génération pour les 3 modèles sur Slurm
# Chaque job va générer des images pour les 3 sets de captions (blip, llava, gemini)

sbatch generate_one_model_slurm.sh blip \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json \
    ../datasets/oxford_flowers102_gemini/partition/data_info.json

sbatch generate_one_model_slurm.sh llava \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json \
    ../datasets/oxford_flowers102_gemini/partition/data_info.json

sbatch generate_one_model_slurm.sh gemini \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json \
    ../datasets/oxford_flowers102_gemini/partition/data_info.json
```

## 8. Évaluation

### A. Score CLIP
Calculez le score CLIP global par modèle pour mesurer l'alignement texte-image.

**Localement :**
```bash
cd ~/dla_projet/dla/evaluation
python compute_clip_score.py
```

**Sur Slurm (recommandé) :**
```bash
cd ~/dla_projet/dla/evaluation
sbatch run_eval.sh
```

### B. Calcul du FID (Fréchet Inception Distance)
Pour calculer le score FID, nous utilisons le notebook **`dataset_FID.ipynb`**.

*   **Note importante** : Le calcul du FID nécessite un fichier de statistiques sur le dataset réel (`stats_oxford_flowers.npz`). Si ce fichier est déjà présent (il a été généré lors de la phase de préparation), **vous pouvez sauter l'étape de "training"** (calcul des stats sur les images réelles) et passer directement à la comparaison avec vos images générées.

### C. Vérification de Fidélité
Générez des planches comparatives pour vérifier visuellement (Originale vs Générée).

**Localement :**
```bash
cd ~/dla_projet/dla/evaluation
# Générer 10 exemples par modèle
python check_fidelity.py --num_samples 10
```

**Sur Slurm :**
```bash
cd ~/dla_projet/dla/evaluation
sbatch run_fidelity_check.sh
```
*Les résultats seront dans `evaluation/outputs/fidelity_check`.*

### D. Membership Inference Attack (MIA)
Pour évaluer si le modèle a mémorisé les données d'entraînement (overfitting/privacy), nous utilisons une attaque par inférence d'appartenance.

**Localement :**
```bash
cd ~/dla_projet/dla/evaluation
python run_mia.py
```

**Sur Slurm :**
```bash
sbatch run_mia_slurm.sh
```
