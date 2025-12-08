# Guide de Fine-tuning PixArt (Méthode Officielle)

Ce guide utilise le **repo officiel PixArt-alpha** avec extraction de features pré-calculées, comme dans le notebook `train.ipynb`.

## 🎯 Vue d'ensemble du processus

```
┌─────────────────────────────────────────────────────────┐
│ ÉTAPE 0: Setup (une seule fois)                        │
│ - Cloner PixArt-alpha                                  │
│ - Télécharger modèle pré-entraîné                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ ÉTAPE 1: Extract Features (pour chaque dataset)        │
│ - Encoder les images avec VAE → latents                │
│ - Encoder les captions avec T5 → embeddings            │
│ Durée: ~1-2h par dataset                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ ÉTAPE 2: Training (pour chaque modèle)                 │
│ - Fine-tuner PixArt avec features pré-calculées        │
│ - Pas besoin de VAE/T5 en mémoire pendant training     │
│ Durée: ~5-6h par modèle                                 │
└─────────────────────────────────────────────────────────┘
```

## 📦 Structure attendue

```
dla_projet/dla/
├── PixArt-alpha/                    # ← Repo officiel (cloné automatiquement)
│   ├── train_scripts/
│   ├── tools/
│   │   └── extract_features.py
│   └── output/
│       └── pretrained_models/
│           ├── PixArt-XL-2-512x512.pth
│           └── sd-vae-ft-ema/
│
├── datasets/                         # ← Vos datasets avec captions
│   ├── oxford_flowers102_blip/
│   │   ├── images/
│   │   │   ├── 0.png
│   │   │   └── ...
│   │   ├── captions/
│   │   │   ├── 0.txt
│   │   │   └── ...
│   │   ├── partition/
│   │   │   └── data_info.json      # ← IMPORTANT: Format spécifique requis
│   │   ├── caption_feature_wmask/   # ← Créé par extract_features.py
│   │   └── img_vae_features/        # ← Créé par extract_features.py
│   ├── oxford_flowers102_llava/
│   └── oxford_flowers102_gemini/
│
├── finetune/
│   ├── configs/
│   │   ├── flowers_blip_512.py
│   │   ├── flowers_llava_512.py
│   │   └── flowers_gemini_512.py
│   ├── setup_pixart.sh
│   ├── extract_features.sh
│   ├── train_slurm_official.sh
│   └── logs/
│
└── outputs/                          # ← Modèles fine-tunés
    ├── pixart_oxford_flowers102_blip/
    ├── pixart_oxford_flowers102_llava/
    └── pixart_oxford_flowers102_gemini/
```

## 🚀 Instructions étape par étape

### ÉTAPE 0: Setup initial (une seule fois)

#### Sur votre machine locale

```bash
cd dla_projet/dla/finetune
chmod +x setup_pixart.sh extract_features.sh train_slurm_official.sh
```

#### Sur le Mésocentre

```bash
# Se connecter
ssh votre_login@mesocentre.fr

# Aller dans le projet
cd ~/dla_projet/dla/finetune

# Lancer le setup (télécharge PixArt + modèle pré-entraîné)
bash setup_pixart.sh
```

**Ce que fait `setup_pixart.sh`** :
- Clone le repo PixArt-alpha
- Crée l'environnement conda `diffusion`
- Installe PyTorch + dépendances
- Télécharge le modèle PixArt-XL-2-512x512 (~5GB)
- Télécharge le VAE (~300MB)

⏱️ **Durée**: ~15-20 min

---

### ÉTAPE 1: Extraction des features

**IMPORTANT**: Cette étape doit être faite **AVANT** le training pour chaque dataset.

#### Pourquoi extraire les features ?

Le repo officiel PixArt utilise une approche optimisée :
1. **Avant le training** : On encode une seule fois les images (VAE) et captions (T5)
2. **Pendant le training** : On charge directement les features pré-calculées

**Avantages** :
- ✅ **Beaucoup plus rapide** : pas besoin de ré-encoder à chaque epoch
- ✅ **Moins de VRAM** : VAE et T5 pas chargés pendant le training
- ✅ **Plus stable** : exactement les mêmes features à chaque itération

#### Lancer l'extraction

```bash
# Sur le Mésocentre
cd ~/dla_projet/dla/finetune

# Extraire features pour BLIP
sbatch extract_features.sh blip

# Extraire features pour LLaVA
sbatch extract_features.sh llava

# Extraire features pour Gemini
sbatch extract_features.sh gemini
```

**Ce qui est créé** :
```
datasets/oxford_flowers102_llava/
├── caption_feature_wmask/      # Features T5 (1020 fichiers .npz)
│   ├── 0.npz
│   ├── 1.npz
│   └── ...
└── img_vae_features/           # Features VAE (1020 fichiers .npy)
    ├── 0.npy
    ├── 1.npy
    └── ...
```

⏱️ **Durée**: ~1-2h par dataset (1020 images)

**Suivre l'avancement** :
```bash
# Voir les jobs
squeue -u $USER

# Suivre les logs
tail -f logs/extract_<JOB_ID>.out
```

---

### ÉTAPE 2: Training

Une fois les features extraites, lancez le training :

#### Option A: Training séquentiel (recommandé)

```bash
cd ~/dla_projet/dla/finetune

# Job 1: BLIP
JOB1=$(sbatch train_slurm_official.sh blip | awk '{print $4}')

# Job 2: LLaVA (démarre après BLIP)
sbatch --dependency=afterok:$JOB1 train_slurm_official.sh llava

# Job 3: Gemini (démarre après LLaVA)
sbatch --dependency=afterok:$JOB2 train_slurm_official.sh gemini
```

#### Option B: Training parallèle (si 3 GPUs disponibles)

```bash
sbatch train_slurm_official.sh blip
sbatch train_slurm_official.sh llava
sbatch train_slurm_official.sh gemini
```

⏱️ **Durée estimée** :
- **V100**: ~5-6h par modèle
- **A100**: ~3-4h par modèle

**Suivre l'avancement** :
```bash
# Voir les jobs
squeue -u $USER

# Suivre les logs
tail -f logs/train_<JOB_ID>.out

# Voir sur W&B (si configuré)
# https://wandb.ai/votre-username/pixart-flowers-experiment
```

---

## ⚙️ Configuration des hyperparamètres

Les configs sont dans `finetune/configs/flowers_*_512.py`.

### Paramètres importants

| Paramètre | Valeur défaut | Description |
|-----------|---------------|-------------|
| `train_batch_size` | 16 | Batch size (réduire si OOM) |
| `num_epochs` | 200 | Nombre d'époques |
| `lr` | 2e-5 | Learning rate |
| `save_model_steps` | 500 | Fréquence de sauvegarde |
| `gradient_clip` | 0.01 | Gradient clipping |
| `grad_checkpointing` | True | Active le checkpointing |

### Adapter selon votre GPU

**Si OOM (Out of Memory)** :

Modifier dans `flowers_*_512.py`:
```python
train_batch_size = 8  # Au lieu de 16
gradient_accumulation_steps = 2  # Compense le batch plus petit
```

**Pour accélérer (si beaucoup de VRAM)** :
```python
train_batch_size = 32  # Plus gros batch
num_workers = 16  # Plus de workers
```

---

## 📊 Monitoring

### Vérifier que tout fonctionne

**Logs SLURM** :
```bash
tail -f logs/train_<JOB_ID>.out
```

**Logs de training** (chercher ces patterns):
```
Epoch 1/200
Step 10/1000: loss=0.152
Step 20/1000: loss=0.134
...
Saving checkpoint at step 500...
```

### Weights & Biases (recommandé)

1. **Configurer W&B** :
```bash
# Sur le Mésocentre
conda activate diffusion
wandb login
# Entrer votre clé API
```

2. **Voir les métriques en temps réel** :
- Allez sur https://wandb.ai
- Projet : `pixart-flowers-experiment`
- Runs : `model_blip`, `model_llava`, `model_gemini`

**Métriques trackées** :
- Loss
- Learning rate
- Gradient norm
- Images générées (tous les 200 steps)

---

## 🔍 Format du `data_info.json`

**IMPORTANT** : Votre `partition/data_info.json` doit avoir ce format exact :

```json
[
  {
    "height": 500,
    "width": 500,
    "ratio": 1.0,
    "path": "images/0.png",
    "prompt": "The image features a beautiful white flower with yellow tips, possibly a water lily..."
  },
  {
    "height": 500,
    "width": 500,
    "ratio": 1.0,
    "path": "images/1.png",
    "prompt": "This is a vibrant purple flower with delicate petals..."
  },
  ...
]
```

**Champs requis** :
- `height`, `width`, `ratio` : Dimensions de l'image
- `path` : Chemin relatif depuis la racine du dataset
- `prompt` : Caption (BLIP/LLaVA/Gemini)

---

## 🐛 Dépannage

### Erreur: "PixArt-alpha not found"

```bash
cd ~/dla_projet/dla/finetune
bash setup_pixart.sh
```

### Erreur: "Features not found"

Vous avez oublié l'étape d'extraction :
```bash
sbatch extract_features.sh llava
```

### Erreur: "Out of memory"

Réduire le batch size dans `configs/flowers_*_512.py`:
```python
train_batch_size = 8  # ou même 4
```

### Job en PENDING longtemps

```bash
# Vérifier les ressources
sinfo -p gpu

# Voir pourquoi en attente
squeue -u $USER -l
```

### Training ne converge pas

Essayer de :
- Réduire le learning rate : `lr=1e-5`
- Augmenter le warmup : `num_warmup_steps=1000`
- Vérifier les captions (pas vides, pas trop courtes)

---

## 📈 Estimation des ressources

### Par dataset (extraction features)

| Dataset | Temps | VRAM | Stockage |
|---------|-------|------|----------|
| 1020 images | ~1-2h | ~12GB | ~2GB |

### Par modèle (training)

| GPU | Batch size | Temps | VRAM |
|-----|------------|-------|------|
| V100 (32GB) | 16 | ~5-6h | ~18GB |
| A100 (40GB) | 32 | ~3-4h | ~24GB |
| RTX 3090 (24GB) | 12 | ~6-7h | ~20GB |

### Total pour l'expérience complète

**Extraction (3 datasets)** : ~3-6h
**Training (3 modèles)** : ~15-18h (séquentiel) ou ~6h (parallèle)
**TOTAL** : **~20-24h** (séquentiel) ou **~10-12h** (parallèle)

---

## ✅ Checklist

Avant de lancer le training :

- [ ] `setup_pixart.sh` exécuté avec succès
- [ ] PixArt-XL-2-512x512.pth présent dans `PixArt-alpha/output/pretrained_models/`
- [ ] Datasets transférés dans `datasets/`
- [ ] `data_info.json` au bon format
- [ ] Features extraites (dossiers `caption_feature_wmask/` et `img_vae_features/` créés)
- [ ] W&B configuré (optionnel)
- [ ] Assez de quota GPU sur le Mésocentre

---

## 🎯 Prochaines étapes

Après le training, vous aurez 3 modèles fine-tunés :
```
outputs/
├── pixart_oxford_flowers102_blip/
├── pixart_oxford_flowers102_llava/
└── pixart_oxford_flowers102_gemini/
```

**Ensuite** :
1. **Générer des images** avec chaque modèle
2. **Calculer les FID scores** pour comparer
3. **Valider l'hypothèse** : Gemini > LLaVA > BLIP

Les scripts de génération et d'évaluation seront créés dans la prochaine étape !
