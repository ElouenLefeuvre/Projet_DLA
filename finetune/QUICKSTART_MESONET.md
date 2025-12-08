# Guide de démarrage rapide - Mésonet

Guide complet pour lancer votre projet de fine-tuning PixArt sur Mésonet.

## 🎯 Checklist avant de commencer

- [ ] Connecté à Mésonet : `ssh votre_login@mesonet.insa-rouen.fr`
- [ ] Datasets avec captions générés sur Google Colab (BLIP/LLaVA/Gemini)
- [ ] Fichiers `data_info.json` créés pour chaque dataset

---

## 📦 ÉTAPE 1 : Transférer vos données sur Mésonet

### Depuis votre machine locale (pas depuis Mésonet)

```bash
# Ouvrir un nouveau terminal LOCAL (pas Mésonet)

# Aller dans votre dossier projet
cd /home/jvollet/Bureau/dla_projet/dla

# Transférer le dossier complet vers Mésonet
rsync -avz --progress . \
    votre_login@mesonet.insa-rouen.fr:~/dla_projet/dla/

# OU si vous avez les datasets sur Google Drive/Colab
# Téléchargez-les d'abord localement, puis :
rsync -avz --progress datasets/ \
    votre_login@mesonet.insa-rouen.fr:~/dla_projet/dla/datasets/
```

**⏱️ Durée** : ~10-30 min (selon la taille des datasets)

**Ce qui est transféré** :
```
~/dla_projet/dla/
├── datasets/                    # Vos 3 datasets
│   ├── oxford_flowers102_blip/
│   ├── oxford_flowers102_llava/
│   └── oxford_flowers102_gemini/
├── finetune/                    # Scripts de training
│   ├── configs/
│   ├── setup_pixart.sh
│   ├── extract_features.sh
│   └── train_slurm_official.sh
└── notebooks/                   # (optionnel)
```

---

## 🔧 ÉTAPE 2 : Setup initial sur Mésonet

### Se connecter et configurer l'environnement

```bash
# Sur Mésonet
ssh votre_login@mesonet.insa-rouen.fr

# Aller dans le projet
cd ~/dla_projet/dla

# Vérifier que tout est bien arrivé
ls -la datasets/
ls -la finetune/

# Rendre les scripts exécutables
chmod +x finetune/*.sh

# Lancer le setup (clone PixArt + download modèle)
cd finetune
bash setup_pixart.sh
```

**⏱️ Durée** : ~15-20 min

**Ce que fait setup_pixart.sh** :
1. Clone le repo PixArt-alpha
2. Crée l'environnement conda `diffusion`
3. Installe PyTorch + dépendances
4. Télécharge PixArt-XL-2-512x512.pth (~5GB)
5. Télécharge VAE (~300MB)

**Vérifier que tout est OK** :
```bash
# Après setup
ls -la ../PixArt-alpha/output/pretrained_models/
# Vous devriez voir :
# - PixArt-XL-2-512x512.pth
# - sd-vae-ft-ema/
```

---

## 🚀 ÉTAPE 3 : Extraction des features (OBLIGATOIRE avant training)

### Lancer l'extraction pour les 3 datasets

```bash
# Sur Mésonet, dans ~/dla_projet/dla/finetune

# Créer le dossier logs
mkdir -p logs

# Soumettre les 3 jobs d'extraction
sbatch extract_features.sh blip
sbatch extract_features.sh llava
sbatch extract_features.sh gemini
```

**⏱️ Durée** : ~1-2h par dataset (3-6h total en parallèle)

### Vérifier l'état des jobs

```bash
# Voir vos jobs en cours
squeue -u $USER

# Output exemple :
#  JOBID  PARTITION  NAME           USER      ST  TIME
#  12345  gpu        extract_feat   jvollet   R   0:15
#  12346  gpu        extract_feat   jvollet   R   0:15
#  12347  gpu        extract_feat   jvollet   R   0:15
```

### Suivre les logs en temps réel

```bash
# Remplacer <JOB_ID> par votre numéro de job
tail -f logs/extract_<JOB_ID>.out

# Vous devriez voir :
# Extracting features for oxford_flowers102_llava
# Processing: 100%|██████████| 1020/1020
```

### Annuler un job si nécessaire

```bash
scancel <JOB_ID>
```

### Vérifier que les features sont créées

```bash
# Après extraction (quand jobs terminés)
ls -la ../datasets/oxford_flowers102_llava/

# Vous devriez voir :
# drwxr-xr-x caption_feature_wmask/   # Features T5
# drwxr-xr-x img_vae_features/        # Features VAE
```

**Compter les fichiers** :
```bash
# Devrait afficher 1020 (nombre d'images)
ls ../datasets/oxford_flowers102_llava/caption_feature_wmask/ | wc -l
ls ../datasets/oxford_flowers102_llava/img_vae_features/ | wc -l
```

---

## 🎓 ÉTAPE 4 : Lancer le training

### Une fois les features extraites, lancer le training

#### Option A : Training séquentiel (1 GPU, un après l'autre)

```bash
# Sur Mésonet, dans ~/dla_projet/dla/finetune

# Job 1 : BLIP
JOB1=$(sbatch train_slurm_official.sh blip | awk '{print $4}')
echo "Job BLIP lancé : $JOB1"

# Job 2 : LLaVA (attend la fin de BLIP)
JOB2=$(sbatch --dependency=afterok:$JOB1 train_slurm_official.sh llava | awk '{print $4}')
echo "Job LLaVA lancé : $JOB2 (démarre après $JOB1)"

# Job 3 : Gemini (attend la fin de LLaVA)
JOB3=$(sbatch --dependency=afterok:$JOB2 train_slurm_official.sh gemini | awk '{print $4}')
echo "Job Gemini lancé : $JOB3 (démarre après $JOB2)"
```

**⏱️ Durée totale** : ~15-18h (5-6h × 3 modèles)

---

#### Option B : Training parallèle (3 GPUs simultanés)

```bash
# Lancer les 3 en même temps
sbatch train_slurm_official.sh blip
sbatch train_slurm_official.sh llava
sbatch train_slurm_official.sh gemini
```

**⏱️ Durée totale** : ~5-6h (si 3 GPUs disponibles)

---

### Vérifier l'état du training

```bash
# Voir tous vos jobs
squeue -u $USER

# Format détaillé
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

**États possibles** :
- `PD` (PENDING) : En attente de ressources
- `R` (RUNNING) : En cours d'exécution
- `CG` (COMPLETING) : En train de se terminer

### Suivre les logs du training

```bash
# Suivre le log en temps réel
tail -f logs/train_<JOB_ID>.out

# Vous verrez :
# Epoch 1/200
# Step 10: loss=0.152
# Step 20: loss=0.134
# ...
# Saving checkpoint at step 500...
```

---

## 📊 ÉTAPE 5 : Monitoring avec Weights & Biases (optionnel)

### Configurer W&B

```bash
# Sur Mésonet
conda activate diffusion
wandb login
# Coller votre clé API W&B
```

### Voir les métriques en temps réel

1. Aller sur https://wandb.ai
2. Projet : `pixart-flowers-experiment`
3. Runs : `model_blip`, `model_llava`, `model_gemini`

**Métriques trackées** :
- Loss curve
- Learning rate
- Samples générés tous les 200 steps

---

## 🔍 Commandes utiles Mésonet

### Gestion des jobs

```bash
# Voir vos jobs
squeue -u $USER

# Voir tous les jobs GPU
squeue -p gpu

# Détails d'un job
scontrol show job <JOB_ID>

# Annuler un job
scancel <JOB_ID>

# Annuler tous vos jobs
scancel -u $USER

# Temps restant pour un job
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.10L"
```

### Vérifier les ressources disponibles

```bash
# Voir les partitions GPU
sinfo -p gpu

# Voir les nœuds disponibles
sinfo -N -p gpu

# Utilisation GPU en temps réel (si job actif)
srun --jobid=<JOB_ID> nvidia-smi
```

### Gestion des fichiers

```bash
# Espace disque utilisé
du -sh ~/dla_projet/

# Quota disque
quota -s

# Nettoyer les checkpoints intermédiaires
rm -rf ../outputs/*/checkpoint-*
# (Garde seulement final_model)
```

---

## 🐛 Dépannage

### Erreur : "PixArt-alpha not found"

```bash
cd ~/dla_projet/dla/finetune
bash setup_pixart.sh
```

### Erreur : "Features not found"

Vous avez oublié l'extraction :
```bash
sbatch extract_features.sh llava
```

Attendez que le job se termine avant de lancer le training.

### Erreur : "Out of memory" pendant training

Modifier `finetune/configs/flowers_*_512.py` :
```python
train_batch_size = 8  # Au lieu de 16
```

Puis relancer :
```bash
sbatch train_slurm_official.sh llava
```

### Job en PENDING trop longtemps

```bash
# Voir pourquoi
squeue -u $USER -l

# Vérifier les ressources disponibles
sinfo -p gpu

# Adapter la partition si nécessaire
# Modifier train_slurm_official.sh :
#SBATCH --partition=gpu_short  # ou autre partition dispo
```

### Vérifier les erreurs dans les logs

```bash
# Voir les erreurs
cat logs/train_<JOB_ID>.err

# Dernières lignes du log
tail -50 logs/train_<JOB_ID>.out
```

---

## 📁 Structure finale après training

```
~/dla_projet/dla/
├── PixArt-alpha/
│   └── output/pretrained_models/
│       ├── PixArt-XL-2-512x512.pth
│       └── sd-vae-ft-ema/
│
├── datasets/
│   ├── oxford_flowers102_blip/
│   │   ├── images/
│   │   ├── captions/
│   │   ├── partition/data_info.json
│   │   ├── caption_feature_wmask/    # ← Créé par extract
│   │   └── img_vae_features/         # ← Créé par extract
│   ├── oxford_flowers102_llava/
│   └── oxford_flowers102_gemini/
│
├── outputs/
│   ├── pixart_oxford_flowers102_blip/
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   └── final_model/              # ← Modèle final
│   ├── pixart_oxford_flowers102_llava/
│   └── pixart_oxford_flowers102_gemini/
│
└── finetune/
    ├── logs/
    │   ├── extract_12345.out
    │   └── train_12346.out
    └── ...
```

---

## ✅ Timeline complète

| Étape | Durée | Commande |
|-------|-------|----------|
| **1. Transfer** | 10-30 min | `rsync` depuis local |
| **2. Setup** | 15-20 min | `bash setup_pixart.sh` |
| **3. Extract features** | 3-6h | `sbatch extract_features.sh` × 3 |
| **4. Training (parallèle)** | 5-6h | `sbatch train_slurm_official.sh` × 3 |
| **4. Training (séquentiel)** | 15-18h | Avec `--dependency` |
| **TOTAL (parallèle)** | **~10-12h** | |
| **TOTAL (séquentiel)** | **~20-24h** | |

---

## 🎯 Prochaines étapes

Une fois le training terminé :

1. **Vérifier les modèles** :
```bash
ls -la outputs/*/final_model/
```

2. **Télécharger les modèles** (optionnel, pour inférence locale) :
```bash
# Depuis votre machine locale
rsync -avz --progress \
    votre_login@mesonet.insa-rouen.fr:~/dla_projet/dla/outputs/ \
    ./outputs/
```

3. **Générer des images** et **calculer FID** (prochains scripts à créer)

---

## 💡 Conseils

### Pour gagner du temps
- Lancez l'extraction des 3 datasets en parallèle (si GPUs dispo)
- Utilisez W&B pour monitorer sans se connecter en SSH
- Gardez les checkpoints importants (500, 1000, etc.) pour comparer

### Pour économiser l'espace
- Supprimez les checkpoints intermédiaires après training
- Compressez les datasets avant transfer : `tar -czf datasets.tar.gz datasets/`

### Pour débugger
- Testez d'abord avec un seul dataset (LLaVA)
- Réduisez `num_epochs` à 10 pour un test rapide
- Vérifiez les logs régulièrement

---

Bonne chance avec votre expérience ! 🚀
