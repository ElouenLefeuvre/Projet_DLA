# Guide d'utilisation sur Mésocentre (SLURM)

Ce guide explique comment lancer le fine-tuning de PixArt sur le Mésocentre avec SLURM.

## 📋 Prérequis

### 1. Transférer les datasets sur le Mésocentre

Depuis votre machine locale ou Google Drive:

```bash
# Se connecter au Mésocentre
ssh votre_login@mesocentre.univ.fr

# Créer la structure de dossiers
cd /home/votre_login/dla_projet/dla
mkdir -p datasets

# Transférer les datasets (depuis votre machine locale)
# Option A: rsync (recommandé)
rsync -avz --progress datasets/oxford_flowers102_blip/ \
    votre_login@mesocentre.univ.fr:~/dla_projet/dla/datasets/oxford_flowers102_blip/

rsync -avz --progress datasets/oxford_flowers102_llava/ \
    votre_login@mesocentre.univ.fr:~/dla_projet/dla/datasets/oxford_flowers102_llava/

rsync -avz --progress datasets/oxford_flowers102_gemini/ \
    votre_login@mesocentre.univ.fr:~/dla_projet/dla/datasets/oxford_flowers102_gemini/

# Option B: scp
scp -r datasets/oxford_flowers102_* \
    votre_login@mesocentre.univ.fr:~/dla_projet/dla/datasets/
```

### 2. Installer l'environnement Python

```bash
# Sur le Mésocentre
cd ~/dla_projet/dla/finetune

# Créer un environnement conda
module load anaconda3  # ou la commande appropriée
conda create -n diffusion python=3.10 -y
conda activate diffusion

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configurer Weights & Biases (optionnel)

```bash
# Sur le Mésocentre
wandb login
# Entrer votre clé API W&B
```

## 🚀 Lancement des trainings

### Option A: Lancer les 3 modèles en parallèle (si 3 GPUs disponibles)

```bash
cd ~/dla_projet/dla/finetune

# Rendre les scripts exécutables
chmod +x train_slurm.sh launch_all_trainings.sh

# Lancer les 3 jobs en parallèle
./launch_all_trainings.sh
```

**Avantages**: Fini en ~6-8h
**Inconvénients**: Nécessite 3 GPUs simultanément

### Option B: Lancer séquentiellement (1 GPU, un après l'autre)

```bash
cd ~/dla_projet/dla/finetune

# Rendre le script exécutable
chmod +x launch_sequential.sh

# Lancer le pipeline séquentiel
./launch_sequential.sh
```

**Avantages**: Nécessite seulement 1 GPU
**Inconvénients**: Durée totale ~18-24h

### Option C: Lancer manuellement un seul modèle

```bash
# Lancer uniquement LLaVA par exemple
sbatch train_slurm.sh llava

# Ou BLIP
sbatch train_slurm.sh blip

# Ou Gemini
sbatch train_slurm.sh gemini
```

## 📊 Monitoring

### Vérifier le statut des jobs

```bash
# Voir tous vos jobs
squeue -u $USER

# Voir les détails d'un job
scontrol show job <JOB_ID>

# Annuler un job
scancel <JOB_ID>
```

### Suivre les logs en temps réel

```bash
# Trouver votre JOB_ID avec squeue
squeue -u $USER

# Suivre les logs (remplacer <JOB_ID>)
tail -f logs/pixart_<JOB_ID>.out

# Voir les erreurs
tail -f logs/pixart_<JOB_ID>.err
```

### Vérifier l'utilisation GPU

```bash
# Si vous avez un job actif sur un nœud
srun --jobid=<JOB_ID> nvidia-smi
```

## ⚙️ Configuration SLURM

Le fichier `train_slurm.sh` est configuré avec:

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `--time` | 08:00:00 | Durée max: 8h |
| `--gres` | gpu:1 | 1 GPU par job |
| `--cpus-per-task` | 8 | 8 CPU cores |
| `--mem` | 64G | 64GB RAM |
| `--partition` | gpu | Partition GPU |

### Adapter à votre Mésocentre

Si votre Mésocentre a des partitions différentes, modifiez `train_slurm.sh`:

```bash
# Exemple MesoPSL
#SBATCH --partition=gpu_p100      # ou gpu_v100, gpu_a100
#SBATCH --qos=qos_gpu-t3          # QoS appropriée
#SBATCH --account=your_account    # Votre compte/projet

# Exemple Jean Zay
#SBATCH --account=abc@v100        # Votre allocation
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:v100:1         # Type de GPU spécifique
```

## 📁 Structure attendue

```
dla_projet/dla/
├── datasets/
│   ├── oxford_flowers102_blip/
│   │   ├── images/
│   │   ├── captions/
│   │   └── partition/
│   │       └── train.json
│   ├── oxford_flowers102_llava/
│   │   └── ...
│   └── oxford_flowers102_gemini/
│       └── ...
├── finetune/
│   ├── train_pixart.py
│   ├── train_slurm.sh
│   ├── launch_all_trainings.sh
│   ├── launch_sequential.sh
│   └── logs/
└── outputs/
    ├── pixart_blip/
    ├── pixart_llava/
    └── pixart_gemini/
```

## 🔧 Dépannage

### Erreur: "Dataset non trouvé"

```bash
# Vérifier que les datasets sont présents
ls -la ~/dla_projet/dla/datasets/

# Vérifier la structure
ls -la ~/dla_projet/dla/datasets/oxford_flowers102_llava/partition/
# Doit contenir train.json
```

### Erreur: "Out of memory"

Modifiez `train_slurm.sh`:

```bash
# Réduire le batch size
--train_batch_size 4 \           # Au lieu de 8
--gradient_accumulation_steps 4  # Au lieu de 2
```

### Job en PENDING trop longtemps

```bash
# Voir pourquoi le job est en attente
squeue -u $USER -l

# Vérifier les ressources disponibles
sinfo -p gpu
```

### Environnement conda non trouvé

Modifiez `train_slurm.sh`:

```bash
# Remplacer cette ligne:
conda activate diffusion

# Par le chemin absolu:
source /home/votre_login/miniconda3/etc/profile.d/conda.sh
conda activate diffusion

# Ou pour virtualenv:
source /home/votre_login/dla_projet/venv/bin/activate
```

## 📈 Estimation des ressources

### Par modèle (10,000 steps)

| GPU Type | Temps | VRAM utilisée | Coût (si applicable) |
|----------|-------|---------------|---------------------|
| V100 (32GB) | ~5-6h | ~18GB | - |
| A100 (40GB) | ~3-4h | ~20GB | - |
| RTX 3090 (24GB) | ~5-6h | ~16GB | - |

### Total pour les 3 modèles

- **Parallèle** (3 GPUs): ~6-8h
- **Séquentiel** (1 GPU): ~18-24h

## 📝 Checklist avant lancement

- [ ] Datasets transférés sur le Mésocentre
- [ ] Environnement Python installé (`conda env list`)
- [ ] Scripts exécutables (`chmod +x *.sh`)
- [ ] W&B configuré (si `--use_wandb`)
- [ ] Partition SLURM adaptée à votre centre
- [ ] Assez de quota/heures de calcul disponibles

## 🎯 Après le training

Une fois les trainings terminés, vous aurez:

```
outputs/
├── pixart_blip/
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── final_model/
├── pixart_llava/
│   └── final_model/
└── pixart_gemini/
    └── final_model/
```

Prochaine étape: **Générer des images** et **calculer les scores FID** !
