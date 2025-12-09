#!/bin/bash
#SBATCH --job-name=pixart_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --account=m25206

# Script de training PixArt avec le repo officiel
# Utilise les features VAE/T5 pré-extraites

CAPTION_TYPE=${1:-"llava"}  # blip, llava, ou gemini

echo "=========================================="
echo "Fine-tuning PixArt avec captions $CAPTION_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Initialiser conda pour SLURM
eval "$(conda shell.bash hook)"

# Activer l'environnement
conda activate pixart

# Aller dans le répertoire PixArt-alpha
cd $SLURM_SUBMIT_DIR/..
cd PixArt-alpha

# Ajouter le répertoire parent au PYTHONPATH pour trouver le module diffusion
export PYTHONPATH="${SLURM_SUBMIT_DIR}/..:$PYTHONPATH"

# Créer le dossier de logs
mkdir -p ../finetune/logs

# Configuration selon le type de captions
CONFIG_FILE="../finetune/configs/flowers_${CAPTION_TYPE}_512.py"

# Vérifier que le fichier de config existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERREUR: Configuration non trouvée: $CONFIG_FILE"
    exit 1
fi

# Vérifier que les features ont été extraites
DATASET_ROOT="../datasets/oxford_flowers102_${CAPTION_TYPE}"
if [ ! -d "${DATASET_ROOT}/caption_feature_wmask" ]; then
    echo "ERREUR: Features T5 non trouvées. Lancez d'abord extract_features.sh"
    exit 1
fi
if [ ! -d "${DATASET_ROOT}/img_vae_features" ]; then
    echo "ERREUR: Features VAE non trouvées. Lancez d'abord extract_features.sh"
    exit 1
fi

# Afficher les infos GPU
nvidia-smi

echo ""
echo "Configuration:"
echo "  Type captions: $CAPTION_TYPE"
echo "  Config file: $CONFIG_FILE"
echo "  Dataset: $DATASET_ROOT"
echo ""

# Désactiver wandb (pas configuré sur le cluster)
export WANDB_MODE="offline"
export WANDB_PROJECT="pixart-flowers-experiment"
export WANDB_NAME="model_${CAPTION_TYPE}"

echo "Démarrage du training..."
echo ""

# Assigner un port différent selon le type de caption pour éviter les conflits
case $CAPTION_TYPE in
    blip)   MASTER_PORT=29500 ;;
    llava)  MASTER_PORT=29501 ;;
    gemini) MASTER_PORT=29502 ;;
    *)      MASTER_PORT=29500 ;;
esac

echo "Master port: $MASTER_PORT"
echo ""

# Lancer le training avec le repo officiel PixArt-alpha
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    train_scripts/train.py \
    "$CONFIG_FILE" \
    --work-dir "../outputs/pixart_oxford_flowers102_${CAPTION_TYPE}" \
    --report_to="tensorboard" \
    --loss_report_name="train_loss"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training terminé avec succès!"
    echo "Modèle sauvegardé dans: ../outputs/pixart_oxford_flowers102_${CAPTION_TYPE}"
    echo "=========================================="
else
    echo ""
    echo "ERREUR lors du training"
    exit 1
fi
