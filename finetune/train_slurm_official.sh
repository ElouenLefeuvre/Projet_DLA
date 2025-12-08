#!/bin/bash
#SBATCH --job-name=pixart_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=mesonet
#SBATCH --account=m25206
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

# Script de training PixArt avec le repo officiel
# Utilise les features VAE/T5 pré-extraites

CAPTION_TYPE=${1:-"llava"}  # blip, llava, ou gemini

echo "=========================================="
echo "Fine-tuning PixArt avec captions $CAPTION_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Activer l'environnement
source ~/.bashrc
conda activate diffusion

# Aller dans le répertoire PixArt-alpha
cd $SLURM_SUBMIT_DIR
cd PixArt-alpha

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

# Configurer wandb (optionnel)
export WANDB_PROJECT="pixart-flowers-experiment"
export WANDB_NAME="model_${CAPTION_TYPE}"

echo "Démarrage du training..."
echo ""

# Lancer le training avec le repo officiel PixArt-alpha
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29500 \
    train_scripts/train.py \
    "$CONFIG_FILE" \
    --work-dir "../outputs/pixart_oxford_flowers102_${CAPTION_TYPE}" \
    --report_to="wandb" \
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
