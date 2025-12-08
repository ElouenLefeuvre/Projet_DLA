#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

# Script pour extraire les features VAE et T5 avant le training
# Cette étape DOIT être faite AVANT le training

CAPTION_TYPE=${1:-"llava"}  # blip, llava, ou gemini

echo "=========================================="
echo "Extraction des features"
echo "Type de captions: $CAPTION_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Activer l'environnement
source ~/.bashrc
conda activate diffusion

# Aller dans le répertoire PixArt-alpha
cd $SLURM_SUBMIT_DIR
cd PixArt-alpha

# Configuration des chemins
DATASET_ROOT="../datasets/oxford_flowers102_${CAPTION_TYPE}"
JSON_PATH="${DATASET_ROOT}/partition/data_info.json"
T5_SAVE_ROOT="${DATASET_ROOT}/caption_feature_wmask"
VAE_SAVE_ROOT="${DATASET_ROOT}/img_vae_features"
PRETRAINED_MODELS_DIR="output/pretrained_models"

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  JSON: $JSON_PATH"
echo "  T5 features: $T5_SAVE_ROOT"
echo "  VAE features: $VAE_SAVE_ROOT"
echo ""

# Vérifier que le dataset existe
if [ ! -f "$JSON_PATH" ]; then
    echo "ERREUR: JSON non trouvé: $JSON_PATH"
    exit 1
fi

# Créer les dossiers de sortie
mkdir -p "$T5_SAVE_ROOT"
mkdir -p "$VAE_SAVE_ROOT"

# Afficher les infos GPU
nvidia-smi

echo ""
echo "Démarrage de l'extraction des features..."
echo ""

# Extraire les features
python tools/extract_features.py \
    --img_size 512 \
    --json_path "$JSON_PATH" \
    --t5_save_root "$T5_SAVE_ROOT" \
    --vae_save_root "$VAE_SAVE_ROOT" \
    --pretrained_models_dir "$PRETRAINED_MODELS_DIR" \
    --dataset_root "$DATASET_ROOT"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Extraction terminée avec succès!"
    echo "Features T5 sauvegardées dans: $T5_SAVE_ROOT"
    echo "Features VAE sauvegardées dans: $VAE_SAVE_ROOT"
    echo "=========================================="
else
    echo ""
    echo "ERREUR lors de l'extraction des features"
    exit 1
fi
