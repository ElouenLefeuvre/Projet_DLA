#!/bin/bash
#SBATCH --job-name=convert_checkpoint
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --account=m25206

# Script de conversion d'un checkpoint vers le format diffusers
# Utilisation: sbatch convert_checkpoint_to_diffusers.sh [blip|llava|gemini] [epoch_number|latest]
# Exemples:
#   sbatch convert_checkpoint_to_diffusers.sh blip latest    # Dernier checkpoint
#   sbatch convert_checkpoint_to_diffusers.sh blip 150       # Checkpoint de l'epoch 150
#   sbatch convert_checkpoint_to_diffusers.sh blip           # Dernier checkpoint (par défaut)

CAPTION_TYPE=${1:-"llava"}  # blip, llava, ou gemini
CHECKPOINT_EPOCH=${2:-"latest"}  # Numéro d'epoch ou "latest"

echo "=========================================="
echo "Conversion checkpoint vers diffusers"
echo "Type de captions: $CAPTION_TYPE"
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

# Créer le dossier de logs
mkdir -p ../finetune/logs

# Définir les chemins
CHECKPOINT_DIR="../outputs/pixart_oxford_flowers102_${CAPTION_TYPE}/checkpoints"
OUTPUT_DIR="../outputs/pixart_oxford_flowers102_${CAPTION_TYPE}_diffusers"

# Vérifier que le dossier de checkpoints existe
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERREUR: Dossier de checkpoints non trouvé: $CHECKPOINT_DIR"
    echo "Assurez-vous d'avoir lancé le training d'abord."
    exit 1
fi

# Trouver le checkpoint approprié
if [ "$CHECKPOINT_EPOCH" = "latest" ]; then
    # Trouver le checkpoint le plus récent (par date de modification)
    LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pth 2>/dev/null | head -n 1)
    CHECKPOINT_DESC="le plus récent (par date)"
elif [[ "$CHECKPOINT_EPOCH" =~ ^[0-9]+$ ]]; then
    # Chercher un checkpoint spécifique pour cet epoch
    # Format: epoch_XXX_step_YYY.pth
    # Trier par numéro de step pour prendre le plus élevé
    LATEST_CHECKPOINT=$(ls "$CHECKPOINT_DIR"/epoch_${CHECKPOINT_EPOCH}_step_*.pth 2>/dev/null | \
        sed 's/.*step_\([0-9]*\)\.pth/\1 &/' | \
        sort -n | \
        tail -n 1 | \
        cut -d' ' -f2-)
    CHECKPOINT_DESC="epoch $CHECKPOINT_EPOCH"
else
    echo "ERREUR: CHECKPOINT_EPOCH doit être 'latest' ou un numéro d'epoch"
    exit 1
fi

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERREUR: Aucun checkpoint trouvé dans $CHECKPOINT_DIR"
    if [[ "$CHECKPOINT_EPOCH" =~ ^[0-9]+$ ]]; then
        echo "Aucun checkpoint trouvé pour l'epoch $CHECKPOINT_EPOCH"
        echo "Checkpoints disponibles:"
        ls "$CHECKPOINT_DIR"/*.pth 2>/dev/null | sed 's/.*epoch_/  epoch_/' | head -n 10
    fi
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Type captions: $CAPTION_TYPE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Checkpoint sélectionné: $CHECKPOINT_DESC"
echo "  Fichier: $LATEST_CHECKPOINT"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Créer le dossier de sortie
mkdir -p "$OUTPUT_DIR"

# Afficher les infos GPU
nvidia-smi

echo ""
echo "Démarrage de la conversion..."
echo ""

# Convertir le checkpoint vers le format diffusers
python tools/convert_pixart_alpha_to_diffusers.py \
    --orig_ckpt_path "$LATEST_CHECKPOINT" \
    --dump_path "$OUTPUT_DIR" \
    --only_transformer=True \
    --image_size 512 \
    --multi_scale_train=False

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Conversion terminée avec succès!"
    echo "Checkpoint source: $LATEST_CHECKPOINT"
    echo "Modèle diffusers sauvegardé dans: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "ERREUR lors de la conversion"
    exit 1
fi
