#!/bin/bash
#SBATCH --job-name=pixart_finetune
#SBATCH --output=logs/pixart_%j.out
#SBATCH --error=logs/pixart_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Configuration du modèle à entraîner
# Modifier cette variable pour chaque run : "blip", "llava", ou "gemini"
CAPTION_TYPE=${1:-"llava"}  # Par défaut: llava

echo "=========================================="
echo "Démarrage du fine-tuning PixArt"
echo "Type de captions: $CAPTION_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Activer l'environnement conda/virtualenv
# À ADAPTER selon votre configuration Mésocentre
source ~/.bashrc
conda activate diffusion  # ou: source venv/bin/activate

# Aller dans le répertoire du projet
cd $SLURM_SUBMIT_DIR
cd finetune

# Créer le dossier de logs s'il n'existe pas
mkdir -p logs

# Configuration des chemins selon le type de captions
DATA_ROOT="../datasets/oxford_flowers102_${CAPTION_TYPE}"
OUTPUT_DIR="../outputs/pixart_${CAPTION_TYPE}"
RUN_NAME="model_${CAPTION_TYPE}"

# Vérifier que le dataset existe
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERREUR: Dataset non trouvé: $DATA_ROOT"
    echo "Assurez-vous d'avoir uploadé les datasets sur le Mésocentre"
    exit 1
fi

# Afficher les infos GPU
nvidia-smi

echo ""
echo "Configuration du training:"
echo "  - Data root: $DATA_ROOT"
echo "  - Output dir: $OUTPUT_DIR"
echo "  - Run name: $RUN_NAME"
echo ""

# Lancer le training
python train_pixart.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --partition_file "partition/train.json" \
    --model_name "PixArt-alpha/PixArt-XL-2-512-MS" \
    --resolution 512 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 10000 \
    --learning_rate 1e-5 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 500 \
    --gradient_checkpointing \
    --mixed_precision "fp16" \
    --save_steps 1000 \
    --logging_steps 50 \
    --dataloader_num_workers 4 \
    --use_wandb \
    --wandb_project "pixart-flowers-experiment" \
    --run_name "$RUN_NAME"

echo ""
echo "=========================================="
echo "Training terminé!"
echo "Modèle sauvegardé dans: $OUTPUT_DIR"
echo "=========================================="
