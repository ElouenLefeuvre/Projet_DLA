#!/bin/bash
#SBATCH --job-name=generate_model
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --account=m25206

# Script SLURM pour générer des images avec UN seul modèle fine-tuné
#
# Usage:
#   sbatch generate_one_model_slurm.sh <model_type> [json1] [json2] [json3] ...
#
# Exemples:
#   sbatch generate_one_model_slurm.sh llava
#   sbatch generate_one_model_slurm.sh llava datasets/oxford_flowers102_blip_test/partition/data_info.json
#   sbatch generate_one_model_slurm.sh llava datasets/oxford_flowers102_blip_test/partition/data_info.json datasets/oxford_flowers102_llava_test/partition/data_info.json datasets/oxford_flowers102_gemini_test/partition/data_info.json

MODEL_TYPE=${1:-"llava"}  # blip, llava, ou gemini
shift  # Enlever le premier argument pour récupérer les JSON

# Récupérer les JSON en argument (ou laisser vide pour le défaut)
JSON_PATHS="$@"

echo "=========================================="
echo "Génération d'images - Modèle $MODEL_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Initialiser conda pour SLURM
eval "$(conda shell.bash hook)"

# Activer l'environnement
conda activate pixart

# Aller dans le répertoire du projet
cd $SLURM_SUBMIT_DIR/..

# Configuration
NUM_STEPS=20
GUIDANCE=4.5

# Afficher les infos GPU
nvidia-smi

echo ""
echo "Configuration:"
echo "  Modèle: $MODEL_TYPE"
echo "  Inference steps: $NUM_STEPS"
echo "  Guidance scale: $GUIDANCE"
if [ -z "$JSON_PATHS" ]; then
    echo "  JSON: Utilise les 3 par défaut (blip, llava, gemini)"
else
    echo "  JSON personnalisés:"
    for json in $JSON_PATHS; do
        echo "    - $json"
    done
fi
echo ""

# Construire la commande
CMD="python generation/generate_images_from_models.py \
    --model_type $MODEL_TYPE \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE \
    --device cuda"

# Ajouter les JSON si spécifiés
if [ -n "$JSON_PATHS" ]; then
    CMD="$CMD --json_paths $JSON_PATHS"
fi

# Lancer la génération
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Génération terminée avec succès !"
    echo "=========================================="
    echo ""
    echo "Images générées dans: generated_images/$MODEL_TYPE/"
    for JSON_TYPE in data_info_blip data_info_llava data_info_gemini; do
        COUNT=$(ls generated_images/$MODEL_TYPE/$JSON_TYPE/*.png 2>/dev/null | wc -l)
        echo "  - $JSON_TYPE: $COUNT images"
    done
    echo ""
else
    echo ""
    echo "❌ Erreur lors de la génération"
    exit 1
fi
