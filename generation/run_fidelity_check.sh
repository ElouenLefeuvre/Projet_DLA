#!/bin/bash
#SBATCH --job-name=fidelity_check
#SBATCH --output=logs/fidelity_check_%j.out
#SBATCH --error=logs/fidelity_check_%j.err
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --account=m25206

echo "=========================================="
echo "Fidelity Check - Comparaison modèles fine-tunés"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate pixart

# Aller dans le dossier racine du projet
cd $SLURM_SUBMIT_DIR/..

# Afficher les infos GPU
nvidia-smi

echo ""
echo "Lancement du script de vérification de fidélité..."
echo ""

# Lancer la génération
python generation/check_fidelity.py \
    --num_samples 10 \
    --num_inference_steps 20 \
    --guidance_scale 4.5 \
    --models blip llava gemini

if [ $? -eq 0 ]; then
    echo "Fidelity check completed successfully!"
else
    echo "Error during fidelity check!"
    exit 1
fi
