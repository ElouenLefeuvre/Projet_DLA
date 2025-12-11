#!/bin/bash
#SBATCH --job-name=clip_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --account=m25206

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running CLIP Score Evaluation..."
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate pixart

# Aller dans le dossier evaluation
cd evaluation

# Lancer le script python
# Le script va chercher ../generated_images etc. car on est dans evaluation/
python compute_clip_score.py

echo "Evaluation terminée."
