#!/bin/bash
#SBATCH --job-name=mia_attack
#SBATCH --output=logs/mia_%j.out
#SBATCH --error=logs/mia_%j.err
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --account=m25206

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running Membership Inference Attack (MIA)"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate pixart

# Installer les dépendances manquantes à la volée
pip install scikit-learn matplotlib

# Aller dans le dossier où le script est soumis (dla/evaluation)
cd $SLURM_SUBMIT_DIR

# 1. Attaque sur BLIP
echo ""
echo ">>> Running MIA on BLIP Model..."
python run_mia.py --model_type blip --num_samples 100

# 2. Attaque sur LLaVA
echo ""
echo ">>> Running MIA on LLaVA Model..."
python run_mia.py --model_type llava --num_samples 100

# 3. Attaque sur Gemini
echo ""
echo ">>> Running MIA on Gemini Model..."
python run_mia.py --model_type gemini --num_samples 100

echo ""
echo "=========================================="
echo "MIA Evaluation Completed!"
