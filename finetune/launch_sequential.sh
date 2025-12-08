#!/bin/bash
# Script pour lancer les 3 trainings séquentiellement (un après l'autre)
# Utile si vous n'avez accès qu'à 1 GPU à la fois

echo "=========================================="
echo "Lancement séquentiel des 3 fine-tunings"
echo "=========================================="

mkdir -p logs

# Job 1: BLIP
echo ""
echo "[1/3] Soumission du training BLIP..."
JOB1=$(sbatch train_slurm.sh blip | awk '{print $4}')
echo "✓ Job BLIP soumis (ID: $JOB1)"

# Job 2: LLaVA (dépend de la fin du job BLIP)
echo "[2/3] Soumission du training LLaVA (après BLIP)..."
JOB2=$(sbatch --dependency=afterok:$JOB1 train_slurm.sh llava | awk '{print $4}')
echo "✓ Job LLaVA soumis (ID: $JOB2, attend la fin de $JOB1)"

# Job 3: Gemini (dépend de la fin du job LLaVA)
echo "[3/3] Soumission du training Gemini (après LLaVA)..."
JOB3=$(sbatch --dependency=afterok:$JOB2 train_slurm.sh gemini | awk '{print $4}')
echo "✓ Job Gemini soumis (ID: $JOB3, attend la fin de $JOB2)"

echo ""
echo "=========================================="
echo "Pipeline séquentiel configuré!"
echo "=========================================="
echo "Ordre d'exécution:"
echo "  1. BLIP   (Job $JOB1)"
echo "  2. LLaVA  (Job $JOB2) → démarre après BLIP"
echo "  3. Gemini (Job $JOB3) → démarre après LLaVA"
echo ""
echo "Durée estimée totale: ~18-24h (3 x 6-8h)"
echo ""
echo "Pour suivre:"
echo "  squeue -u \$USER"
echo "  tail -f logs/pixart_${JOB1}.out"
echo "=========================================="
