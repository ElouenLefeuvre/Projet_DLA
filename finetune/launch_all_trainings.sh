#!/bin/bash
# Script pour lancer les 3 trainings en parallèle ou séquentiellement

echo "=========================================="
echo "Lancement des 3 fine-tunings PixArt"
echo "=========================================="

# Créer le dossier logs
mkdir -p logs

# Option 1: Lancer les 3 jobs en parallèle (si vous avez accès à plusieurs GPUs)
echo ""
echo "Soumission des jobs SLURM..."
echo ""

# Job 1: BLIP
JOB1=$(sbatch train_slurm.sh blip | awk '{print $4}')
echo "✓ Job BLIP soumis (ID: $JOB1)"

# Job 2: LLaVA
JOB2=$(sbatch train_slurm.sh llava | awk '{print $4}')
echo "✓ Job LLaVA soumis (ID: $JOB2)"

# Job 3: Gemini
JOB3=$(sbatch train_slurm.sh gemini | awk '{print $4}')
echo "✓ Job Gemini soumis (ID: $JOB3)"

echo ""
echo "=========================================="
echo "Tous les jobs sont soumis!"
echo "=========================================="
echo "IDs des jobs:"
echo "  - BLIP:   $JOB1"
echo "  - LLaVA:  $JOB2"
echo "  - Gemini: $JOB3"
echo ""
echo "Pour suivre les jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Pour voir les logs en temps réel:"
echo "  tail -f logs/pixart_${JOB1}.out  # BLIP"
echo "  tail -f logs/pixart_${JOB2}.out  # LLaVA"
echo "  tail -f logs/pixart_${JOB3}.out  # Gemini"
echo ""
echo "Pour annuler un job:"
echo "  scancel <JOB_ID>"
echo "=========================================="
