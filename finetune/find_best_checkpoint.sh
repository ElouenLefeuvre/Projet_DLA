#!/bin/bash

# Script pour analyser les logs et trouver le meilleur checkpoint
# Utilisation: ./find_best_checkpoint.sh [blip|llava|gemini]

CAPTION_TYPE=${1:-"blip"}
LOG_FILE="../outputs/pixart_oxford_flowers102_${CAPTION_TYPE}/train_log.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "ERREUR: Log file non trouvé: $LOG_FILE"
    exit 1
fi

echo "=========================================="
echo "Analyse des checkpoints pour: $CAPTION_TYPE"
echo "=========================================="
echo ""

# Extraire les loss moyennes par epoch (dernières 50 epochs)
echo "Loss moyenne par epoch (dernières 50 epochs):"
echo "Epoch | Loss moyenne | Loss min | Loss max"
echo "------|--------------|----------|----------"

grep "train_loss" "$LOG_FILE" | tail -n 3200 | \
awk -F'train_loss:' '{print $2}' | \
awk -F',' '{print $1}' | \
awk 'NR%64==0 {
    if (count > 0) {
        avg = sum/count
        printf "%-5d | %.4f       | %.4f    | %.4f\n", epoch, avg, min, max
    }
    epoch++
    sum=0; count=0; min=999; max=0
}
{
    sum+=$1; count++
    if ($1 < min) min=$1
    if ($1 > max) max=$1
}
END {
    if (count > 0) {
        avg = sum/count
        printf "%-5d | %.4f       | %.4f    | %.4f\n", epoch, avg, min, max
    }
}' epoch=151

echo ""
echo "=========================================="
echo "Recommandations:"
echo "=========================================="

# Trouver l'epoch avec la loss moyenne la plus basse
BEST_EPOCH=$(grep "train_loss" "$LOG_FILE" | tail -n 3200 | \
awk -F'train_loss:' '{print $2}' | \
awk -F',' '{print $1}' | \
awk 'NR%64==0 {
    if (count > 0) {
        avg = sum/count
        if (avg < best_avg || best_avg == 0) {
            best_avg = avg
            best_epoch = epoch
        }
    }
    epoch++
    sum=0; count=0
}
{
    sum+=$1; count++
}
END {
    print best_epoch
}' epoch=151)

echo "Meilleur epoch (loss moyenne la plus basse): $BEST_EPOCH"
echo ""
echo "Commandes suggérées:"
echo "  # Convertir le meilleur checkpoint:"
echo "  sbatch finetune/convert_checkpoint_to_diffusers.sh $CAPTION_TYPE $BEST_EPOCH"
echo ""
echo "  # Ou convertir le dernier checkpoint:"
echo "  sbatch finetune/convert_checkpoint_to_diffusers.sh $CAPTION_TYPE latest"
echo ""
echo "  # Ou convertir un epoch spécifique (ex: 190):"
echo "  sbatch finetune/convert_checkpoint_to_diffusers.sh $CAPTION_TYPE 190"
