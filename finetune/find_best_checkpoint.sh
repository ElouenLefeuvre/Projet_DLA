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
echo "Epoch | Loss moyenne | Loss min | Loss max | Samples"
echo "------|--------------|----------|----------|--------"

# Extraire epoch et loss de chaque ligne
grep "Step/Epoch" "$LOG_FILE" | \
grep "train_loss" | \
tail -n 3200 | \
awk -F'Step/Epoch \\[|\\]\\[|/|train_loss:|,' '{
    # $2 = step, $3 = epoch, $6 = loss
    epoch = $3
    loss = $6
    gsub(/^[ \t]+/, "", loss)  # trim whitespace
    if (loss > 0) {
        sum[epoch] += loss
        count[epoch]++
        if (loss < min[epoch] || min[epoch] == 0) min[epoch] = loss
        if (loss > max[epoch]) max[epoch] = loss
    }
}
END {
    # Trier et afficher les epochs
    n = asorti(sum, sorted_epochs)
    for (i = 1; i <= n; i++) {
        e = sorted_epochs[i]
        if (count[e] > 0) {
            avg = sum[e] / count[e]
            printf "%-5d | %.4f       | %.4f    | %.4f    | %d\n", e, avg, min[e], max[e], count[e]
        }
    }
}'

echo ""
echo "=========================================="
echo "Recommandations:"
echo "=========================================="

# Trouver l'epoch avec la loss moyenne la plus basse
BEST_EPOCH=$(grep "Step/Epoch" "$LOG_FILE" | \
grep "train_loss" | \
tail -n 3200 | \
awk -F'Step/Epoch \\[|\\]\\[|/|train_loss:|,' '{
    epoch = $3
    loss = $6
    gsub(/^[ \t]+/, "", loss)
    if (loss > 0) {
        sum[epoch] += loss
        count[epoch]++
    }
}
END {
    best_avg = 999
    best_epoch = 0
    for (e in sum) {
        if (count[e] > 0) {
            avg = sum[e] / count[e]
            if (avg < best_avg) {
                best_avg = avg
                best_epoch = e
            }
        }
    }
    print best_epoch
}')

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
