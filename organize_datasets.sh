#!/bin/bash
# Script pour organiser les datasets au format attendu par PixArt

echo "=========================================="
echo "Organisation des datasets"
echo "=========================================="

# Créer la structure pour chaque dataset
for CAPTION_TYPE in blip llava gemini; do
    echo ""
    echo "Processing $CAPTION_TYPE..."

    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"

    # Créer les dossiers nécessaires
    mkdir -p "$DATASET_DIR/partition"
    mkdir -p "$DATASET_DIR/images"
    mkdir -p "$DATASET_DIR/captions"

    # Copier le data_info.json au bon endroit
    if [ -f "data_info/data_info_${CAPTION_TYPE}.json" ]; then
        cp "data_info/data_info_${CAPTION_TYPE}.json" "$DATASET_DIR/partition/data_info.json"
        echo "  ✓ Copié data_info_${CAPTION_TYPE}.json vers $DATASET_DIR/partition/data_info.json"
    else
        echo "  ✗ ERREUR: data_info/data_info_${CAPTION_TYPE}.json non trouvé"
    fi

    # Vérifier la structure
    if [ -f "$DATASET_DIR/partition/data_info.json" ]; then
        NUM_ENTRIES=$(jq '. | length' "$DATASET_DIR/partition/data_info.json" 2>/dev/null || echo "?")
        echo "  ✓ Dataset $CAPTION_TYPE: $NUM_ENTRIES entrées"
    fi
done

echo ""
echo "=========================================="
echo "Structure créée :"
echo "=========================================="
tree -L 3 datasets/ 2>/dev/null || find datasets/ -type d | head -20

echo ""
echo "=========================================="
echo "Prochaines étapes :"
echo "=========================================="
echo "1. Vérifier que vos images sont dans datasets/oxford_flowers102_*/images/"
echo "2. Vérifier que vos captions sont dans datasets/oxford_flowers102_*/captions/"
echo "3. Transférer vers Mésonet :"
echo "   rsync -avz datasets/ juvollet@juliet1.insa-rouen.fr:~/dla_projet/dla/datasets/"
echo "=========================================="
