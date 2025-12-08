#!/bin/bash
# Script pour reconstruire les datasets uniquement depuis les JSON existants
# Sans télécharger les images depuis HuggingFace

echo "=========================================="
echo "Reconstruction des datasets depuis JSON"
echo "=========================================="

# Vérifier que Python3 est disponible
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé. Installez-le d'abord."
    exit 1
fi

echo ""
echo "1. Création des structures de dossiers..."

for CAPTION_TYPE in blip llava gemini; do
    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"

    # Créer tous les dossiers nécessaires
    mkdir -p "$DATASET_DIR/images"
    mkdir -p "$DATASET_DIR/captions"
    mkdir -p "$DATASET_DIR/partition"

    echo "   ✓ Structure créée pour $CAPTION_TYPE"
done

echo ""
echo "2. Copie des JSON dans partition/..."

for CAPTION_TYPE in blip llava gemini; do
    JSON_FILE="data_info/data_info_${CAPTION_TYPE}.json"
    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"

    if [ ! -f "$JSON_FILE" ]; then
        echo "   ✗ ERREUR: $JSON_FILE non trouvé"
        continue
    fi

    cp "$JSON_FILE" "$DATASET_DIR/partition/data_info.json"
    echo "   ✓ $CAPTION_TYPE: data_info.json copié"
done

echo ""
echo "3. Création des fichiers captions depuis les JSON..."

for CAPTION_TYPE in blip llava gemini; do
    echo "   Processing $CAPTION_TYPE..."

    JSON_FILE="data_info/data_info_${CAPTION_TYPE}.json"
    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"
    CAPTIONS_DIR="$DATASET_DIR/captions"

    if [ ! -f "$JSON_FILE" ]; then
        echo "   ✗ ERREUR: $JSON_FILE non trouvé"
        continue
    fi

    # Créer les captions
    python3 << EOF
import json

with open('$JSON_FILE', 'r') as f:
    data = json.load(f)

count = 0
for item in data:
    # Extraire le numéro de l'image depuis le path
    # Ex: "images/0.png" -> "0"
    image_num = item['path'].split('/')[-1].split('.')[0]

    # Créer le fichier caption
    caption_file = f'$CAPTIONS_DIR/{image_num}.txt'
    with open(caption_file, 'w') as cf:
        cf.write(item['prompt'])
    count += 1

print(f'   ✓ {count} captions créées')
EOF

done

echo ""
echo "=========================================="
echo "4. Vérification des structures créées"
echo "=========================================="

for CAPTION_TYPE in blip llava gemini; do
    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"

    NUM_CAPTIONS=$(ls "$DATASET_DIR/captions"/*.txt 2>/dev/null | wc -l)
    HAS_JSON="❌"
    [ -f "$DATASET_DIR/partition/data_info.json" ] && HAS_JSON="✓"

    echo ""
    echo "Dataset: $CAPTION_TYPE"
    echo "  Captions: $NUM_CAPTIONS fichiers"
    echo "  JSON:     $HAS_JSON"
    echo "  Images:   (dossier vide - à remplir manuellement)"
done

echo ""
echo "=========================================="
echo "Structure des datasets :"
echo "=========================================="
tree -L 3 datasets/ 2>/dev/null || find datasets/ -type d | sed 's|[^/]*/| |g'

echo ""
echo "=========================================="
echo "⚠️  IMPORTANT: IMAGES MANQUANTES"
echo "=========================================="
echo ""
echo "Les captions et JSON sont prêts, mais les IMAGES sont manquantes."
echo ""
echo "Options pour obtenir les images :"
echo ""
echo "1. Si vous avez déjà les images quelque part :"
echo "   cp -r CHEMIN_VERS_IMAGES/* datasets/oxford_flowers102_llava/images/"
echo "   cp -r CHEMIN_VERS_IMAGES/* datasets/oxford_flowers102_blip/images/"
echo "   cp -r CHEMIN_VERS_IMAGES/* datasets/oxford_flowers102_gemini/images/"
echo ""
echo "2. Télécharger depuis HuggingFace (utilise setup_complete_datasets.sh)"
echo ""
echo "Une fois les images ajoutées, transférez vers Mésonet :"
echo "   rsync -avz datasets/ juvollet@juliet1.insa-rouen.fr:~/dla_projet/dla/datasets/"
echo ""
echo "=========================================="
