#!/bin/bash
# Script pour créer les fichiers captions à partir des JSON

echo "=========================================="
echo "Création des captions depuis les JSON"
echo "=========================================="

for CAPTION_TYPE in blip llava gemini; do
    echo ""
    echo "Processing $CAPTION_TYPE..."

    JSON_FILE="data_info/data_info_${CAPTION_TYPE}.json"
    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"
    CAPTIONS_DIR="$DATASET_DIR/captions"

    if [ ! -f "$JSON_FILE" ]; then
        echo "  ✗ ERREUR: $JSON_FILE non trouvé"
        continue
    fi

    # Créer le dossier captions
    mkdir -p "$CAPTIONS_DIR"

    # Extraire et créer les fichiers captions
    echo "  Extraction des captions..."
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

print(f'  ✓ {count} fichiers captions créés dans $CAPTIONS_DIR')
EOF

done

echo ""
echo "=========================================="
echo "Terminé !"
echo "=========================================="
echo ""
echo "Vérification :"
for CAPTION_TYPE in blip llava gemini; do
    NUM_CAPTIONS=$(ls datasets/oxford_flowers102_${CAPTION_TYPE}/captions/*.txt 2>/dev/null | wc -l)
    echo "  $CAPTION_TYPE: $NUM_CAPTIONS captions"
done

echo ""
echo "=========================================="
echo "ATTENTION : Les IMAGES doivent encore être ajoutées !"
echo "=========================================="
echo "Les images doivent être dans :"
echo "  datasets/oxford_flowers102_llava/images/"
echo "  datasets/oxford_flowers102_blip/images/"
echo "  datasets/oxford_flowers102_gemini/images/"
echo ""
echo "Si vous avez les images quelque part, copiez-les :"
echo "  cp -r VOTRE_DOSSIER/images/* datasets/oxford_flowers102_llava/images/"
echo "=========================================="
