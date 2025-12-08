#!/bin/bash
# Script complet pour créer les 3 datasets à partir des JSON

echo "=========================================="
echo "Setup complet des datasets Oxford Flowers"
echo "=========================================="

# Vérifier que Python3 est disponible
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé. Installez-le d'abord."
    exit 1
fi

# Installer les dépendances si nécessaire
echo ""
echo "1. Vérification des dépendances Python..."
pip3 install --quiet datasets pillow 2>&1 | grep -v "already satisfied" || echo "  ✓ Dépendances OK"

# Télécharger les images Oxford Flowers depuis HuggingFace
echo ""
echo "2. Téléchargement des images Oxford Flowers depuis HuggingFace..."
echo "   (Cela peut prendre 5-10 minutes)"

python3 << 'EOF'
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

print("   Chargement du dataset...")
dataset = load_dataset("dpdl-benchmark/oxford_flowers102")
train_dataset = dataset["train"]

# Créer un dossier temporaire pour les images
output_dir = Path("temp_flowers_images")
output_dir.mkdir(exist_ok=True)

print(f"   Sauvegarde de {len(train_dataset)} images...")
for idx in tqdm(range(len(train_dataset)), desc="   Images"):
    image = train_dataset[idx]["image"]
    image.save(f"{output_dir}/{idx}.png")

print(f"   ✓ {len(train_dataset)} images sauvegardées dans {output_dir}/")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du téléchargement des images"
    exit 1
fi

# Créer les captions depuis les JSON
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

    # Créer les dossiers
    mkdir -p "$CAPTIONS_DIR"

    # Créer les captions
    python3 << EOF
import json

with open('$JSON_FILE', 'r') as f:
    data = json.load(f)

for item in data:
    image_num = item['path'].split('/')[-1].split('.')[0]
    caption_file = f'$CAPTIONS_DIR/{image_num}.txt'
    with open(caption_file, 'w') as cf:
        cf.write(item['prompt'])
EOF

    NUM_CAPTIONS=$(ls "$CAPTIONS_DIR"/*.txt 2>/dev/null | wc -l)
    echo "   ✓ $NUM_CAPTIONS captions créées"
done

# Copier les images dans les 3 datasets
echo ""
echo "4. Copie des images dans les 3 datasets..."

for CAPTION_TYPE in blip llava gemini; do
    IMAGES_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}/images"
    mkdir -p "$IMAGES_DIR"

    echo "   Copie vers $CAPTION_TYPE..."
    cp -r temp_flowers_images/* "$IMAGES_DIR/"

    NUM_IMAGES=$(ls "$IMAGES_DIR"/*.png 2>/dev/null | wc -l)
    echo "   ✓ $NUM_IMAGES images copiées"
done

# Créer les data_info.json dans partition/
echo ""
echo "5. Création des fichiers partition/data_info.json..."

for CAPTION_TYPE in blip llava gemini; do
    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"
    mkdir -p "$DATASET_DIR/partition"

    cp "data_info/data_info_${CAPTION_TYPE}.json" "$DATASET_DIR/partition/data_info.json"
    echo "   ✓ $CAPTION_TYPE: partition/data_info.json créé"
done

# Nettoyer le dossier temporaire
echo ""
echo "6. Nettoyage..."
rm -rf temp_flowers_images
echo "   ✓ Dossier temporaire supprimé"

# Vérification finale
echo ""
echo "=========================================="
echo "Vérification finale"
echo "=========================================="

for CAPTION_TYPE in blip llava gemini; do
    DATASET_DIR="datasets/oxford_flowers102_${CAPTION_TYPE}"

    NUM_IMAGES=$(ls "$DATASET_DIR/images"/*.png 2>/dev/null | wc -l)
    NUM_CAPTIONS=$(ls "$DATASET_DIR/captions"/*.txt 2>/dev/null | wc -l)
    HAS_JSON="❌"
    [ -f "$DATASET_DIR/partition/data_info.json" ] && HAS_JSON="✓"

    echo ""
    echo "Dataset: $CAPTION_TYPE"
    echo "  Images:   $NUM_IMAGES fichiers"
    echo "  Captions: $NUM_CAPTIONS fichiers"
    echo "  JSON:     $HAS_JSON"
done

echo ""
echo "=========================================="
echo "Structure des datasets :"
echo "=========================================="
tree -L 3 datasets/ 2>/dev/null || find datasets/ -type d | sed 's|[^/]*/| |g'

echo ""
echo "=========================================="
echo "✅ Setup terminé avec succès !"
echo "=========================================="
echo ""
echo "Prochaines étapes :"
echo "1. Transférer vers Mésonet :"
echo "   rsync -avz datasets/ juvollet@juliet1.insa-rouen.fr:~/dla_projet/dla/datasets/"
echo ""
echo "2. Sur Mésonet, lancer l'extraction des features :"
echo "   cd ~/dla_projet/dla/finetune"
echo "   sbatch extract_features.sh blip"
echo "   sbatch extract_features.sh llava"
echo "   sbatch extract_features.sh gemini"
echo ""
echo "=========================================="
