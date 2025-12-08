#!/bin/bash
# Script pour configurer l'environnement PixArt-alpha

echo "=========================================="
echo "Configuration de l'environnement PixArt"
echo "=========================================="

# Aller dans le répertoire du projet
cd "$(dirname "$0")/.."

# Cloner le repo officiel PixArt-alpha si pas déjà fait
if [ ! -d "PixArt-alpha" ]; then
    echo "Clonage du repo PixArt-alpha..."
    git clone https://github.com/PixArt-alpha/PixArt-alpha.git
    echo "✓ Repo cloné"
else
    echo "✓ Repo PixArt-alpha déjà présent"
fi

cd PixArt-alpha

# Créer l'environnement conda si pas déjà fait
if ! conda env list | grep -q "diffusion"; then
    echo ""
    echo "Création de l'environnement conda 'diffusion'..."
    conda create -n diffusion python=3.10 -y
    echo "✓ Environnement créé"
else
    echo "✓ Environnement 'diffusion' déjà présent"
fi

# Activer l'environnement
source ~/.bashrc
conda activate diffusion

# Installer les dépendances
echo ""
echo "Installation des dépendances..."
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install wandb
pip install accelerate
echo "✓ Dépendances installées"

# Télécharger le modèle pré-entraîné
echo ""
echo "Téléchargement du modèle PixArt-XL-2-512x512..."
python tools/download.py --model_names "PixArt-XL-2-512x512.pth"
echo "✓ Modèle téléchargé"

# Vérifier que tout est en place
echo ""
echo "=========================================="
echo "Vérification de l'installation"
echo "=========================================="

if [ -f "output/pretrained_models/PixArt-XL-2-512x512.pth" ]; then
    echo "✓ Modèle PixArt présent"
else
    echo "✗ ERREUR: Modèle PixArt non trouvé"
    exit 1
fi

if [ -d "output/pretrained_models/sd-vae-ft-ema" ]; then
    echo "✓ VAE présent"
else
    echo "✗ ERREUR: VAE non trouvé"
    exit 1
fi

echo ""
echo "=========================================="
echo "Installation terminée avec succès!"
echo "=========================================="
echo ""
echo "Prochaines étapes:"
echo "1. Transférer vos datasets vers datasets/"
echo "2. Lancer extract_features.sh pour chaque dataset"
echo "3. Lancer train_slurm_official.sh pour le training"
echo ""
echo "Exemple:"
echo "  sbatch finetune/extract_features.sh llava"
echo "  sbatch finetune/train_slurm_official.sh llava"
echo "=========================================="
