#!/bin/bash
# Script pour configurer l'environnement PixArt-alpha

echo "=========================================="
echo "Configuration de l'environnement PixArt"
echo "=========================================="

# Vérifier si conda est installé
if ! command -v conda &> /dev/null; then
    echo "⚠ Conda n'est pas installé. Installation de Miniconda..."
    
    # Télécharger Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    # Installer Miniconda
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    
    # Initialiser conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    # Nettoyer
    rm /tmp/miniconda.sh
    
    echo "✓ Miniconda installé"
    echo "⚠ Veuillez redémarrer votre terminal et relancer ce script"
    exit 0
else
    echo "✓ Conda déjà installé"
    # Initialiser conda pour ce script
    eval "$(conda shell.bash hook)"
fi

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
if ! conda env list | grep -q "pixart"; then
    echo ""
    echo "Création de l'environnement conda 'pixart'..."
    conda create -n pixart python=3.9 -y
    echo "✓ Environnement créé"
else
    echo "✓ Environnement 'pixart' déjà présent"
fi

# Activer l'environnement
source ~/.bashrc
conda activate pixart

# Installer les dépendances
echo ""
echo "Installation des dépendances (PyTorch 2.1.1)..."
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
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
