#!/bin/bash
# Script pour détecter la configuration du cluster et adapter les scripts

echo "=========================================="
echo "Détection de la configuration du cluster"
echo "=========================================="

echo ""
echo "1. Partitions disponibles :"
sinfo -o "%20P %5a %.10l %16F"

echo ""
echo "2. Partitions avec GPUs :"
sinfo -o "%P %G" | grep -i gpu || sinfo -o "%P %G" | grep -v "^PARTITION"

echo ""
echo "=========================================="
echo "Configuration détectée"
echo "=========================================="

# Détecter une partition GPU
GPU_PARTITION=$(sinfo -o "%P %G" | grep -i gpu | head -1 | awk '{print $1}' | sed 's/\*//')

if [ -z "$GPU_PARTITION" ]; then
    # Fallback : chercher toutes les partitions
    echo "❌ Aucune partition avec 'gpu' dans le nom trouvée"
    echo ""
    echo "Partitions disponibles :"
    sinfo -o "%P" | tail -n +2
    echo ""
    echo "⚠️  ACTION REQUISE :"
    echo "   1. Regardez la liste ci-dessus"
    echo "   2. Identifiez la partition GPU (demandez à l'admin si besoin)"
    echo "   3. Modifiez manuellement les scripts :"
    echo "      - finetune/extract_features.sh ligne 5"
    echo "      - finetune/train_slurm_official.sh ligne 5"
    echo ""
    echo "   Remplacez : #SBATCH --partition=gpu"
    echo "   Par :       #SBATCH --partition=VOTRE_PARTITION"
else
    echo "✓ Partition GPU détectée : $GPU_PARTITION"
    echo ""
    echo "Application de la configuration..."

    # Mettre à jour extract_features.sh
    sed -i "s/#SBATCH --partition=gpu/#SBATCH --partition=$GPU_PARTITION/" extract_features.sh
    echo "✓ extract_features.sh mis à jour"

    # Mettre à jour train_slurm_official.sh
    sed -i "s/#SBATCH --partition=gpu/#SBATCH --partition=$GPU_PARTITION/" train_slurm_official.sh
    echo "✓ train_slurm_official.sh mis à jour"

    echo ""
    echo "=========================================="
    echo "Configuration appliquée avec succès !"
    echo "Partition GPU : $GPU_PARTITION"
    echo "=========================================="
fi

echo ""
echo "3. Modules disponibles :"
module avail 2>&1 | grep -i -E "(cuda|python|anaconda|miniconda)" | head -10

echo ""
echo "=========================================="
echo "Prochaines étapes :"
echo "=========================================="
echo "1. Charger les modules nécessaires (exemple) :"
echo "   module load cuda/11.8"
echo "   module load anaconda3"
echo ""
echo "2. Lancer setup_pixart.sh :"
echo "   bash setup_pixart.sh"
echo ""
echo "3. Vérifier que les scripts ont la bonne partition :"
echo "   grep 'partition' extract_features.sh"
echo "   grep 'partition' train_slurm_official.sh"
echo "=========================================="
