#!/bin/bash
# Script de diagnostic SLURM pour identifier le problème

echo "=========================================="
echo "Diagnostic SLURM - Mésonet INSA Rouen"
echo "=========================================="

echo ""
echo "1. Votre compte et associations :"
sacctmgr show user $USER withassoc format=user,account,partition,qos,maxjobs,maxsubmit -p

echo ""
echo "2. Limites QOS :"
sacctmgr show qos format=name,maxjobspu,maxsubmitpu,maxwall -p

echo ""
echo "3. Comptes disponibles :"
sacctmgr show assoc where user=$USER format=account,partition,qos -p

echo ""
echo "4. Test de soumission simplifié :"
echo "#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --partition=mesonet
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

hostname
nvidia-smi
" > /tmp/test_job.sh

echo "Tentative de soumission d'un job test..."
sbatch /tmp/test_job.sh 2>&1

echo ""
echo "=========================================="
echo "Analyse :"
echo "=========================================="

# Vérifier si un compte est nécessaire
ACCOUNTS=$(sacctmgr show assoc where user=$USER format=account -n -p)
NUM_ACCOUNTS=$(echo "$ACCOUNTS" | wc -l)

if [ $NUM_ACCOUNTS -gt 1 ]; then
    echo "⚠️  Vous avez plusieurs comptes associés :"
    echo "$ACCOUNTS"
    echo ""
    echo "SOLUTION : Spécifier un compte avec --account"
    echo "Exemple : #SBATCH --account=NOM_DU_COMPTE"
elif [ $NUM_ACCOUNTS -eq 1 ]; then
    echo "✓ Vous avez 1 compte : $ACCOUNTS"
    echo "Essayez d'ajouter explicitement :"
    echo "#SBATCH --account=$ACCOUNTS"
else
    echo "❌ Aucun compte trouvé. Contactez l'administrateur."
fi

echo ""
echo "=========================================="
echo "Actions recommandées :"
echo "=========================================="
echo "1. Vérifier le résultat ci-dessus"
echo "2. Si un compte est listé, l'ajouter aux scripts"
echo "3. Contacter support-mesonet@insa-rouen.fr si le problème persiste"
echo "=========================================="
