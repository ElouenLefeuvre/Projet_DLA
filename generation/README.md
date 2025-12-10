# Génération d'images avec les modèles fine-tunés

## 📁 Fichiers

- **generate_images_from_models.py** - Script Python de génération
- **generate_one_model_slurm.sh** - Script SLURM pour Mésonet
- **README.md** - Ce fichier

## 🚀 Utilisation sur Mésonet

### 1. Transférer les fichiers

```bash
# Sur votre machine locale
cd /home/jvollet/Bureau/dla_projet/dla
rsync -avz generation/ juvollet@juliet.mesonet.fr:~/dla_projet/dla/generation/
```

### 2. Lancer la génération

**⚠️ IMPORTANT: Vous DEVEZ spécifier les fichiers JSON à utiliser**

```bash
# Sur Mésonet
ssh juvollet@juliet.mesonet.fr
cd ~/dla_projet/dla/generation

# Utiliser 1 seul JSON
sbatch generate_one_model_slurm.sh llava ../datasets/oxford_flowers102_blip/partition/data_info.json

# Utiliser 2 JSON
sbatch generate_one_model_slurm.sh llava \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json

# Utiliser les 3 JSON (blip, llava, gemini)
sbatch generate_one_model_slurm.sh llava \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json \
    ../datasets/oxford_flowers102_gemini/partition/data_info.json

# Lancer les 3 modèles en parallèle avec les 3 JSON chacun
sbatch generate_one_model_slurm.sh blip \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json \
    ../datasets/oxford_flowers102_gemini/partition/data_info.json

sbatch generate_one_model_slurm.sh llava \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json \
    ../datasets/oxford_flowers102_gemini/partition/data_info.json

sbatch generate_one_model_slurm.sh gemini \
    ../datasets/oxford_flowers102_blip/partition/data_info.json \
    ../datasets/oxford_flowers102_llava/partition/data_info.json \
    ../datasets/oxford_flowers102_gemini/partition/data_info.json
```

**Utilisation manuelle (Python direct):**

```bash
# 1 JSON
python generate_images_from_models.py \
    --model_type llava \
    --json_paths ../datasets/oxford_flowers102_blip/partition/data_info.json

# Plusieurs JSON
python generate_images_from_models.py \
    --model_type llava \
    --json_paths \
        ../datasets/oxford_flowers102_blip/partition/data_info.json \
        ../datasets/oxford_flowers102_llava/partition/data_info.json \
        ../datasets/oxford_flowers102_gemini/partition/data_info.json
```

### 3. Suivre l'avancement

```bash
squeue -u $USER
tail -f ../logs/generate_*.out
```

## 📂 Résultat



**Total: 9,180 images**

## ⏱️ Temps estimé

- 1 modèle: ~2-3h
- 3 modèles (parallèle): ~3h
- 3 modèles (séquentiel): ~6-8h
