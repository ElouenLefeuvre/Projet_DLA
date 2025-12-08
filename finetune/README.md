# Fine-tuning PixArt avec captions LLaVA/BLIP/Gemini

Ce dossier contient le script de fine-tuning pour PixArt utilisant les captions générées par LLaVA, BLIP ou Gemini.

## Installation

```bash
pip install -r requirements.txt
```

## Structure du dataset attendue

Le script attend un dataset au format généré par `dataset_gen_LAVA.ipynb`:

```
./datasets/oxford_flowers102_llava/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── captions/
│   ├── 0.txt
│   ├── 1.txt
│   └── ...
└── partition/
    ├── train.json
    ├── data_info.json
    └── ...
```

### Format du fichier `train.json`

```json
[
  {
    "height": 500,
    "width": 500,
    "ratio": 1.0,
    "path": "images/0.png",
    "prompt": "The image features a beautiful white flower...",
    "split": "train"
  },
  ...
]
```

## Utilisation

### Training basique

```bash
python train_pixart.py \
  --data_root ./datasets/oxford_flowers102_llava \
  --output_dir ./outputs/pixart_llava \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --gradient_checkpointing
```

### Training avec Weights & Biases

```bash
python train_pixart.py \
  --data_root ./datasets/oxford_flowers102_llava \
  --output_dir ./outputs/pixart_llava \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --use_wandb \
  --wandb_project "pixart-flowers-experiment" \
  --run_name "llava_captions"
```

### Training des 3 modèles (pour l'expérience complète)

#### Modèle A - BLIP (captions basiques)
```bash
python train_pixart.py \
  --data_root ./datasets/oxford_flowers102_blip \
  --output_dir ./outputs/pixart_blip \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --use_wandb \
  --run_name "model_a_blip"
```

#### Modèle B - LLaVA (captions riches)
```bash
python train_pixart.py \
  --data_root ./datasets/oxford_flowers102_llava \
  --output_dir ./outputs/pixart_llava \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --use_wandb \
  --run_name "model_b_llava"
```

#### Modèle C - Gemini (captions très riches)
```bash
python train_pixart.py \
  --data_root ./datasets/oxford_flowers102_gemini \
  --output_dir ./outputs/pixart_gemini \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --use_wandb \
  --run_name "model_c_gemini"
```

## Paramètres importants

| Paramètre | Description | Défaut |
|-----------|-------------|---------|
| `--data_root` | Chemin racine du dataset | Required |
| `--output_dir` | Répertoire de sortie | Required |
| `--train_batch_size` | Batch size | 4 |
| `--gradient_accumulation_steps` | Steps d'accumulation | 4 |
| `--max_train_steps` | Nombre max de steps | 10000 |
| `--learning_rate` | Learning rate | 1e-5 |
| `--resolution` | Résolution des images | 512 |
| `--gradient_checkpointing` | Économise VRAM | False |
| `--mixed_precision` | fp16/bf16/no | fp16 |
| `--save_steps` | Fréquence de sauvegarde | 500 |

## Optimisation VRAM

Si vous manquez de VRAM, essayez:

1. **Réduire le batch size**:
   ```bash
   --train_batch_size 1 --gradient_accumulation_steps 16
   ```

2. **Activer gradient checkpointing**:
   ```bash
   --gradient_checkpointing
   ```

3. **Réduire la résolution**:
   ```bash
   --resolution 256
   ```

4. **Utiliser bf16 au lieu de fp16** (si GPU Ampere ou plus récent):
   ```bash
   --mixed_precision bf16
   ```

## Sortie

Le script sauvegarde:
- Checkpoints intermédiaires: `{output_dir}/checkpoint-{step}/`
- Modèle final: `{output_dir}/final_model/`

## Durée estimée

- ~6h pour 10,000 steps sur une RTX 3090
- ~3h pour 10,000 steps sur une A100

## Étapes suivantes

Après le fine-tuning, utilisez les scripts de génération et d'évaluation FID:
1. `generate_images.py` - Générer des images avec chaque modèle
2. `evaluate_fid.py` - Calculer les scores FID et comparer
