# Configuration pour fine-tuning PixArt avec captions BLIP (Modèle A)
# Basé sur PixArt_xl2_img512_internal_for_pokemon_sample_training.py

import os

_base_ = ['../../PixArt-alpha/configs/PixArt_xl2_internal.py']

# Dataset configuration
# Chemin relatif au fichier de config
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
dataset_name = 'oxford_flowers102_blip'
image_list_json = ['data_info.json']

data = dict(
    type='InternalData',
    root=f'{data_root}/datasets/{dataset_name}',
    image_list_json=image_list_json,
    transform='default_train',
    load_vae_feat=True  # Utilise les features pré-extraites
)

image_size = 512

# Model settings
window_block_indexes = []
window_size = 0
use_rel_pos = False
model = 'PixArt_XL_2'
fp32_attention = True

# Chemins des modèles pré-entraînés
load_from = f"{data_root}/PixArt-alpha/output/pretrained_models/PixArt-XL-2-512x512.pth"
vae_pretrained = f"{data_root}/PixArt-alpha/output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 1.0

# Training settings
use_fsdp = False  # FSDP si multi-GPU
num_workers = 8
train_batch_size = 16  # Adapter selon votre GPU
num_epochs = 200
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-5,
    weight_decay=3e-2,
    eps=1e-10
)

lr_schedule_args = dict(num_warmup_steps=500)

# Logging and saving
eval_sampling_steps = 200
log_interval = 20
save_model_steps = 500

# Output directory
work_dir = f'{data_root}/outputs/pixart_{dataset_name}'
