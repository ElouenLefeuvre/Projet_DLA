"""
Script de fine-tuning PixArt avec captions LLaVA/BLIP/Gemini.
Basé sur le notebook dataset_gen_LAVA.ipynb

Usage:
    python train_pixart.py --data_root ./datasets/oxford_flowers102_llava --output_dir ./outputs/pixart_llava
"""

import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import PixArtAlphaPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm
import wandb
from accelerate import Accelerator


class FlowerDataset(Dataset):
    """
    Dataset personnalisé pour le fine-tuning de PixArt.
    Charge les images et captions depuis le format généré par dataset_gen_LAVA.ipynb
    """

    def __init__(self, data_root, partition_file="partition/train.json", image_size=512):
        """
        Args:
            data_root: chemin racine du dataset (ex: ./datasets/oxford_flowers102_llava)
            partition_file: fichier JSON contenant les infos (default: partition/train.json)
            image_size: taille des images pour le training
        """
        self.data_root = Path(data_root)
        self.image_size = image_size

        # Charger le fichier partition
        partition_path = self.data_root / partition_file
        with open(partition_path, 'r') as f:
            self.data_info = json.load(f)

        print(f"Chargé {len(self.data_info)} exemples depuis {partition_path}")

        # Transforms pour les images
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalisation pour diffusion models
        ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info[idx]

        # Charger l'image
        image_path = self.data_root / item["path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Charger la caption (prompt)
        caption = item["prompt"]

        return {
            "image": image,
            "caption": caption
        }


def collate_fn(batch):
    """Fonction de collate personnalisée pour le DataLoader"""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {"images": images, "captions": captions}


def train_pixart(args):
    """
    Fonction principale de training.
    """
    # Initialiser Accelerator pour multi-GPU / mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
    )

    # Initialiser wandb si activé
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )

    # Charger le modèle PixArt pré-entraîné
    print("Chargement du modèle PixArt...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )

    # Extraire les composants nécessaires
    unet = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Geler VAE et text encoder (on ne fine-tune que l'UNet)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Activer gradient checkpointing pour économiser la VRAM
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Créer le dataset et dataloader
    print(f"Chargement du dataset depuis {args.data_root}...")
    dataset = FlowerDataset(
        data_root=args.data_root,
        partition_file=args.partition_file,
        image_size=args.resolution
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Préparer avec Accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # Déplacer vae et text_encoder sur le device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Training loop
    print("Démarrage du training...")
    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_main_process,
        desc="Training"
    )

    for epoch in range(args.num_train_epochs):
        unet.train()

        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Encoder les images avec VAE
                latents = vae.encode(batch["images"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encoder les captions
                text_inputs = tokenizer(
                    batch["captions"],
                    padding="max_length",
                    max_length=args.max_sequence_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(accelerator.device)

                with torch.no_grad():
                    text_embeddings = text_encoder(
                        text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask
                    )[0]

                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample

                # Compute loss
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

                # Backprop
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Log et sauvegarde
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                # Sauvegarder checkpoint
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Checkpoint sauvegardé: {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Sauvegarder le modèle final
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipe.transformer = unet
        pipe.save_pretrained(os.path.join(args.output_dir, "final_model"))
        print(f"Modèle final sauvegardé: {args.output_dir}/final_model")

    accelerator.end_training()

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PixArt avec captions custom")

    # Dataset arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Chemin racine du dataset (ex: ./datasets/oxford_flowers102_llava)")
    parser.add_argument("--partition_file", type=str, default="partition/train.json",
                        help="Fichier JSON contenant les données d'entraînement")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS",
                        help="Nom du modèle PixArt pré-entraîné")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Résolution des images (default: 512)")
    parser.add_argument("--max_sequence_length", type=int, default=120,
                        help="Longueur max des séquences de texte")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Répertoire de sortie pour les checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="Nombre d'époques d'entraînement")
    parser.add_argument("--max_train_steps", type=int, default=10000,
                        help="Nombre max de steps d'entraînement")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size pour le training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Nombre de steps d'accumulation de gradient")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Activer gradient checkpointing (économise VRAM)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Nombre de workers pour le DataLoader")

    # Optimizer arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                        choices=["linear", "cosine", "constant", "constant_with_warmup"],
                        help="Type de scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Nombre de steps de warmup")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Fréquence de logging")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Fréquence de sauvegarde des checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Utiliser Weights & Biases pour le logging")
    parser.add_argument("--wandb_project", type=str, default="pixart-flowers",
                        help="Nom du projet W&B")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Nom du run W&B")

    args = parser.parse_args()

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Générer un nom de run si non spécifié
    if args.run_name is None:
        dataset_name = Path(args.data_root).name
        args.run_name = f"{dataset_name}_{args.model_name.split('/')[-1]}"

    # Lancer le training
    train_pixart(args)


if __name__ == "__main__":
    main()
