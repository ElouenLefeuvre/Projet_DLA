#!/usr/bin/env python3
"""
Script de Membership Inference Attack (MIA) basé sur la Loss.
Compare la Loss de reconstruction entre les données d'entraînement (Members) et de test (Non-Members).
"""

import os
import json
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from sklearn.metrics import roc_curve, auc

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset_samples(dataset_dir, num_samples=100):
    """Charge des échantillons aléatoires (image_path, prompt) depuis un dataset."""
    json_path = os.path.join(dataset_dir, "partition", "data_info.json")
    if not os.path.exists(json_path):
        # Fallback pour certains datasets qui n'ont pas le dossier partition
        json_path = os.path.join(dataset_dir, "data_info.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON non trouvé: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Sélectionner aléatoirement
    if len(data) > num_samples:
        samples = random.sample(data, num_samples)
    else:
        samples = data
        
    results = []
    for item in samples:
        img_path = os.path.join(dataset_dir, item['path'])
        if os.path.exists(img_path):
            results.append({
                "path": img_path,
                "prompt": item['prompt']
            })
            
    return results

def compute_loss(pipeline, image, prompt, generator=None):
    """Calcule la loss de diffusion pour une image et un prompt donnés."""
    # 1. Préparer l'image (VAE encoding)
    # Resize et Normalize comme pendant le training
    img_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    pixel_values = img_transform(image).unsqueeze(0).to(device).to(dtype=torch.float16)
    
    with torch.no_grad():
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample(generator=generator)
        latents = latents * pipeline.vae.config.scaling_factor
        
    # 2. Préparer le prompt (T5 encoding)
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, _, _ = pipeline.encode_prompt(
            prompt, do_classifier_free_guidance=False
        )
        
    # 3. Ajouter du bruit
    noise = torch.randn(latents.shape, device=device, generator=generator).to(dtype=torch.float16)
    
    # On choisit un timestep t aléatoire (ou fixe pour être déterministe)
    # Pour MIA, souvent les timesteps intermédiaires (ex: 100-300 sur 1000) révèlent le plus d'overfitting
    # Ici on prend un t random pour simuler la loss d'entraînement
    bsz = latents.shape[0]
    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    
    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps).to(dtype=torch.float16)
    
    # 4. Prédire le bruit/velocity avec le Transformer
    with torch.no_grad():
        model_pred = pipeline.transformer(
            noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
        ).sample
        
    # 5. Calculer la Loss (MSE)
    # PixArt utilise généralement 'sigma_squared' weighting ou predict epsilon. 
    # On compare la prediction avec la target (noise ou velocity selon la config)
    target = noise
    if pipeline.scheduler.config.prediction_type == "v_prediction":
        target = pipeline.scheduler.get_velocity(latents, noise, timesteps)
        
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Membership Inference Attack (MIA) evaluation")
    parser.add_argument("--model_type", type=str, default="blip", choices=["blip", "llava", "gemini"])
    parser.add_argument("--num_samples", type=int, default=100, help="Nombre d'images Train et Test à évaluer")
    args = parser.parse_args()
    
    # Chemins
    model_path = f"../outputs/pixart_oxford_flowers102_{args.model_type}/diffusers"
    train_dir = f"../datasets/oxford_flowers102_{args.model_type}"
    test_dir = f"../datasets/oxford_flowers102_{args.model_type}_test"
    
    print(f"==================================================")
    print(f"MIA Evaluation for Model: {args.model_type}")
    print(f"Model Path: {model_path}")
    print(f"Train Dir (Members): {train_dir}")
    print(f"Test Dir (Non-Members): {test_dir}")
    print(f"Samples: {args.num_samples} per group")
    print(f"==================================================")
    
    # 1. Charger le pipeline
    print("Chargement du modèle...")
    if os.path.exists(os.path.join(model_path, "transformer")):
        transformer = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16)
    else:
        # Fallback si structure différente
        transformer = Transformer2DModel.from_pretrained(model_path, torch_dtype=torch.float16)
        
    pipeline = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        transformer=transformer,
        torch_dtype=torch.float16
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # 2. Charger les données
    print("Chargement des échantillons...")
    train_samples = load_dataset_samples(train_dir, args.num_samples)
    test_samples = load_dataset_samples(test_dir, args.num_samples)
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    # 3. Calculer les losses
    train_losses = []
    test_losses = []
    
    generator = torch.Generator(device=device).manual_seed(42)
    
    print("Calcul des losses (Members)...")
    for s in tqdm(train_samples):
        img = Image.open(s['path']).convert('RGB')
        loss = compute_loss(pipeline, img, s['prompt'], generator)
        train_losses.append(loss)
        
    print("Calcul des losses (Non-Members)...")
    for s in tqdm(test_samples):
        img = Image.open(s['path']).convert('RGB')
        loss = compute_loss(pipeline, img, s['prompt'], generator)
        test_losses.append(loss)
        
    # 4. Analyse et Métriques
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    
    mean_train = np.mean(train_losses)
    mean_test = np.mean(test_losses)
    
    print("\n---------------- Résultats ----------------")
    print(f"Mean Loss (Members/Train):     {mean_train:.5f}")
    print(f"Mean Loss (Non-Members/Test):  {mean_test:.5f}")
    
    # Calcul AUC (On suppose que Loss(Member) < Loss(Non-Member))
    # On utilise -Loss comme score car ROC attend que le score soit plus élevé pour la classe positive (Members)
    y_true = np.concatenate([np.ones(len(train_losses)), np.zeros(len(test_losses))])
    y_scores = np.concatenate([-train_losses, -test_losses])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"MIA Attack AUC-ROC:            {roc_auc:.4f}")
    print("-------------------------------------------")
    
    if roc_auc > 0.55:
        print("⚠️  INDICE DE MÉMORISATION DÉTECTÉ (AUC > 0.55)")
    else:
        print("✅  Peu ou pas de mémorisation détectée (AUC proche de 0.50)")
        
    # 5. Sauvegarder Histogramme
    plt.figure(figsize=(10, 6))
    plt.hist(train_losses, bins=30, alpha=0.5, label='Members (Train)', density=True, color='blue')
    plt.hist(test_losses, bins=30, alpha=0.5, label='Non-Members (Test)', density=True, color='red')
    plt.xlabel('Reconstruction Loss (MSE)')
    plt.ylabel('Density')
    plt.title(f'Distribution des Losses [{args.model_type}] (AUC={roc_auc:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_png = f"mia_histogram_{args.model_type}.png"
    plt.savefig(output_png)
    print(f"\nHistogramme sauvegardé: {output_png}")

if __name__ == "__main__":
    main()
