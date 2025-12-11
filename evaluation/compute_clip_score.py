#!/usr/bin/env python3
"""
Script pour calculer le CLIP Score des images générées.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Configuration
GENERATED_DIR = "../generated_images"
DATASETS_DIR = "../datasets"
MODELS_TO_EVALUATE = ["base", "blip", "llava", "gemini"]
CAPTION_TYPES = ["blip_captions", "llava_captions", "gemini_captions"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_captions(caption_type):
    """
    Charge les captions depuis le bon fichier JSON.
    Ex: blip_captions -> datasets/oxford_flowers102_blip_test/partition/data_info.json
    """
    # Extraire le type (blip, llava, gemini) de "blip_captions"
    dataset_type = caption_type.split('_')[0]
    
    json_path = os.path.join(
        DATASETS_DIR, 
        f"oxford_flowers102_{dataset_type}_test", 
        "partition", 
        "data_info.json"
    )
    
    if not os.path.exists(json_path):
        print(f"⚠️ JSON non trouvé: {json_path}")
        return {}
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Créer un dictionnaire {image_id: prompt}
    captions = {}
    for item in data:
        # Extraire l'ID de l'image (ex: "images/123.jpg" -> "123")
        image_id = os.path.splitext(os.path.basename(item['path']))[0]
        # Nettoyer l'ID s'il contient "images_" (ex: "images_123" -> "123")
        image_id = image_id.replace('images_', '')
        
        captions[image_id] = item['prompt']
        
    return captions

def calculate_clip_score(model, processor, image_paths, prompts, device):
    """Calcule le CLIP score pour un lot d'images et leurs prompts."""
    scores = []
    
    # Traiter par batch pour éviter OOM
    batch_size = 32
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        
        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # Préparer les inputs
            inputs = processor(
                text=batch_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Calculer la similarité (logits_per_image est l'image-text similarity)
            logits_per_image = outputs.logits_per_image
            batch_scores = logits_per_image.diag().cpu().numpy() # Score diagonal (image_i vs text_i)
            scores.extend(batch_scores)
            
        except Exception as e:
            print(f"Erreur lors du calcul CLIP: {e}")
            continue
            
    return scores

def main():
    print(f"Initialisation du modèle CLIP (ViT-L/14) sur {DEVICE}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    results = {}
    
    # Pour chaque modèle (base, blip, llava, gemini)
    for model_name in MODELS_TO_EVALUATE:
        model_results = {}
        all_model_scores = []  # Pour stocker tous les scores de ce modèle
        print(f"\nEvaluating Model: {model_name}")
        
        # Pour chaque type de caption (sous-dossier)
        for caption_type in CAPTION_TYPES:
            captions_dict = load_captions(caption_type)
            if not captions_dict:
                continue
                
            img_dir = os.path.join(GENERATED_DIR, model_name, caption_type)
            if not os.path.exists(img_dir):
                print(f"⚠️ Dossier d'images non trouvé: {img_dir}")
                continue
                
            # Récupérer les images et leurs prompts correspondants
            image_paths = []
            prompts = []
            
            files = os.listdir(img_dir)
            files = [f for f in files if f.endswith(('.png', '.jpg'))]
            
            print(f"  Sous-dossier: {caption_type} ({len(files)} images)")
            
            for f in files:
                image_id = os.path.splitext(f)[0]
                if image_id in captions_dict:
                    image_paths.append(os.path.join(img_dir, f))  
                    prompts.append(captions_dict[image_id])
            
            if not image_paths:
                print("    Aucune image correspondante trouvée.")
                continue
                
            # Calculer CLIP Score
            scores = calculate_clip_score(model, processor, image_paths, prompts, DEVICE)
            
            # Ajouter aux scores globaux du modèle
            all_model_scores.extend(scores)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            model_results[caption_type] = {
                "mean": float(avg_score),
                "std": float(std_score),
                "count": len(scores)
            }
            
            print(f"    CLIP Score: {avg_score:.2f} ± {std_score:.2f}")
            
        # Calculer le score GLOBAL pour ce modèle
        if all_model_scores:
            global_avg = np.mean(all_model_scores)
            global_std = np.std(all_model_scores)
            model_results["GLOBAL"] = {
                "mean": float(global_avg),
                "std": float(global_std),
                "count": len(all_model_scores)
            }
            print(f"  👉 GLOBAL Model Score: {global_avg:.2f} ± {global_std:.2f}")
            
        results[model_name] = model_results
        
    # Sauvegarder les résultats
    output_file = "clip_scores.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nRésultats sauvegardés dans {output_file}")
    
    # Afficher un tableau récapitulatif
    print("\n" + "="*80)
    print(f"{'Model':<10} | {'Caption Type':<20} | {'CLIP Score':<15}")
    print("-" * 80)
    for model_name, res in results.items():
        for cap_type, stats in res.items():
            # Mettre en évidence la ligne GLOBAL
            if cap_type == "GLOBAL":
                print(f"{model_name:<10} | {cap_type:<20} | {stats['mean']:.4f} ± {stats['std']:.4f} ***")
            else:
                print(f"{model_name:<10} | {cap_type:<20} | {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
