#!/usr/bin/env python3
"""
Script pour générer des images avec les 3 modèles fine-tunés (BLIP, LLaVA, Gemini)
à partir des captions des fichiers JSON.

Usage:
    python generate_images_from_models.py --model_type blip
    python generate_images_from_models.py --model_type llava
    python generate_images_from_models.py --model_type gemini
"""

import json
import argparse
import os
from pathlib import Path
import torch
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler, Transformer2DModel
from tqdm import tqdm
from PIL import Image

def load_captions_from_json(json_paths):
    """
    Charge les captions depuis un ou plusieurs fichiers JSON.

    Args:
        json_paths: str ou list[str] - Chemin(s) vers le(s) fichier(s) JSON

    Returns:
        dict: Dictionnaire avec clés = nom du JSON, valeurs = liste des captions
    """
    # Si c'est un seul chemin (string), le convertir en liste
    if isinstance(json_paths, str):
        json_paths = [json_paths]

    all_captions = {}

    for json_path in json_paths:
        # Extraire le nom du fichier sans extension pour l'utiliser comme clé
        json_name = Path(json_path).stem  # Ex: "data_info_blip" -> "data_info_blip"

        # Charger le JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extraire les prompts et les IDs d'images
        captions = []
        for item in data:
            image_id = item['path'].split('/')[-1].split('.')[0]  # Ex: "0" depuis "images/0.png"
            prompt = item['prompt']
            captions.append({
                'id': image_id,
                'prompt': prompt,
                'original_path': item['path']
            })

        all_captions[json_name] = captions

    return all_captions

def setup_pipeline(model_path, device='cuda'):
    """Configure le pipeline PixArt avec le modèle fine-tuné."""
    print(f"Chargement du transformer fine-tuné depuis: {model_path}")

    # 1. Charger le Transformer fine-tuné
    # On gère le cas où model_path contient un sous-dossier 'transformer' ou est directement le dossier transformer
    if os.path.exists(os.path.join(model_path, "transformer")):
        transformer = Transformer2DModel.from_pretrained(
            model_path, 
            subfolder="transformer", 
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    else:
        transformer = Transformer2DModel.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            use_safetensors=True
        )

    print("Chargement du pipeline de base (PixArt-XL-2-512x512)...")
    
    # 2. Charger le pipeline de base avec le transformer remplacé
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        transformer=transformer,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    # Utiliser DPM-Solver pour accélérer la génération
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Déplacer sur GPU
    pipe = pipe.to(device)

    # Activer l'optimisation mémoire
    pipe.enable_attention_slicing()

    return pipe

def generate_images(pipe, captions_dict, output_base_dir, num_inference_steps=20, guidance_scale=4.5):
    """
    Génère les images pour toutes les captions de tous les JSON.

    Args:
        pipe: Pipeline PixArt
        captions_dict: dict - {json_name: [captions]}
        output_base_dir: str - Dossier de base pour les outputs
        num_inference_steps: int
        guidance_scale: float
    """
    output_base_dir = Path(output_base_dir)

    total_images = sum(len(captions) for captions in captions_dict.values())

    print(f"Génération de {total_images} images depuis {len(captions_dict)} JSON...")
    print(f"Dossier de base: {output_base_dir}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print()

    for json_name, captions in captions_dict.items():
        # Créer un sous-dossier pour chaque JSON
        output_dir = output_base_dir / json_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Génération pour '{json_name}': {len(captions)} images")
        print(f"   Dossier: {output_dir}")

        for caption_data in tqdm(captions, desc=f"  {json_name}", leave=False):
            image_id = caption_data['id']
            prompt = caption_data['prompt']

            # Générer l'image
            image = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
            ).images[0]

            # Sauvegarder l'image
            output_path = output_dir / f"{image_id}.png"
            image.save(output_path)

        print(f"   ✓ {len(captions)} images générées\n")

    print(f"✅ Total: {total_images} images générées dans {output_base_dir}")

def main():
    parser = argparse.ArgumentParser(description="Générer des images avec un modèle PixArt fine-tuné")
    parser.add_argument('--model_type', type=str, required=True, choices=['blip', 'llava', 'gemini'],
                        help="Type de modèle à utiliser (blip, llava, ou gemini)")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Chemin vers le modèle fine-tuné (défaut: outputs/pixart_oxford_flowers102_<type>/diffusers)")
    parser.add_argument('--json_paths', type=str, nargs='+', required=True,
                        help="Chemin(s) vers le(s) fichier(s) JSON (plusieurs possibles). Ex: --json_paths file1.json file2.json")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Dossier de sortie (par défaut: generated_images/<model_type>)")
    parser.add_argument('--num_inference_steps', type=int, default=20,
                        help="Nombre de steps d'inférence (défaut: 20)")
    parser.add_argument('--guidance_scale', type=float, default=4.5,
                        help="Guidance scale (défaut: 4.5)")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device à utiliser (cuda ou cpu)")

    args = parser.parse_args()

    # Chemins par défaut
    if args.model_path is None:
        args.model_path = f"outputs/pixart_oxford_flowers102_{args.model_type}/diffusers"

    if args.json_paths is None:
        # Aucun JSON par défaut - l'utilisateur DOIT spécifier
        print("❌ ERREUR: Vous devez spécifier au moins un fichier JSON avec --json_paths")
        print()
        print("Exemples:")
        print("  python generate_images_from_models.py --model_type llava --json_paths datasets/oxford_flowers102_blip/partition/data_info.json")
        print("  python generate_images_from_models.py --model_type llava --json_paths datasets/oxford_flowers102_blip/partition/data_info.json datasets/oxford_flowers102_llava/partition/data_info.json")
        return

    if args.output_dir is None:
        args.output_dir = f"generated_images/{args.model_type}"

    print("="*60)
    print(f"Génération d'images avec le modèle {args.model_type.upper()}")
    print("="*60)
    print(f"Modèle: {args.model_path}")
    print(f"JSON(s): {len(args.json_paths)} fichier(s)")
    for json_path in args.json_paths:
        print(f"  - {json_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    print()

    # Vérifier que le modèle existe
    if not os.path.exists(args.model_path):
        print(f"❌ ERREUR: Modèle non trouvé: {args.model_path}")
        print("Vérifiez que le training a bien terminé et que le modèle est sauvegardé.")
        return

    # Vérifier que les JSON existent
    for json_path in args.json_paths:
        if not os.path.exists(json_path):
            print(f"❌ ERREUR: JSON non trouvé: {json_path}")
            return

    # Charger les captions de tous les JSON
    print("1. Chargement des captions...")
    captions_dict = load_captions_from_json(args.json_paths)
    total_captions = sum(len(captions) for captions in captions_dict.values())
    print(f"   ✓ {total_captions} captions chargées depuis {len(captions_dict)} JSON(s)")
    for json_name, captions in captions_dict.items():
        print(f"     - {json_name}: {len(captions)} captions")
    print()

    # Setup du pipeline
    print("2. Chargement du modèle...")
    pipe = setup_pipeline(args.model_path, device=args.device)
    print("   ✓ Modèle chargé")
    print()

    # Générer les images
    print("3. Génération des images...")
    generate_images(
        pipe=pipe,
        captions_dict=captions_dict,
        output_base_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )

    print()
    print("="*60)
    print("✅ Génération terminée !")
    print("="*60)

if __name__ == "__main__":
    main()
