#!/usr/bin/env python3
"""
Script pour vérifier la fidélité des modèles fine-tunés en comparant
les images générées avec les images originales du dataset d'entraînement.

Usage:
    python check_fidelity.py --num_samples 10
"""

import json
import argparse
import os
import random
from pathlib import Path
import torch
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler, Transformer2DModel
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import textwrap

def load_random_samples(dataset_path, num_samples=10):
    """
    Charge des échantillons aléatoires depuis le dataset.
    
    Args:
        dataset_path: Chemin vers le dataset (ex: datasets/oxford_flowers102_blip)
        num_samples: Nombre d'échantillons à charger
        
    Returns:
        list: Liste de dictionnaires avec 'id', 'prompt', 'image_path'
    """
    # Chercher le fichier data_info.json
    json_path = Path(dataset_path) / "partition" / "data_info.json"
    
    if not json_path.exists():
        json_path = Path(dataset_path) / "data_info.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"data_info.json non trouvé dans {dataset_path}")
    
    # Charger le JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Sélectionner des échantillons aléatoires
    samples = random.sample(data, min(num_samples, len(data)))
    
    # Préparer les données
    result = []
    for item in samples:
        image_id = item['path'].split('/')[-1].split('.')[0]
        image_path = Path(dataset_path) / item['path']
        
        result.append({
            'id': image_id,
            'prompt': item['prompt'],
            'image_path': str(image_path)
        })
    
    return result

def setup_pipeline(model_path, device='cuda'):
    """Configure le pipeline PixArt avec le modèle fine-tuné."""
    
    print(f"Chargement du transformer fine-tuné depuis: {model_path}")
    
    # Charger le Transformer fine-tuné
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
    
    # Charger le pipeline de base avec le transformer remplacé
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

def create_text_image(text, width=512, height=512, font_size=16):
    """
    Crée une image avec du texte centré.
    
    Args:
        text: Texte à afficher
        width: Largeur de l'image
        height: Hauteur de l'image
        font_size: Taille de la police
        
    Returns:
        PIL.Image: Image avec le texte
    """
    # Créer une image blanche
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Essayer de charger une police, sinon utiliser la police par défaut
    try:
        # Essayer plusieurs polices courantes
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Wrapper le texte pour qu'il tienne dans l'image
    margin = 20
    max_width = width - 2 * margin
    
    # Découper le texte en lignes
    lines = []
    for paragraph in text.split('\n'):
        # Estimer le nombre de caractères par ligne
        avg_char_width = font_size * 0.6
        chars_per_line = int(max_width / avg_char_width)
        wrapped = textwrap.wrap(paragraph, width=chars_per_line)
        lines.extend(wrapped)
    
    # Calculer la hauteur totale du texte
    line_height = font_size + 4
    total_text_height = len(lines) * line_height
    
    # Commencer à dessiner depuis le haut avec une marge
    y = margin
    
    for line in lines:
        # Centrer horizontalement chaque ligne
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        
        # Dessiner le texte
        draw.text((x, y), line, fill='black', font=font)
        y += line_height
        
        # Arrêter si on dépasse la hauteur
        if y > height - margin:
            break
    
    return img

def generate_comparison(pipe, sample, output_path, num_inference_steps=20, guidance_scale=4.5):
    """
    Génère une image et la compare avec l'originale.
    
    Args:
        pipe: Pipeline PixArt
        sample: Dict avec 'prompt', 'image_path', 'id'
        output_path: Chemin de sortie pour l'image de comparaison
        num_inference_steps: Nombre de steps d'inférence
        guidance_scale: Guidance scale
    """
    # Générer l'image
    generated = pipe(
        prompt=sample['prompt'],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=512,
        width=512,
    ).images[0]
    
    # Charger l'image originale
    if os.path.exists(sample['image_path']):
        original = Image.open(sample['image_path']).convert('RGB').resize((512, 512))
    else:
        # Si l'image n'existe pas, créer un placeholder noir
        original = Image.new('RGB', (512, 512), color='black')
    
    # Créer une image avec le prompt
    prompt_text = f"ID: {sample['id']}\n\nPrompt:\n{sample['prompt']}"
    prompt_img = create_text_image(prompt_text, width=512, height=512, font_size=14)
    
    # Créer une image de comparaison en 3 colonnes: Original | Prompt | Généré
    comparison = Image.new('RGB', (1536, 512))  # 3 * 512 de largeur
    comparison.paste(original, (0, 0))
    comparison.paste(prompt_img, (512, 0))
    comparison.paste(generated, (1024, 0))
    
    # Ajouter des labels en haut de chaque colonne
    draw = ImageDraw.Draw(comparison)
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        label_font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                label_font = ImageFont.truetype(font_path, 20)
                break
        if label_font is None:
            label_font = ImageFont.load_default()
    except:
        label_font = ImageFont.load_default()
    
    # Dessiner les labels avec fond semi-transparent
    labels = [
        ("ORIGINAL", 256, 10),
        ("PROMPT", 768, 10),
        ("GENERATED", 1280, 10)
    ]
    
    for label_text, x_center, y in labels:
        bbox = draw.textbbox((0, 0), label_text, font=label_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = x_center - text_width // 2
        
        # Fond semi-transparent (rectangle blanc)
        padding = 5
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill='white',
            outline='black',
            width=2
        )
        draw.text((x, y), label_text, fill='black', font=label_font)
    
    # Sauvegarder
    comparison.save(output_path)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Vérifier la fidélité des modèles fine-tunés")
    parser.add_argument('--num_samples', type=int, default=10,
                        help="Nombre d'échantillons à générer par modèle (défaut: 10)")
    parser.add_argument('--output_dir', type=str, default='outputs/fidelity_check',
                        help="Dossier de sortie (défaut: outputs/fidelity_check)")
    parser.add_argument('--num_inference_steps', type=int, default=20,
                        help="Nombre de steps d'inférence (défaut: 20)")
    parser.add_argument('--guidance_scale', type=float, default=4.5,
                        help="Guidance scale (défaut: 4.5)")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device à utiliser (cuda ou cpu)")
    parser.add_argument('--models', type=str, nargs='+', default=['blip', 'llava', 'gemini'],
                        help="Modèles à tester (défaut: blip llava gemini)")
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Vérification de la fidélité des modèles fine-tunés")
    print("="*60)
    print(f"Nombre d'échantillons par modèle: {args.num_samples}")
    print(f"Modèles à tester: {', '.join(args.models)}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    print()
    
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Traitement du modèle: {model_type.upper()}")
        print(f"{'='*60}\n")
        
        # Chemins
        model_path = f"outputs/pixart_oxford_flowers102_{model_type}_diffusers"
        dataset_path = f"datasets/oxford_flowers102_{model_type}"
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vérifier que le modèle existe
        if not os.path.exists(model_path):
            print(f"⚠️  ATTENTION: Modèle non trouvé: {model_path}")
            print(f"   Assurez-vous d'avoir converti le checkpoint avec convert_checkpoint_to_diffusers.sh")
            print(f"   Passage au modèle suivant...\n")
            continue
        
        # Vérifier que le dataset existe
        if not os.path.exists(dataset_path):
            print(f"⚠️  ATTENTION: Dataset non trouvé: {dataset_path}")
            print(f"   Passage au modèle suivant...\n")
            continue
        
        # Charger des échantillons aléatoires
        print(f"1. Chargement de {args.num_samples} échantillons aléatoires...")
        try:
            samples = load_random_samples(dataset_path, args.num_samples)
            print(f"   ✓ {len(samples)} échantillons chargés\n")
        except Exception as e:
            print(f"   ❌ Erreur lors du chargement: {e}")
            continue
        
        # Charger le modèle
        print("2. Chargement du modèle...")
        try:
            pipe = setup_pipeline(model_path, device=args.device)
            print("   ✓ Modèle chargé\n")
        except Exception as e:
            print(f"   ❌ Erreur lors du chargement du modèle: {e}")
            continue
        
        # Générer les comparaisons
        print("3. Génération des images de comparaison...")
        for i, sample in enumerate(tqdm(samples, desc=f"  {model_type}")):
            output_path = model_output_dir / f"comparison_{sample['id']}.jpg"
            
            try:
                generate_comparison(
                    pipe=pipe,
                    sample=sample,
                    output_path=output_path,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale
                )
                
                # Sauvegarder aussi le prompt
                prompt_path = model_output_dir / f"comparison_{sample['id']}.txt"
                with open(prompt_path, 'w') as f:
                    f.write(sample['prompt'])
                    
            except Exception as e:
                print(f"\n   ⚠️  Erreur pour l'échantillon {sample['id']}: {e}")
                continue
        
        print(f"   ✓ {len(samples)} comparaisons générées dans {model_output_dir}\n")
        
        # Libérer la mémoire
        del pipe
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✅ Vérification terminée !")
    print(f"Résultats dans: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
