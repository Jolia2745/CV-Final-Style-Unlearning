import os
import csv
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def calculate_clip_scores(image_path, text_prompts):
    """
    Calculate CLIP similarity scores between an image and multiple text prompts
    
    Args:
        image_path (str): Path to the image
        text_prompts (list): List of text prompts to compare against
    
    Returns:
        list: List of similarity scores for each prompt
    """
    image = Image.open(image_path).convert("RGB")
    
    # Process image
    image_inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    # Process text prompts
    text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Get image features
        image_features = clip_model.get_image_features(**image_inputs)
        # Get text features
        text_features = clip_model.get_text_features(**text_inputs)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        
    return similarities.tolist()

def process_style_directory(style):
    """
    Process all images in a style directory and calculate preservation and effectiveness scores
    
    Args:
        style (str): Either 'anime' or 'realistic'
    """
    base_dir = "/scratch/lx2154/w2w/weights2weights/editing/images_ani-real_clip"
    results = []
    
    # Create eval_results directory if it doesn't exist
    os.makedirs(f"eval_results/{style}", exist_ok=True)
    
    # Set style prompt based on directory
    style_prompt = style
    if style_prompt == "realistic": 
        print(1)
        style_prompt = "A realistic portrait featuring lifelike proportions, intricate details in the facial features, and subtle textures in the skin and hair. The lighting highlights natural contours, with soft shadows adding depth and dimension. The eyes are expressive and detailed, capturing a sense of emotion, while the overall composition emphasizes realism through accurate anatomy and true-to-life shading."
    
    else:
        print(2)
        style_prompt = "An anime-style portrait featuring clean, expressive linework, vibrant colors, and soft shading. The character has large eyes with sparkling highlights, a small, delicate nose, and a slightly stylized mouth. The overall style is sleek, polished, and characteristic of anime art."
    
    
    # Iterate through each seed directory
    for seed in os.listdir(base_dir):
        seed_dir = os.path.join(base_dir, seed)
        if not os.path.isdir(seed_dir):
            continue
        
        # Calculate scores for each image
        for image_name in sorted(os.listdir(seed_dir)):
            if not image_name.endswith('.png'):
                continue
                
            image_path = os.path.join(seed_dir, image_name)  
            # Calculate both preservation and effectiveness scores
            scores = calculate_clip_scores(
                image_path, 
                text_prompts=["a girl", style_prompt]
            )
            
            # Store results
            results.append({
                'image_path': image_path,
                'preservation_score': scores[0],  # Score for "a girl"
                'effectiveness_score': scores[1]  # Score for style prompt
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = f"eval_results/{style}/all_clip.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

# Process both styles
for style in ['anime', 'realistic']:
    print(f"Processing {style} images...")
    process_style_directory(style)
    print(f"Finished processing {style} images\n")