import os
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def setup_clip():
    """
    Setup CLIP model and processor
    """
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def calculate_clip_scores(image_path, text_prompts, model, processor):
    """
    Calculate CLIP similarity scores between an image and text prompts
    """
    image = Image.open(image_path).convert("RGB")
    
    # Process image and text
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Normalize features
        image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        
    return similarities.tolist()

def process_evaluation(style, meta_csv_path):
    """
    Process image evaluation with CLIP
    """
    # Setup CLIP
    model, processor = setup_clip()
    
    # Create output directory
    output_dir = f"eval_results/{style}/clip"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read meta csv
    print(f"Reading {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)
    

    if style == "real":
        style_prompt = "A realistic portrait featuring lifelike proportions, intricate details in the facial features, and subtle textures in the skin and hair. The lighting highlights natural contours, with soft shadows adding depth and dimension. The eyes are expressive and detailed, capturing a sense of emotion, while the overall composition emphasizes realism through accurate anatomy and true-to-life shading."
    
    else:
        style_prompt = "An anime-style portrait featuring clean, expressive linework, vibrant colors, and soft shading. The character has large eyes with sparkling highlights, a small, delicate nose, and a slightly stylized mouth. The overall style is sleek, polished, and characteristic of anime art."


    # Set text prompts based on style
    text_prompts = ["a girl", style]
    
    # Prepare list for results
    clip_results = []
    
    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Calculate CLIP scores
            scores = calculate_clip_scores(row['path'], text_prompts, model, processor)
            
            clip_results.append({
                'img_name': row['img_name'],
                'preservation_score': scores[0],  # Score for "a girl"
                'effectiveness_score': scores[1]  # Score for style prompt
            })
            
        except Exception as e:
            print(f"Error processing image: {row['path']}")
            print(f"Error message: {str(e)}")
            continue
    
    if not clip_results:
        print(f"Warning: No CLIP results calculated")
        return None
    
    # Create and save all_{style}.csv
    all_df = pd.DataFrame(clip_results)
    output_path = f"{output_dir}/all_{style}.csv"
    all_df.to_csv(output_path, index=False)
    print(f"Saved all CLIP results to {output_path}")
    
    return all_df

if __name__ == "__main__":
    # Process anime dataset
    process_evaluation(
        style='anime',
        meta_csv_path='meta_anime.csv'
    )
    
    # Process realistic dataset
    # process_evaluation(
    #     style='real',
    #     meta_csv_path='meta_real.csv'
    # )