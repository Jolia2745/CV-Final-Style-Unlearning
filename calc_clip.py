import pandas as pd
import torch
import requests
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModelWithProjection, AutoProcessor, CLIPVisionModelWithProjection
from tqdm import tqdm

def load_clip_models(device):
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return text_model, text_tokenizer, vision_model, vision_processor

def load_images_from_urls(urls):
    loaded_images = []
    for url in urls:
        try:
            response = requests.get(url, stream=True, timeout=5)
            image = Image.open(response.raw)
            # Convert to RGB to ensure compatibility
            image = image.convert('RGB')
            loaded_images.append(image)
        except Exception as e:
            print(f"Error loading image {url}: {e}")
    return loaded_images

def compute_cosine_similarity(embeds1, embeds2):
    """Compute cosine similarity between embeddings"""
    return torch.nn.functional.cosine_similarity(embeds1, embeds2).mean().item()

def classify_anime_style(df_path):
    # Ensure device consistency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CSV
    df = pd.read_csv(df_path)
    
    # Initialize score columns if they don't exist
    if 'image_anime_score' not in df.columns:
        df['image_anime_score'] = 0.0
    if 'tag_anime_score' not in df.columns:
        df['tag_anime_score'] = 0.0
    if 'image_person_score' not in df.columns:
        df['image_person_score'] = 0.0
    if 'tag_person_score' not in df.columns:
        df['tag_person_score'] = 0.0
    if 'image_realistic_score' not in df.columns:
        df['image_realistic_score'] = 0.0
    if 'tag_realistic_score' not in df.columns:
        df['tag_realistic_score'] = 0.0

    
    # Load models to the same device
    text_model, text_tokenizer, vision_model, vision_processor = load_clip_models(device)
    
    # Prepare embeddings on the same device
    anime_style_input = text_tokenizer(["anime style"], return_tensors="pt").to(device)
    anime_style_embed = text_model(**anime_style_input).text_embeds
    
    person_style_input = text_tokenizer(["men, women, human characters, people, person"], return_tensors="pt").to(device)
    person_style_embed = text_model(**person_style_input).text_embeds

    realistic_style_input = text_tokenizer(["realistic style"], return_tensors="pt").to(device)
    realistic_style_embed = text_model(**realistic_style_input).text_embeds

    # Process each row with error handling and periodic saving
    try:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Models"):
            # Skip already processed rows
            if row['tag_realistic_score'] != 0:
                continue
            
            try:
                # Encode tags
                tags_input = "".join(row['tags'])
                
                if tags_input:
                    tags_input_tokens = text_tokenizer([tags_input], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
                    tags_embed = text_model(**tags_input_tokens).text_embeds
                    tag_anime_score = compute_cosine_similarity(tags_embed, anime_style_embed)
                    tag_person_score = compute_cosine_similarity(tags_embed, person_style_embed)
                    tag_realistic_score = compute_cosine_similarity(tags_embed, realistic_style_embed)
                else:
                    tag_anime_score = tag_person_score = tag_realistic_score = 0
                
                # process images
                image_urls = eval(row['imagesUrl']) if isinstance(row['imagesUrl'], str) else row['imagesUrl']
                loaded_images = load_images_from_urls(image_urls)

                if loaded_images:
                    # Batch process images
                    image_input = vision_processor(images=loaded_images, return_tensors="pt").to(device)
                    image_embeds = vision_model(**image_input).image_embeds
                    image_anime_score = compute_cosine_similarity(image_embeds, anime_style_embed.expand_as(image_embeds))
                    image_person_score = compute_cosine_similarity(image_embeds, person_style_embed.expand_as(image_embeds))
                    image_realistic_score = compute_cosine_similarity(image_embeds, realistic_style_embed.expand_as(image_embeds))
                else:
                    image_anime_score = image_person_score = image_realistic_score = 0
                
                # # Update DataFrame with scores
                df.at[index, 'image_anime_score'] = round(image_anime_score, 4)
                df.at[index, 'tag_anime_score'] = round(tag_anime_score, 4)
                df.at[index, 'image_person_score'] = round(image_person_score, 4)
                df.at[index, 'tag_person_score'] = round(tag_person_score, 4)
                df.at[index, 'image_realistic_score'] = round(image_realistic_score, 4)
                df.at[index, 'tag_realistic_score'] = round(tag_realistic_score, 4)
                
                # Save progress every 50 iterations
                if index % 50 == 0:
                    df.to_csv('models_with_scores_checkpoint.csv', index=False)
                
            except Exception as row_error:
                print(f"Error processing row {index}: {row_error}")
                # Continue with next row even if this one fails
                continue
        
        # Final save after complete processing
        df.to_csv('models_with_scores.csv', index=False)
        return df
    
    except Exception as e:
        # Save whatever progress has been made in case of a major error
        print(f"Major error occurred: {e}")
        df.to_csv('models_with_scores_final_checkpoint.csv', index=False)
        raise

# Usage
if __name__ == "__main__":
    classify_anime_style('models_with_scores_checkpoint.csv')