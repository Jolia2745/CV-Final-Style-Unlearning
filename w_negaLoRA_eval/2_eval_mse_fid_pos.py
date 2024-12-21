import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import torch
from pytorch_fid import fid_score
import tempfile
import shutil

def calculate_mse(img1_path, img2_path):
    """
    Calculate MSE between two images
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Cannot read images: \n{img1_path} or \n{img2_path}")
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    mse = np.mean((img1 - img2) ** 2)
    return mse

def calculate_fid(img1_path, img2_path):
    """
    Calculate FID between two images using temporary directories
    """
    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        shutil.copy2(img1_path, os.path.join(dir1, "img1.jpeg"))
        shutil.copy2(img2_path, os.path.join(dir2, "img2.jpeg"))
        
        fid = fid_score.calculate_fid_given_paths(
            [dir1, dir2],
            batch_size=1,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=2048,
            num_workers=0
        )
        return fid

def process_image_name(img_name):
    """
    Process image name to extract information and handle negative scales
    """
    parts = img_name.split('_')
    prompt_idx = parts[0]
    lora_idx = parts[1]
    sign = parts[2]
    raw_scale = float(parts[3].replace('.jpeg', ''))
    
    # Convert scale to negative if it's a neg image and scale doesn't start with '-'
    scale = -raw_scale if sign == 'neg' and not parts[3].startswith('-') else raw_scale
    
    return prompt_idx, lora_idx, sign, scale

def process_evaluation_vs_positive(type_name, meta_csv_path, metric='mse'):
    """
    Process image evaluation comparing all images with positive scale=1.0 image
    """
    # Create output directory
    output_dir = f"eval_results/{type_name}/{metric}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read meta csv
    print(f"Reading {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)
    
    # Prepare list for results
    results = []
    
    # Process image names to extract information
    df['prompt_style_index'], df['lora_index'], df['sign'], df['scale'] = zip(
        *df['img_name'].apply(process_image_name)
    )
    
    # Choose evaluation function based on metric
    eval_func = calculate_mse if metric == 'mse' else calculate_fid
    
    for (prompt_idx, lora_idx), group in tqdm(df.groupby(['prompt_style_index', 'lora_index'])):
        # Find positive 1.0 scale image
        pos_one_imgs = group[(group['sign'] == 'pos') & (group['scale'] == 1.0)]
        if len(pos_one_imgs) == 0:
            print(f"Warning: No positive scale=1.0 image found for prompt_idx={prompt_idx}, lora_idx={lora_idx}")
            continue
        pos_one_img = pos_one_imgs.iloc[0]
        
        # Process all images (including positive 1.0 itself)
        for _, row in group.iterrows():
            try:
                # Calculate metric
                score = eval_func(row['path'], pos_one_img['path'])
                results.append({
                    'img_name': row['img_name'],
                    metric: score
                })
            except Exception as e:
                print(f"Error processing image: {row['path']}")
                print(f"Error message: {str(e)}")
                continue
    
    if not results:
        print(f"Warning: No {metric.upper()} results calculated")
        return None
    
    # Create and save results to pos_{type}.csv
    results_df = pd.DataFrame(results)
    output_path = f"{output_dir}/pos_{type_name}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    return results_df

if __name__ == "__main__":
    # Process anime dataset with MSE
    process_evaluation_vs_positive(
        type_name='anime',
        meta_csv_path='meta_anime.csv',
        metric='mse'
    )
    
    # Process anime dataset with FID
    process_evaluation_vs_positive(
        type_name='anime',
        meta_csv_path='meta_anime.csv',
        metric='fid'
    )
    
    # Process real dataset with MSE
    process_evaluation_vs_positive(
        type_name='real',
        meta_csv_path='meta_real.csv',
        metric='mse'
    )
    
    # Process real dataset with FID
    process_evaluation_vs_positive(
        type_name='real',
        meta_csv_path='meta_real.csv',
        metric='fid'
    )