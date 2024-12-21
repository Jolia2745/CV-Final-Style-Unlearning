import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torchvision import transforms
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

def process_evaluation(type_name, meta_csv_path, metric='mse'):
    """
    Process image evaluation with specified metric
    """
    # Create output directory
    output_dir = f"eval_results/{type_name}/{metric}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read meta csv
    print(f"Reading {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)
    
    # Prepare list for results
    mse_results = []
    
    # Group by prompt_style_index and lora_index
    df['prompt_style_index'] = df['img_name'].apply(lambda x: x.split('_')[0])
    df['lora_index'] = df['img_name'].apply(lambda x: x.split('_')[1])
    
    # Choose evaluation function based on metric
    eval_func = calculate_mse if metric == 'mse' else calculate_fid
    
    for (prompt_idx, lora_idx), group in tqdm(df.groupby(['prompt_style_index', 'lora_index'])):
        # Find base image in the group
        base_imgs = group[group['img_name'].str.contains('_base_')]
        if len(base_imgs) == 0:
            print(f"Warning: No base image found for prompt_idx={prompt_idx}, lora_idx={lora_idx}")
            continue
        base_img = base_imgs.iloc[0]
        
        # Add base model self-comparison (should be 0)
        mse_results.append({
            'img_name': base_img['img_name'],
            metric: 0.0  # Base image compared with itself should have metric=0
        })
        
        # Process all non-base images
        for _, row in group[~group['img_name'].str.contains('_base_')].iterrows():
            try:
                # Calculate metric
                score = eval_func(row['path'], base_img['path'])
                mse_results.append({
                    'img_name': row['img_name'],
                    metric: score
                })
            except Exception as e:
                print(f"Error processing image: {row['path']}")
                print(f"Error message: {str(e)}")
                continue
    
    if not mse_results:
        print(f"Warning: No {metric.upper()} results calculated")
        return None, None
    
    # Create and save all_{type}.csv
    all_df = pd.DataFrame(mse_results)
    all_df.to_csv(f"{output_dir}/all_{type_name}.csv", index=False)
    print(f"Saved all {metric.upper()} results to {output_dir}/all_{type_name}.csv")
    
    # Calculate average scores
    avg_results = []
    # Add temporary columns for calculation
    temp_df = all_df.copy()
    temp_df['lora_index'] = temp_df['img_name'].apply(lambda x: x.split('_')[1])
    temp_df['lora_sign'] = temp_df['img_name'].apply(lambda x: x.split('_')[2])
    
    # Calculate average for each model
    for model in temp_df['lora_index'].unique():
        model_data = temp_df[temp_df['lora_index'] == model]
        
        # Calculate and store neg average
        neg_data = model_data[model_data['lora_sign'] == 'neg']
        if not neg_data.empty:
            neg_avg = neg_data[metric].mean()
            avg_results.append({
                'img_name': f"{model}_neg_avg",
                metric: neg_avg
            })
        
        # Calculate and store pos average
        pos_data = model_data[model_data['lora_sign'] == 'pos']
        if not pos_data.empty:
            pos_avg = pos_data[metric].mean()
            avg_results.append({
                'img_name': f"{model}_pos_avg",
                metric: pos_avg
            })
        
        # Add base score
        base_data = model_data[model_data['lora_sign'] == 'base']
        if not base_data.empty:
            avg_results.append({
                'img_name': f"{model}_base_avg",
                metric: base_data[metric].iloc[0]
            })
    
    # Create and save avg_{type}.csv
    avg_df = pd.DataFrame(avg_results)
    avg_df.to_csv(f"{output_dir}/avg_{type_name}.csv", index=False)
    print(f"Saved average {metric.upper()} results to {output_dir}/avg_{type_name}.csv")
    
    # Count models where neg > pos
    higher_neg_count = 0
    total_models = len(temp_df['lora_index'].unique())
    
    for model in temp_df['lora_index'].unique():
        neg_data = avg_df[avg_df['img_name'] == f"{model}_neg_avg"]
        pos_data = avg_df[avg_df['img_name'] == f"{model}_pos_avg"]
        if not neg_data.empty and not pos_data.empty:
            if neg_data[metric].iloc[0] > pos_data[metric].iloc[0]:
                higher_neg_count += 1
    
    print(f"\nIn {type_name} dataset: {higher_neg_count} out of {total_models} models have higher negative average {metric.upper()} than positive average {metric.upper()}")
    
    return all_df, avg_df

if __name__ == "__main__":
    # Process anime dataset with MSE
    process_evaluation(
        type_name='anime',
        meta_csv_path='meta_anime.csv',
        metric='mse'
    )
    
    # Process anime dataset with FID
    process_evaluation(
        type_name='anime',
        meta_csv_path='meta_anime.csv',
        metric='fid'
    )
    
    # Process real dataset with MSE
    process_evaluation(
        type_name='real',
        meta_csv_path='meta_real.csv',
        metric='mse'
    )
    
    # Process real dataset with FID
    process_evaluation(
        type_name='real',
        meta_csv_path='meta_real.csv',
        metric='fid'
    )