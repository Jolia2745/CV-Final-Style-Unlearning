import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def process_scale(parts):
    """
    Process scale value from image name parts
    Args:
        parts: List of parts from image name split by '_'
    Returns:
        float: Processed scale value
    """
    scale_str = parts[3].replace('.jpeg', '')
    scale = float(scale_str)
    if parts[2] == 'neg' and not scale_str.startswith('-'):
        scale = -scale
    return scale

def create_detail_visualization(type_name, metric, all_df):
    """
    Create line plots from all_{type}.csv data
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Extract components with corrected scale processing
    components = []
    for _, row in all_df.iterrows():
        parts = row['img_name'].split('_')
        scale = process_scale(parts)
            
        components.append({
            'prompt_type': parts[0],
            'lora_index': parts[1],
            'sign': parts[2],
            'scale': scale,
            metric: row[metric]
        })
    
    processed_df = pd.DataFrame(components)
    
    # Group data by prompt_type and lora_index
    unique_groups = processed_df[['prompt_type', 'lora_index']].drop_duplicates()
    
    # Plot each group
    for idx, (_, row) in enumerate(unique_groups.iterrows()):
        prompt = row['prompt_type']
        lora = row['lora_index']
        
        group_data = processed_df[
            (processed_df['prompt_type'] == prompt) & 
            (processed_df['lora_index'] == lora)
        ].sort_values('scale')
        
        ax = axes[idx]
        ax.plot(group_data['scale'], group_data[metric], 
               color='black', marker='o')
        
        ax.set_title(f'Prompt {prompt}, Lora {lora}', fontsize=10)
        ax.set_xlabel('Lora Scale', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        
        # Set x-axis ticks from -1 to 1
        ax.set_xticks(np.arange(-1.0, 1.1, 0.2))
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(unique_groups), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_path = f"eval_results/{type_name}/{metric}/{type_name}_{metric}_detail_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detail visualization to {output_path}")
    plt.close()

def create_pos_visualization(type_name, metric, pos_df):
    """
    Create visualization for comparison with positive scale=1.0
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Extract components with corrected scale processing
    components = []
    for _, row in pos_df.iterrows():
        parts = row['img_name'].split('_')
        scale = process_scale(parts)
            
        components.append({
            'prompt_type': parts[0],
            'lora_index': parts[1],
            'scale': scale,
            metric: row[metric]
        })
    
    processed_df = pd.DataFrame(components)
    
    unique_groups = processed_df[['prompt_type', 'lora_index']].drop_duplicates()
    
    for idx, (_, row) in enumerate(unique_groups.iterrows()):
        prompt = row['prompt_type']
        lora = row['lora_index']
        
        group_data = processed_df[
            (processed_df['prompt_type'] == prompt) & 
            (processed_df['lora_index'] == lora)
        ].sort_values('scale')
        
        ax = axes[idx]
        ax.plot(group_data['scale'], group_data[metric], 
               color='red', marker='o', label='Comparison with Pos 1.0')
        
        ax.set_title(f'Prompt {prompt}, Lora {lora}\nvs Positive Scale 1.0', fontsize=10)
        ax.set_xlabel('Scale', fontsize=10)
        ax.set_ylabel(f'{metric.upper()} vs Pos 1.0', fontsize=10)
        
        # Set x-axis ticks for full range
        ax.set_xticks(np.arange(-1.0, 1.1, 0.2))
        ax.set_xlim(-1.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(unique_groups), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_path = f"eval_results/{type_name}/{metric}/{type_name}_{metric}_pos_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved positive comparison visualization to {output_path}")
    plt.close()

def create_avg_visualization(type_name, metric, avg_df):
    """
    Create visualization for average values with three points: neg, 0, pos
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    avg_components = []
    for _, row in avg_df.iterrows():
        parts = row['img_name'].split('_')
        lora_index = parts[0]
        avg_type = '_'.join(parts[1:])
        
        avg_components.append({
            'lora_index': lora_index,
            'type': avg_type,
            metric: row[metric]
        })
    
    avg_df_processed = pd.DataFrame(avg_components)
    unique_loras = sorted(avg_df_processed['lora_index'].unique())
    
    for idx, lora in enumerate(unique_loras):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        lora_data = avg_df_processed[avg_df_processed['lora_index'] == lora]
        
        x_coords = {'neg_avg': -1, 'base_avg': 0, 'pos_avg': 1}
        x_values = []
        y_values = []
        
        for avg_type in ['neg_avg', 'base_avg', 'pos_avg']:
            if avg_type in lora_data['type'].values:
                x_values.append(x_coords[avg_type])
                y_values.append(lora_data[lora_data['type'] == avg_type][metric].iloc[0])
        
        ax.plot(x_values, y_values, 'black', marker='o')
        ax.scatter(x_values, y_values, c='black', s=100)
        
        ax.set_title(f'Lora {lora}', fontsize=10)
        ax.set_xlabel('Type', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        
        # Set x-axis ticks
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['neg', '0', 'pos'])
        ax.set_xlim([-1.5, 1.5])
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(unique_loras), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_path = f"eval_results/{type_name}/{metric}/{type_name}_{metric}_avg_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved average visualization to {output_path}")
    plt.close()

def create_visualization(type_name, metric):
    """
    Create visualizations
    """
    all_data_path = f"eval_results/{type_name}/{metric}/all_{type_name}.csv"
    avg_data_path = f"eval_results/{type_name}/{metric}/avg_{type_name}.csv"
    pos_data_path = f"eval_results/{type_name}/{metric}/pos_{type_name}.csv"
    
    print(f"Reading data from {all_data_path}, {avg_data_path}, and {pos_data_path}")
    all_df = pd.read_csv(all_data_path)
    avg_df = pd.read_csv(avg_data_path)
    pos_df = pd.read_csv(pos_data_path)
    
    create_detail_visualization(type_name, metric, all_df)
    create_avg_visualization(type_name, metric, avg_df)
    create_pos_visualization(type_name, metric, pos_df)

if __name__ == "__main__":
    for type_name in ['anime', 'real']:
        for metric in ['mse', 'fid']:
            print(f"\nCreating visualization for {type_name} {metric}...")
            create_visualization(type_name, metric)