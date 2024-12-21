
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_info(img_name):
    """Extract LoRA index and scale from image name"""
    parts = img_name.split('_')
    lora_idx = f"{parts[0]}_{parts[1]}"
    scale = float(parts[-1].replace('.jpeg', ''))
    if parts[2] == 'neg' and not parts[3].startswith('-'):
        scale = -scale
    return lora_idx, scale

def create_lora_visualization(df, lora_indices, style):
    """Create visualization for individual LoRAs"""
    # Calculate required subplot layout
    n_plots = len(lora_indices)
    n_cols = 3  # Set to 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate required rows
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig.suptitle(f'CLIP Scores vs Negative LoRA Scale - {style.capitalize()}', size=16, y=1.02)
    
    # Ensure axes is 2D array even with one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot for each LoRA index
    for idx, lora_idx in enumerate(lora_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        lora_data = df[df['lora_idx'] == lora_idx].sort_values('scale')
        
        # Plot both scores
        ax.plot(lora_data['scale'], lora_data['preservation_score'], 
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color='blue', label='Preservation')
        ax.plot(lora_data['scale'], lora_data['effectiveness_score'], 
                marker='s', linestyle='-', linewidth=2, markersize=6,
                color='red', label='Effectiveness')
        
        # Customize subplot
        ax.set_title(f'LoRA: {lora_idx}')
        ax.set_xlabel('LoRA Scale')
        ax.set_ylabel('CLIP Score')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set x-axis limits for negative range
        ax.set_xlim(-1, 0)
        ax.set_xticks(np.arange(-1, 0.1, 0.2))
        ax.set_ylim(0, 0.35)
    
    # Hide unused subplots
    for row in range(n_rows):
        for col in range(n_cols):
            if row * n_cols + col >= n_plots:
                axes[row, col].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = f"eval_results/{style}/clip/individual_lora_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Individual LoRA visualization saved to {output_path}")
    plt.close()

def create_average_visualization(df, style):
    """Create visualization for average scores across all LoRAs"""
    # Calculate averages across all LoRAs for each scale
    avg_results = df.groupby('scale').agg({
        'preservation_score': ['mean', 'std'],
        'effectiveness_score': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    avg_results.columns = [f"{col[0]}_{col[1]}" for col in avg_results.columns]
    avg_results = avg_results.sort_index()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot average scores with error bars
    plt.errorbar(avg_results.index, 
                avg_results['preservation_score_mean'],
                yerr=avg_results['preservation_score_std'],
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color='blue', label='Preservation',
                capsize=5, capthick=1.5, elinewidth=1.5)
    
    plt.errorbar(avg_results.index, 
                avg_results['effectiveness_score_mean'],
                yerr=avg_results['effectiveness_score_std'],
                marker='s', linestyle='-', linewidth=2, markersize=6,
                color='red', label='Effectiveness',
                capsize=5, capthick=1.5, elinewidth=1.5)
    
    # Customize plot
    plt.title(f'Average CLIP Scores Across All LoRAs - {style.capitalize()}', size=14)
    plt.xlabel('LoRA Scale', size=12)
    plt.ylabel('CLIP Score', size=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # Set axis limits
    plt.xlim(-1, 0)
    plt.ylim(0, 0.35)
    plt.xticks(np.arange(-1, 0.1, 0.2))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure and data
    output_plot = f"eval_results/{style}/clip/average_scores.png"
    output_csv = f"eval_results/{style}/clip/average_scores.csv"
    
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    avg_results.to_csv(output_csv)
    
    print(f"Average visualization saved to {output_plot}")
    print(f"Average scores saved to {output_csv}")
    
    plt.close()
    return avg_results

def process_visualizations(style):
    """Process both individual and average visualizations"""
    # Read the CSV file
    csv_path = f"eval_results/{style}/clip/all_{style}.csv"
    df = pd.read_csv(csv_path)
    
    # Extract lora index and scale
    df['lora_idx'], df['scale'] = zip(*df['img_name'].apply(get_info))
    
    # Filter for negative and zero scales only
    df = df[df['scale'] <= 0]
    
    # Get unique LoRA indices
    lora_indices = sorted(df['lora_idx'].unique())
    
    # Create visualizations
    create_lora_visualization(df, lora_indices, style)
    create_average_visualization(df, style)

if __name__ == "__main__":
    # Process both styles
    for style in ['anime']:
        print(f"\nProcessing {style} style...")
        process_visualizations(style)
