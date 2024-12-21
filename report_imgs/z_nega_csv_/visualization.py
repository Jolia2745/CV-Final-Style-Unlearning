
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_data(style):
    """Read data for a specific style"""
    clip_data = pd.read_csv(f'{style}_clip.csv')
    vlm_data = pd.read_csv(f'{style}_VLM.csv')
    return clip_data, vlm_data

def create_visualization():
    """Create visualization for all styles and metrics"""
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Process each style
    styles = ['anime', 'real']
    for row, style in enumerate(styles):
        # Read data
        clip_data, vlm_data = read_data(style)
        
        # Plot CLIP Preservation Score
        ax = axes[row, 0]
        ax.errorbar(clip_data['scale'], 
                   clip_data['preservation_score_mean'],
                   yerr=clip_data['preservation_score_std'],
                   marker='o', linestyle='-', linewidth=2, markersize=6,
                   color='blue', capsize=5, capthick=1.5, elinewidth=1.5)
        ax.set_title(f'{style.capitalize()} Style - CLIP Preservation Score', size=12)
        ax.set_ylim(0, 0.35)
        
        # Plot CLIP Effectiveness Score
        ax = axes[row, 1]
        ax.errorbar(clip_data['scale'], 
                   clip_data['effectiveness_score_mean'],
                   yerr=clip_data['effectiveness_score_std'],
                   marker='o', linestyle='-', linewidth=2, markersize=6,
                   color='red', capsize=5, capthick=1.5, elinewidth=1.5)
        ax.set_title(f'{style.capitalize()} Style - CLIP Effectiveness Score', size=12)
        ax.set_ylim(0, 0.35)
        
        # Plot VLM Score
        ax = axes[row, 2]
        ax.errorbar(vlm_data['scale'], 
                   vlm_data['style_score_mean'],
                   yerr=vlm_data['style_score_std'],
                   marker='o', linestyle='-', linewidth=2, markersize=6,
                   color='purple', capsize=5, capthick=1.5, elinewidth=1.5)
        ax.set_title(f'{style.capitalize()} Style - VLM Score', size=12)
        ax.set_ylim(0, 100)
        
        # Set common properties for the row
        for ax in axes[row]:
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_xlabel('Negative LoRA Scale', size=10)
            ax.set_xlim(-1.1, 0.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set x ticks
            ax.set_xticks(np.arange(-1.0, 0.1, 0.2))
            # Add vertical line at x=0
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add main title
    fig.suptitle('Comparison of Metrics Across Styles with Negative LoRA Scale', size=14, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('visualization_results', exist_ok=True)
    
    # Save figure
    output_path = 'visualization_results/style_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_visualization()
