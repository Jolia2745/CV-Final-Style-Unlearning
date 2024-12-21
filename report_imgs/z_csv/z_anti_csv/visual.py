
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For environments without display
import matplotlib.pyplot as plt

def create_unlearn_visualization():
    """Create visualization for unlearning results"""
    # Read CSV files
    clip_data = pd.read_csv('avg_clip.csv')
    vlm_data = pd.read_csv('avg_VLM.csv')
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot CLIP Preservation Score
    ax = axes[0]
    ax.errorbar(clip_data['scale'], 
               clip_data['preservation_score_mean'],
               yerr=clip_data['preservation_score_std'],
               marker='o', linestyle='-',
               linewidth=2, markersize=6,
               color='blue',
               label='CLIP Preservation',
               capsize=5, capthick=1.5, elinewidth=1.5)
    ax.set_title('CLIP Preservation Score', size=14)
    ax.set_ylim(0, 0.35)
    ax.legend()
    
    # Plot CLIP Effectiveness Score
    ax = axes[1]
    ax.errorbar(clip_data['scale'], 
               clip_data['effectiveness_score_mean'],
               yerr=clip_data['effectiveness_score_std'],
               marker='o', linestyle='-',
               linewidth=2, markersize=6,
               color='red',
               label='CLIP Effectiveness',
               capsize=5, capthick=1.5, elinewidth=1.5)
    ax.set_title('CLIP Effectiveness Score', size=14)
    ax.set_ylim(0, 0.35)
    ax.legend()
    
    # Plot VLM Score
    ax = axes[2]
    ax.errorbar(vlm_data['scale'], 
               vlm_data['style_score_mean'],
               yerr=vlm_data['style_score_std'],
               marker='o', linestyle='-',
               linewidth=2, markersize=6,
               color='purple',
               label='VLM Score',
               capsize=5, capthick=1.5, elinewidth=1.5)
    ax.set_title('VLM Score', size=14)
    ax.set_ylim(0, 100)
    ax.legend()

    # Set common properties for all subplots
    for ax in axes:
        # Set x-axis properties
        ax.set_xlim(-50000, 550000)
        ticks = np.arange(0, 550000, 100000)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{x/1000:.0f}k' for x in ticks], rotation=45)
        ax.set_xlabel('Unlearning Steps', size=12)
        
        # Set grid properties
        ax.grid(True, linestyle=':', alpha=0.5, color='gray')
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add main title
    fig.suptitle('Anime Style Unlearning Metrics', size=16, y=1.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if not exists
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Save figure
    output_path = "evaluation_results/unlearning_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_unlearn_visualization()
