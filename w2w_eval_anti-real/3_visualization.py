import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def create_visualization(style):
    """
    Create visualization for a specific style with subplots for each seed,
    showing both preservation and effectiveness scores
    """
    # Read the CSV file
    csv_path = f"eval_results/{style}/all_clip.csv"
    df = pd.read_csv(csv_path)
    
    # Extract seed and scale from image path
    df['seed'] = df['image_path'].apply(lambda x: x.split('/')[-2])
    df['scale'] = df['image_path'].apply(lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Get unique seeds
    seeds = sorted(df['seed'].unique())
    n_seeds = len(seeds)
    
    # Calculate subplot layout
    n_cols = 3  # You can adjust this number
    n_rows = int(np.ceil(n_seeds / n_cols))
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'CLIP Scores for Different Seeds - {style.capitalize()} Style', size=16, y=1.02)
    
    # Flatten axes array for easier iteration
    axes_flat = axes.flatten() if n_seeds > 1 else [axes]
    
    # Plot for each seed
    for idx, seed in enumerate(seeds):
        ax = axes_flat[idx]
        seed_data = df[df['seed'] == seed]
        
        # Sort by scale
        seed_data = seed_data.sort_values('scale')
        
        # Create the line plots
        ax.plot(seed_data['scale'], seed_data['preservation_score'], 
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color='blue', label='Preservation')
        ax.plot(seed_data['scale'], seed_data['effectiveness_score'], 
                marker='s', linestyle='-', linewidth=2, markersize=6,
                color='red', label='Effectiveness')
        
        # Customize the subplot
        ax.set_title(f'Seed: {seed}')
        ax.set_xlabel('Scale')
        ax.set_ylabel('CLIP Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Format x-axis to show scale values clearly
        ax.ticklabel_format(style='plain', axis='x')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Hide empty subplots if any
    for idx in range(len(seeds), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = f"eval_results/{style}/dual_scores_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Close the figure to free memory
    plt.close()

# Create visualizations for both styles
for style in ['anime', 'realistic']:
    print(f"\nProcessing {style} style...")
    create_visualization(style)