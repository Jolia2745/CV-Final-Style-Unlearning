import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_info_from_path(image_path):
    """Extract seed and scale from image path"""
    parts = image_path.split('/')
    seed = parts[-2]
    scale = int(parts[-1].replace('.png', '').replace('.pn', ''))
    return seed, scale

def create_seed_visualization(style):
    """
    Create visualization showing all seeds in subplots
    """
    # Read the CSV file
    csv_path = f"./eval_results/{style}/all_VLM.csv"
    df = pd.read_csv(csv_path)
    
    # Extract seed and scale
    df['seed'], df['scale'] = zip(*df['image_path'].apply(get_info_from_path))
    
    # Create figure with subplots (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'VLM Style Scores by Seed - {style.capitalize()}', size=16, y=1.02)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Get unique seeds
    seeds = sorted(df['seed'].unique())
    
    # Plot for each seed
    for idx, seed in enumerate(seeds):
        ax = axes_flat[idx]
        seed_data = df[df['seed'] == seed].sort_values('scale')
        
        # Plot VLM scores
        ax.plot(seed_data['scale'], seed_data['style_score'], 
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color='purple', label='Style Score')
        
        # Add value labels
        for x, y in zip(seed_data['scale'], seed_data['style_score']):
            ax.annotate(f'{y:.1f}', 
                      (x, y), 
                      textcoords="offset points", 
                      xytext=(0,10), 
                      ha='center',
                      fontsize=8)
        
        # Customize subplot
        ax.set_title(f'Seed: {seed}')
        ax.set_xlabel('Scale')
        ax.set_ylabel('VLM Style Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Set axis limits and ticks
        ax.set_ylim(0, 100)
        ax.ticklabel_format(style='plain', axis='x')
        
        # Format x-axis ticks
        ticks = np.arange(0, 900000, 100000)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{x/1000:.0f}k' for x in ticks], rotation=45)
    
    # Hide unused subplots
    for idx in range(len(seeds), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(f"./eval_results/{style}", exist_ok=True)
    
    # Save the figure
    output_path = f"./eval_results/{style}/detailed_VLM.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Seed visualization saved to {output_path}")
    
    plt.close()

def create_average_visualization(style):
    """
    Create visualization showing average scores across seeds
    """
    # Read the CSV file
    csv_path = f"./eval_results/{style}/all_VLM.csv"
    df = pd.read_csv(csv_path)
    
    # Extract seed and scale
    df['seed'], df['scale'] = zip(*df['image_path'].apply(get_info_from_path))
    
    # Calculate averages
    avg_results = df.groupby('scale').agg({
        'style_score': ['mean', 'std', 'count']
    }).round(4)
    
    # Flatten column names
    avg_results.columns = [f"{col[0]}_{col[1]}" for col in avg_results.columns]
    
    # Sort by scale
    avg_results = avg_results.sort_index()
    
    # Save average results
    output_csv = f"./eval_results/{style}/avg_VLM_scores.csv"
    avg_results.to_csv(output_csv)
    print(f"Average results saved to {output_csv}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot average scores with error bars
    plt.errorbar(avg_results.index, 
                avg_results['style_score_mean'],
                yerr=avg_results['style_score_std'],
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color='purple', label='Average Style Score',
                capsize=5, capthick=1.5, elinewidth=1.5)
    
    # Add value labels
    for x, y, std in zip(avg_results.index, 
                        avg_results['style_score_mean'], 
                        avg_results['style_score_std']):
        plt.annotate(f'{y:.1f}±{std:.1f}', 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)
    
    # Customize plot
    plt.title(f'Average VLM Style Scores - {style.capitalize()}', size=14)
    plt.xlabel('Scale', size=12)
    plt.ylabel('Style Score', size=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set axis limits and ticks
    plt.ylim(0, 100)
    
    # Format x-axis ticks
    ticks = np.arange(0, 900000, 100000)
    plt.xticks(ticks, [f'{x/1000:.0f}k' for x in ticks], rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_plot = f"./eval_results/{style}/avg_VLM.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Average visualization saved to {output_plot}")
    
    plt.close()

if __name__ == "__main__":
    # Process both styles
    for style in ['realistic']:  # 可以添加 'realistic' 如果需要
        print(f"\nProcessing {style} style...")
        create_seed_visualization(style)
        create_average_visualization(style)