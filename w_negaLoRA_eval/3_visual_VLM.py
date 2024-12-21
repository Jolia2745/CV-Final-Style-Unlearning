
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_info(img_name):
    """Extract lora index and scale from image name"""
    parts = img_name.split('_')
    lora_idx = f"{parts[0]}_{parts[1]}"
    scale = float(parts[3].replace('.jpeg', ''))
    if parts[2] == 'neg' and not parts[3].startswith('-'):
        scale = -scale
    return lora_idx, scale

def create_average_visualization(style):
    """
    Create visualization showing average scores across all LoRAs for each scale
    """
    # Read the CSV file
    csv_path = f"eval_results/{style}/VLM/all_VLM.csv"
    df = pd.read_csv(csv_path)
    
    # Extract lora index and scale
    df['lora_idx'], df['scale'] = zip(*df['img_name'].apply(get_info))
    
    # Filter for negative and zero scales only
    df = df[df['scale'] <= 0]
    
    # Calculate average and std for each scale
    avg_data = df.groupby('scale').agg({
        'style_score': ['mean', 'std', 'count']
    }).round(4)
    
    # Plot average scores
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(avg_data.index, 
               avg_data['style_score']['mean'],
               yerr=avg_data['style_score']['std'],
               marker='o', linestyle='-', linewidth=2, markersize=6,
               color='purple', label='Average Score',
               capsize=5, capthick=1.5, elinewidth=1.5)
    
    # Customize plot
    plt.title(f'Average VLM Style Scores - {style.capitalize()}', size=14)
    plt.xlabel('LoRA Scale', size=12)
    plt.ylabel('VLM Style Score', size=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set axis limits
    plt.xlim(-1, 0)
    plt.ylim(0, 100)
    plt.xticks(np.arange(-1, 0.1, 0.2))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure and data
    output_path = f"eval_results/{style}/VLM/avg_scores.png"
    output_csv = f"eval_results/{style}/VLM/avg_scores.csv"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    avg_data.to_csv(output_csv)
    
    print(f"Average visualization saved to {output_path}")
    print(f"Average scores saved to {output_csv}")
    print("\nAverage data:")
    print(avg_data)
    
    plt.close()

if __name__ == "__main__":
    # Process both styles
    for style in ['anime', 'real']:
        print(f"\nProcessing {style} style...")
        create_average_visualization(style)
