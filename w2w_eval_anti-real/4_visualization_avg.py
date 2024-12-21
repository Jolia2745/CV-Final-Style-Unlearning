import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_scale_averages(style):
    """
    Create visualization for average preservation and effectiveness scores across scales
    """
    # Read the averages CSV file
    csv_path = f"eval_results/{style}/avg_clip.csv"
    df = pd.read_csv(csv_path)
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot preservation scores
    plt.errorbar(df['scale'], df['preservation_score_mean'], 
                yerr=df['preservation_score_std'],
                fmt='o-',  # Points connected by lines
                capsize=5,  # Cap width for error bars
                capthick=1.5,  # Cap thickness
                elinewidth=1.5,  # Error bar line width
                markersize=8,  # Size of points
                color='blue',
                label='Preservation Score')
    
    # Plot effectiveness scores
    plt.errorbar(df['scale'], df['effectiveness_score_mean'], 
                yerr=df['effectiveness_score_std'],
                fmt='s-',  # Square markers
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                markersize=8,
                color='red',
                label='Effectiveness Score')
    
    # Customize the plot
    plt.title(f'Average Preservation and Effectiveness Scores - {style.capitalize()}', 
              size=14, pad=15)
    plt.xlabel('Scale', size=12)
    plt.ylabel('Score', size=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format axis
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    output_path = f"eval_results/{style}/avg_scores_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Average visualization saved to {output_path}")
    
    plt.close()

def visualize_styles_comparison():
    """
    Create a comparison visualization showing both styles on the same plot
    """
    plt.figure(figsize=(12, 7))
    
    styles = ['anime', 'realistic']
    colors = ['#2E86C1', '#E74C3C']  # Blue for anime, Red for realistic
    
    for style, color in zip(styles, colors):
        # Read data
        csv_path = f"eval_results/{style}/avg_clip.csv"
        df = pd.read_csv(csv_path)
        
        # Plot preservation scores
        plt.errorbar(df['scale'], df['preservation_score_mean'],
                    yerr=df['preservation_score_std'],
                    fmt='o-',
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.5,
                    markersize=8,
                    color=color,
                    label=f'{style.capitalize()} Preservation')
        
        # Plot effectiveness scores (dashed line)
        plt.errorbar(df['scale'], df['effectiveness_score_mean'],
                    yerr=df['effectiveness_score_std'],
                    fmt='s--',  # Square markers with dashed line
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.5,
                    markersize=8,
                    color=color,
                    alpha=0.7,  # Slightly transparent
                    label=f'{style.capitalize()} Effectiveness')
    
    # Customize the plot
    plt.title('Comparison of Average Scores Between Styles', 
              size=14, pad=15)
    plt.xlabel('Scale', size=12)
    plt.ylabel('Score', size=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format axis
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = "eval_results/styles_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to {output_path}")
    
    plt.close()

# Create individual style visualizations
for style in ['anime', 'realistic']:
    print(f"\nProcessing {style} style...")
    visualize_scale_averages(style)

# Create comparison visualization
print("\nCreating styles comparison...")
visualize_styles_comparison()