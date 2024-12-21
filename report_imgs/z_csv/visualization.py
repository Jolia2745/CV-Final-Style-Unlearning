
# Imports
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For environments without display
import matplotlib.pyplot as plt

def plot_metric(ax, data_df, metric_type, score_type, layer_label):
    """Plot individual metric with error bars"""
    if data_df is None:
        ax.text(0.5, 0.5, 'No data available', 
               horizontalalignment='center',
               verticalalignment='center')
        return
            
    if metric_type == 'clip':
        y_data = data_df[f'{score_type}_score_mean']
        yerr = data_df[f'{score_type}_score_std']
        color = 'blue' if score_type == 'preservation' else 'red'
    else:  # VLM
        y_data = data_df['style_score_mean']
        yerr = data_df['style_score_std']
        color = 'purple'
    
    # Plot with consistent style
    ax.errorbar(data_df['scale'], y_data,
               yerr=yerr,
               marker='o', linestyle='-',
               linewidth=2, markersize=6,
               color=color,
               label=layer_label,
               capsize=5, capthick=1.5, elinewidth=1.5)
    
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    ax.legend()

def set_axis_properties(ax, layer_type, metric_type, style):
    """Set axis properties based on layer type, metric type and style"""
    ax.ticklabel_format(style='plain', axis='x')
    ax.set_xlabel('Training Steps', size=12)
    
    # Set x-axis limits and ticks
    # Only extend range for Feed-Forward Layer's realistic plots
    if layer_type == 'ff' and style == 'real':
        ax.set_xlim(-100000, 4000000)
        ticks = np.arange(0, 4200000, 400000)
    else:
        ax.set_xlim(-50000, 1350000)
        ticks = np.arange(0, 1400000, 200000)
    
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{x/1000:.0f}k' for x in ticks], rotation=45)
    
    # Set y-axis range and label based on metric type
    if metric_type == 'vlm':
        ax.set_ylabel('VLM Score', size=12)
        ax.set_ylim(0, 100)
    else:  # clip
        ax.set_ylabel('CLIP Score', size=12)
        ax.set_ylim(0, 0.35)

    # Ensure grid is behind the plot
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    
    # Set spines visibility
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_visualization():
    """Create visualization for all metrics comparison"""
    # Set figure size and layout
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Read all CSV files
    data = {
        'ff': {
            'anime': {
                'clip': pd.read_csv('ff_anime_clip.csv'),
                'vlm': pd.read_csv('ff_anime_VLM.csv')
            },
            'real': {
                'clip': pd.read_csv('ff_real_clip.csv'),
                'vlm': pd.read_csv('ff_real_VLM.csv')
            }
        },
        'v': {
            'anime': {
                'clip': pd.read_csv('v_anime_clip.csv'),
                'vlm': pd.read_csv('v_anime_VLM.csv')
            },
            'real': {
                'clip': pd.read_csv('v_real_clip.csv'),
                'vlm': pd.read_csv('v_real_VLM.csv')
            }
        }
    }

    # Configure all subplots
    for row, layer_type in enumerate(['ff', 'v']):
        layer_label = 'Feed-Forward Layer' if layer_type == 'ff' else 'Attention-V Layer'
        
        metrics_config = [
            (0, 'clip', 'preservation', 'anime', 'Anime Style - CLIP Preservation Score'),
            (1, 'clip', 'effectiveness', 'anime', 'Anime Style - CLIP Effectiveness Score'),
            (2, 'vlm', 'style', 'anime', 'Anime Style - VLM Score'),
            (3, 'clip', 'preservation', 'real', 'Realistic Style - CLIP Preservation Score'),
            (4, 'clip', 'effectiveness', 'real', 'Realistic Style - CLIP Effectiveness Score'),
            (5, 'vlm', 'style', 'real', 'Realistic Style - VLM Score')
        ]
        
        for col, metric_type, score_type, style, title in metrics_config:
            ax = axes[row, col]
            plot_metric(ax, data[layer_type][style][metric_type], 
                       metric_type, score_type, layer_label)
            ax.set_title(title, size=14)
            set_axis_properties(ax, layer_type, metric_type, style)
    
    # Add main title
    fig.suptitle('Comparison of Feed-Forward and Attention-V Layer Performance', size=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs("evaluation_results", exist_ok=True)
    output_path = "evaluation_results/layer_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_visualization()
