import pandas as pd
import os

def calculate_scale_averages(style):
    """
    Calculate average preservation and effectiveness scores for each scale across different seeds
    
    Args:
        style (str): Either 'anime' or 'realistic'
    """
    # Read the CSV file
    csv_path = f"eval_results/{style}/all_clip.csv"
    df = pd.read_csv(csv_path)
    
    # Extract scale from image path
    df['scale'] = df['image_path'].apply(lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Calculate averages for both scores
    scale_averages = df.groupby('scale').agg({
        'preservation_score': ['mean', 'std', 'count'],
        'effectiveness_score': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    scale_averages.columns = [
        f"{col[0]}_{col[1]}" for col in scale_averages.columns
    ]
    
    # Sort by scale
    scale_averages = scale_averages.sort_index()
    
    # Save results
    output_path = f"eval_results/{style}/avg_clip.csv"
    scale_averages.to_csv(output_path)
    
    print(f"\nResults for {style}:")
    print(scale_averages)
    print(f"\nResults saved to {output_path}")
    
    return scale_averages

# Process both styles
for style in ['anime', 'realistic']:
    print(f"\nProcessing {style} style...")
    scale_averages = calculate_scale_averages(style)