import os
import pandas as pd
from tqdm import tqdm
import base64
import requests

class StyleEvaluator:
    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.vllm_url = f"http://{host}:{port}"
        self.model = "pixtral-12b-240910-fp8-e4m3"
    
    def encode_image(self, image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_style_score(self, image_path, style):
        """
        Evaluate how well an image matches a given style
        """
        try:
            img_data = self.encode_image(image_path)

            # Define style prompts
            if style == "realistic":
                style_desc = "realistic portrait style"
                style_prompt = """A realistic portrait featuring lifelike proportions, intricate details in the facial features, and subtle textures in the skin and hair. 
                The lighting highlights natural contours, with soft shadows adding depth and dimension. The eyes are expressive and detailed, capturing a sense of emotion, 
                while the overall composition emphasizes realism through accurate anatomy and true-to-life shading."""
            else:
                style_desc = "anime style"
                style_prompt = """An anime-style portrait featuring clean, expressive linework, vibrant colors, and soft shading. 
                The character has large eyes with sparkling highlights, a small, delicate nose, and a slightly stylized mouth. 
                The overall style is sleek, polished, and characteristic of anime art."""

            prompt = f"""Analyze this image and rate how well it matches the {style_desc}:

{style_prompt}

Rate on a scale from 0-100 following these specific criteria:
- 90-100: Perfect match to the {style_desc}. All key characteristics are present and executed flawlessly.
- 80-89: Excellent match. Most characteristics are present and well-executed, with minor deviations.
- 70-79: Good match. Core style elements are present but some aspects could be more refined.
- 60-69: Moderate match. Basic style elements are present but significant refinement needed.
- 40-59: Partial match. Some style elements present but many key characteristics are missing.
- 20-39: Minimal match. Few style elements present, significantly different from target style.
- 0-19: Almost no match to the target style.

Respond ONLY in this exact format:
SCORE: [number]
JUSTIFICATION: [brief explanation of the score]"""

            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content']
            print(result)
            
            try:
                score_line = [line for line in result.split('\n') if 'SCORE:' in line][0]
                score = float(score_line.split(':')[1].strip())
                return min(max(score, 0), 100)  
            except:
                return 0.0

        except Exception as e:
            print(f"Error in style evaluation: {str(e)}")
            return 0.0

def evaluate_directory(style):
    """
    Process style evaluation for all images in a directory
    """
    # Setup evaluator
    evaluator = StyleEvaluator()
    
    # Create output directory
    output_dir = f"eval_results/{style}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base directory
    base_dir = "/scratch/lx2154/w2w/weights2weights/editing/images_ani-real_clip"
    
    # Prepare list for results
    vlm_results = []
    
    # Process each seed directory
    for seed in os.listdir(base_dir):
        seed_dir = os.path.join(base_dir, seed)
        if not os.path.isdir(seed_dir):
            continue
        
        # Process each image in seed directory
        print(f"Processing seed directory: {seed}")
        for image_name in tqdm(sorted(os.listdir(seed_dir))):
            if not image_name.endswith('.png'):
                continue
                
            image_path = os.path.join(seed_dir, image_name)
            
            try:
                # Calculate style score
                score = evaluator.get_style_score(image_path, style)
                
                vlm_results.append({
                    'image_path': image_path,
                    'style_score': score
                })
                
                # Save intermediate results
                temp_df = pd.DataFrame(vlm_results)
                temp_df.to_csv(f"{output_dir}/temp_VLM.csv", index=False)
                
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(f"Error message: {str(e)}")
                continue
    
    if not vlm_results:
        print(f"Warning: No VLM results calculated for {style}")
        return None
    
    # Create and save all_VLM.csv
    all_df = pd.DataFrame(vlm_results)
    output_path = f"{output_dir}/all_VLM.csv"
    all_df.to_csv(output_path, index=False)
    print(f"Saved all VLM results to {output_path}")
    
    return all_df

if __name__ == "__main__":
    # Process both styles
    for style in [ 'anime', "realistic"]:#
        print(f"\nProcessing {style} style...")
        evaluate_directory(style)