
import os
import pandas as pd
import base64
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import time

class BatchStyleEvaluator:
    def __init__(self, host="localhost", port=8000, batch_size=4, num_threads=4):
        self.host = host
        self.port = port
        self.vllm_url = f"http://{host}:{port}"
        self.model = "pixtral-12b-240910-fp8-e4m3"
        self.batch_size = batch_size
        self.num_threads = num_threads
        
    def encode_image(self, image_path: str) -> str:
        """Encode single image to base64"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def get_style_prompt(self, style: str) -> str:
        """Get style-specific prompt"""
        if style == "real":
            style_desc = "realistic portrait style"
            style_prompt = """A realistic portrait featuring lifelike proportions, intricate details in the facial features, and subtle textures in the skin and hair. 
            The lighting highlights natural contours, with soft shadows adding depth and dimension. The eyes are expressive and detailed, capturing a sense of emotion, 
            while the overall composition emphasizes realism through accurate anatomy and true-to-life shading."""
        else:
            style_desc = "anime style"
            style_prompt = """An anime-style portrait featuring clean, expressive linework, vibrant colors, and soft shading. 
            The character has large eyes with sparkling highlights, a small, delicate nose, and a slightly stylized mouth. 
            The overall style is sleek, polished, and characteristic of anime art."""

        return f"""Analyze this image and rate how well it matches the {style_desc}:

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

    def process_batch_response(self, response_text: str) -> float:
        """Extract score from VLM response"""
        try:
            score_line = [line for line in response_text.split('\n') if 'SCORE:' in line][0]
            score = float(score_line.split(':')[1].strip())
            return min(max(score, 0), 100)
        except:
            return 0.0

    def evaluate_batch(self, image_batch: List[Dict], style: str) -> List[Dict]:
        """Evaluate a batch of images"""
        results = []
        prompt = self.get_style_prompt(style)
        
        try:
            # Prepare batch request
            batch_messages = []
            for img_data in image_batch:
                encoded_image = self.encode_image(img_data['path'])
                if encoded_image:
                    batch_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                            }
                        ]
                    })

            # Send batch request
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": batch_messages
            }

            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Process responses
            for idx, choice in enumerate(response.json()['choices']):
                score = self.process_batch_response(choice['message']['content'])
                results.append({
                    'img_name': image_batch[idx]['img_name'],
                    'style_score': score
                })

        except Exception as e:
            print(f"Error in batch evaluation: {str(e)}")
            # Return default scores for failed batch
            results.extend([{
                'img_name': img['img_name'],
                'style_score': 0.0
            } for img in image_batch])

        return results

    def process_images(self, df: pd.DataFrame, style: str) -> List[Dict]:
        """Process all images using thread pool"""
        all_results = []
        
        # Create batches
        batches = []
        current_batch = []
        for _, row in df.iterrows():
            current_batch.append({
                'img_name': row['img_name'],
                'path': row['path']
            })
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)

        # Process batches with thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_batch = {
                executor.submit(self.evaluate_batch, batch, style): batch 
                for batch in batches
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Batch processing failed: {str(e)}")
                    batch = future_to_batch[future]
                    all_results.extend([{
                        'img_name': img['img_name'],
                        'style_score': 0.0
                    } for img in batch])

        return all_results

def process_evaluation(style: str, meta_csv_path: str):
    """Main evaluation function"""
    evaluator = BatchStyleEvaluator(batch_size=4, num_threads=4)
    
    output_dir = f"eval_results/{style}/VLM"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)
    
    vlm_results = evaluator.process_images(df, style)
    
    if vlm_results:
        all_df = pd.DataFrame(vlm_results)
        output_path = f"{output_dir}/all_VLM.csv"
        all_df.to_csv(output_path, index=False)
        print(f"Saved all VLM results to {output_path}")
    else:
        print("Warning: No VLM results calculated")

if __name__ == "__main__":
    # Process both styles
    for style, meta_file in [
        ('anime', 'meta_anime.csv'),
        ('real', 'meta_real.csv')
    ]:
        print(f"\nProcessing {style} style...")
        process_evaluation(style, meta_file)
