import os
import torch
from safetensors.torch import load_file
from PIL import Image
from peft import PeftModel
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline

def main():
    parser = argparse.ArgumentParser(description="image generation")
    parser.add_argument('--start_index', type=int, help="start")
    parser.add_argument('--end_index', type=int, help="end")
    args = parser.parse_args()
    
    start_index = args.start_index
    end_index = args.end_index
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    lora_path = "./lora/anime_final"
    lora_files = [f for f in os.listdir(lora_path) if os.path.isfile(os.path.join(lora_path, f))]
    lora_mapping = {os.path.splitext(f)[0]: os.path.join(lora_path, f) for f in lora_files}

    print("Available LoRA files:")
    for name, path in lora_mapping.items():
        print(f"{name} -> {path}")

    selected_lora_files = list(lora_mapping.items())[start_index:end_index]
    print(f"processing lora files from {start_index} to {end_index}")

    prompt_style_index = 1

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16).to(device)

    prompts=["a girl, anime"]

    negative_prompt = "blurry, bad proportions, asymmetric ears, broken wrist, additional limbs, asymmetric, collapsed eyeshadow, altered appendages, broken finger, bad anatomy, elongated throat, double face, conjoined, bad face, broken hand, out of frame, disconnected limb, 3d, bad ears, amputee, cross-eyed, disfigured, cartoon, bad eyes, cloned face, combined appendages, broken leg, copied visage, absent limbs, childish, cropped head, cloned head, desiccated, duplicated features, dismembered, disproportionate, cripple"

    lora_scales = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    output_dir = "./outputs/t2i/anime_final"
    os.makedirs(output_dir, exist_ok=True)

    lora_index=0+start_index
    for lora_name, path in selected_lora_files:
        print(f"start loading lora index {lora_index}")
        lora_output_dir = os.path.join(output_dir, lora_name)
        os.makedirs(lora_output_dir, exist_ok=True)

        try:
            PeftModel.is_peft_available = lambda: True
            pipe.load_lora_weights(lora_path, weight_name=os.path.basename(path))
        except Exception as e:
            print(f"Failed to load LoRA weights for {lora_name}: {e}")
            continue

        for prompt in prompts:
            for scale in lora_scales:
                print(f"Generating for prompt: '{prompt}' with LoRA scale: {scale}")
                
                generator = torch.manual_seed(2870305590)  
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=1024, 
                    width=1024, 
                    num_inference_steps=30,
                    guidance_scale=5,
                    cross_attention_kwargs={"scale": scale},
                    generator=generator
                )
                
                if scale > 0:
                    lora_sign = 'pos'
                elif scale < 0:
                    lora_sign = 'neg'
                else:
                    lora_sign = 'base'
                filename = f"{prompt_style_index}_{lora_index}_{lora_sign}_{scale}.jpeg"
                result.images[0].save(os.path.join(lora_output_dir, filename))
                print(f"Saved image to {os.path.join(lora_output_dir, filename)}")
        lora_index += 1

if __name__ == "__main__":
    main()
