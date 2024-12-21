import os
import pandas as pd
from pathlib import Path

def create_meta_csv(directory, name):
 
    image_data = []
    

    for lora_dir in os.listdir(directory):
        lora_path = os.path.join(directory, lora_dir)
        
        # 确保是目录而不是文件
        if os.path.isdir(lora_path):
            # 遍历该lora目录下的所有图片
            for img_name in os.listdir(lora_path):
                if img_name.endswith('.jpeg'):
        
                    img_path = f"{directory}/{lora_dir}/{img_name}"
                    
        
                    image_data.append({
                        'img_name': img_name,
                        'path': img_path
                    })
    

    df = pd.DataFrame(image_data)
    
    # 保存为CSV文件
    df.to_csv(name, index=False)
    
    return df

# 使用示例
if __name__ == "__main__":
    anime_dir = "./anime"
    anime_name = "meta_anime.csv"
    real_dir = "./realistic"
    real_name = "meta_real.csv"

    try:
        df = create_meta_csv(anime_dir, anime_name)
        print(f"共处理了 {len(df)} 张图片")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")