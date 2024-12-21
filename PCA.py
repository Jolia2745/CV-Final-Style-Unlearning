import torch
from peft.utils.save_and_load import load_peft_weights
from safetensors.torch import load_file
import pandas as pd
import os
from tqdm import tqdm
import ast

def pca(a):
    device = torch.device("cpu")
    a = a.to(torch.float32)
    a = a.to(device)
    mean = torch.mean(a, dim=0)
    std = torch.std(a, dim=0)
    scaled_data = (a - mean)/std
    print('in PCA')
    u,s,v = torch.svd_lowrank(scaled_data, q=857, niter=20, M=None)
    return u,s,v, mean, std

def project(a,v,k, mean, std):
    data = (a-mean)/std
    new = torch.matmul(data,v[:, :k])
    return new

def unproject(projected,v,k, mean, std):
    new = torch.matmul(projected,v[:, :k].T)
    new = new*std+mean
    return new

torch.set_default_device('cpu')
# flat_weights = torch.load('/scratch/lx2154/w2w/attn_v6.pt')

# u,s,v, mean, std = pca(flat_weights) #flat_weights is m*n
# print(v.shape, mean.shape, std.shape)
# torch.save(mean, "mean6.pt")
# torch.save(v, "v6.pt")
# torch.save(std, "std6.pt")

# # flat_weights = torch.load('/scratch/lx2154/w2w/attn_v4.pt')
# # v=torch.load('/scratch/lx2154/w2w/v4.pt')
# # mean=torch.load('/scratch/lx2154/w2w/mean4.pt')
# # std=torch.load('/scratch/lx2154/w2w/std4.pt')

# projection = project(flat_weights, v, 10000, mean, std)#w: (1*n) -> (1*k)
# # unprojection = unproject(projection, v, 10000, mean, std)
# # torch.save(projection, "projection_3.pt")
# # torch.save(unprojection, "unprojection_3.pt")
# pinverse = torch.linalg.pinv(projection)
# torch.save(pinverse,"pinverse6.pt")


adapters_weights = load_file("/scratch/lx2154/w2w/loras/137223.safetensors")
w=[]
for key in adapters_weights.keys():
    if "unet" in key and "ff" in key and "attn" not in key:
        w.append(adapters_weights[key].flatten())
w = torch.cat(w, 0).unsqueeze(0).bfloat16()
print(w.shape)
v=v.bfloat16()
print(v.shape)
weights0 = w@v
print(weights0.shape)
torch.save(weights0,'weights0_3.pt')

#get the flattened v
df = pd.read_csv("models_ani-real_downloaded.csv")
saved_models = pd.DataFrame(columns=["model_id", "ani_real"])
result_matrix=[]
output_csv = 'ani_real_df6.csv'

with tqdm(total=len(df), desc="Processing LoRAs", position=0, leave=True) as pbar:
    for _, row in df.iterrows():
        skip=False #whether to skip this model
        model_id = row['id']
        anime_score = row["image_anime_score"]
        # realistic_score=row["image_realistic_score"]
        rank = row["rank"]
        
        if rank!=16:
            # print("rank not 16 skipping")
            pbar.update(1)
            continue

        ani_real = 1 if anime_score>0.24 else -1
        safetensor_path = f"loras/{model_id}.safetensors"
        w=[]
        
        try:
            adapters_weights = load_file(safetensor_path)
        except Exception as e:
            print(f"Error loading {model_id}: {e}")
            pbar.update(1)
            continue
        
        # Skip model if number of attn_v layers doesn't match
        relevant_key_count = sum(1 for key in adapters_weights.keys() if "unet" in key and "v" in key and "attn" in key)
        if relevant_key_count != 420:
            print(f"Skipping {model_id}, relevant key count: {relevant_key_count}")
            pbar.update(1)
            continue

        # #skip model if rank isn't 32
        # for idx, key in enumerate(adapters_weights.keys()):
        #     if "unet" in key and "attn" in key and "v" in key and "down" in key and 'alpha' not in key:
        #         try:
        #             rank=adapters_weights[key].shape[0]
        #         except Exception as e:
        #             print(key,adapters_weights[key].shape)
        #             print(e)
        #         if rank != 32:
        #             skip=True
        #             break      
        # if skip:
        #     # print(f'skipping {model_id}, rank not 32')
        #     pbar.update(1)
        #     continue

        #到此时model v_layers=420, rank=32, flatten its attn_v
        for idx, key in enumerate(adapters_weights.keys()):
            if "unet" in key and "attn" in key and "v" in key:
                w.append(adapters_weights[key].flatten())
        
        w = torch.cat(w, 0).unsqueeze(0) #(1,n)
        
        if w.shape[1]==6287500:
            if not model_id in saved_models['model_id'].values:
                new_row = {"model_id": model_id, "ani_real": ani_real}
                saved_models = pd.concat([saved_models, pd.DataFrame([new_row])], ignore_index=True)
                #append the row and save the w (保证对应性)
                saved_models.to_csv(output_csv, index=False)
                result_matrix.append(w)
                # print(f"Model {model_id} saved to {output_csv}.")
            else:
                print(f"Model {model_id} already exists in {output_csv}, skipping.")
        else:
            print(f'{model_id} has rank 16 but flatten shape doesn\'t match')    
        pbar.update(1)


if result_matrix:
    output_file='attn_v6.pt'
    result_matrix = torch.cat(result_matrix, 0)
    torch.save(result_matrix, output_file)
    print(f"Batch saved to {output_file} with shape: {result_matrix.shape}")
else:
    print("No valid tensors in batch, skipping save.")