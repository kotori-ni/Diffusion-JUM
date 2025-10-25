import os
import torch
import numpy as np
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss

def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx

iterations = 5
model_name = "wavinghand"
version = "v1"
ori_data = np.load(f'./OUTPUT/{model_name}/samples/{model_name}_norm_truth_500_train.npy')
csv_path = f"./Record/record_{model_name}_{version}.csv"

# Write header to CSV
with open(csv_path, 'w') as f:
    f.write("Model,Context_FID,Correlational_Score\n")

for i in tqdm(range(16,26)):
    fake_data = np.load(f'./OUTPUT/{model_name}/ddpm_fake_{model_name}_{i}.npy')
    context_fid_score = []

    for j in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
        context_fid_score.append(context_fid)
        print(f'Iter {j}: ', 'context-fid =', context_fid, '\n')
        
    context_mean, context_sigma = display_scores(context_fid_score)

    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)

    correlational_score = []
    size = int(x_real.shape[0] / iterations)

    for j in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())
        print(f'Iter {j}: ', 'cross-correlation =', loss.item(), '\n')

    corr_mean, corr_sigma = display_scores(correlational_score)

    # Append to CSV
    with open(csv_path, 'a') as f:
        f.write(f"{model_name}_{i},{context_mean:.3f}±{context_sigma:.3f},{corr_mean:.3f}±{corr_sigma:.3f}\n")