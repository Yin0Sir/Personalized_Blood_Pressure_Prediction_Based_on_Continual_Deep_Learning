import torch
import torch.utils.data as data
import random
import numpy as np
import pandas as pd
import json
import os
import math
import copy
from mat73 import loadmat
from Model_Def.Trainer import Model_Trainer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.load.*weights_only=False.*")

def Seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CLDataset(data.Dataset):
    def __init__(self, Input, Label):
        self.Input = Input
        self.Label = Label

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        return self.Input[idx, :], self.Label[[idx]]

def _subject_to_key(sub):
    """把 mat73 读出来的 Subject cell 元素统一成可哈希的 str key"""
    # 1) 不断拆单元素嵌套：[['S001']] / [array(['S001'])] / array(['S001'])
    while True:
        if isinstance(sub, list) and len(sub) == 1:
            sub = sub[0]
            continue
        if isinstance(sub, np.ndarray) and sub.size == 1:
            sub = sub.item()
            continue
        break

    # 2) MATLAB char array / 字符列表 -> 拼字符串：['S','0','0','1'] 或 array(['S','0','0','1'])
    if isinstance(sub, np.ndarray) and sub.dtype.kind in ("U", "S"):
        return "".join(sub.astype(str).flatten().tolist())
    if isinstance(sub, list) and all(isinstance(x, str) for x in sub):
        return "".join(sub)

    # 3) 正常字符串/其它兜底
    if isinstance(sub, str):
        return sub
    return str(sub)

def load_and_sort_user_data(mat_path, target_label='SBP'):
    print(f"Loading data from {mat_path}...")
    Data = loadmat(mat_path)
    subset = Data['Subset']
    
    subjects = subset['Subject']
    signals  = subset['Signals'][:, 0:2, :]  # 获取 ECG 和 PPG
    labels   = subset[target_label]
    time_seq = subset['Time']

    user2idx = {}
    for i, sub in enumerate(subjects):
        sub_key = _subject_to_key(sub)        # ✅ 修复：保证可哈希
        if sub_key not in user2idx:
            user2idx[sub_key] = []
        user2idx[sub_key].append(i)
        
    user2idx_sorted = {}
    for sub_key, idx_list in user2idx.items():
        idx_array = np.array(idx_list, dtype=np.int64)
        # 获取时间起点并排序
        t0_values = time_seq[idx_array, 0, 0] if time_seq.ndim == 3 else time_seq[idx_array, 0]
        sort_args = np.argsort(t0_values)
        user2idx_sorted[sub_key] = idx_array[sort_args]
        
    return signals, labels, user2idx_sorted

def split_user_stream(idx_sorted, train_ratio=0.7, val_ratio=0.2):
    T = len(idx_sorted)
    return (idx_sorted[:math.floor(train_ratio * T)], 
            idx_sorted[math.floor(train_ratio * T):math.floor((train_ratio + val_ratio) * T)], 
            idx_sorted[math.floor((train_ratio + val_ratio) * T):])

def split_train_to_batches(train_idx, K=3):
    T_train = len(train_idx)
    batch_size = T_train // K
    return [train_idx[i * batch_size : (i + 1) * batch_size if i < K - 1 else T_train] for i in range(K)]

if __name__ == '__main__':
    Seed(6)
    target = 'SBP'  # 或 DBP
    mat_path = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset.mat'
    pretrained_model_path = 'PTH/121517/trained_model.pth'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 数据准备
    signals, labels, user2idx_sorted = load_and_sort_user_data(mat_path, target)
    valid_users = [u for u in user2idx_sorted.keys() if len(user2idx_sorted[u]) >= 300]
    selected_users = random.sample(valid_users, min(10, len(valid_users))) # 测试跑10个人
    
    all_results = []
    modes = ['global_eval', 'seq_ft', 'seq_ewc']
    layers_to_unfreeze = ["final_fc"] # 你可以像 Fine_tune 里一样加入 resnet.layer4
    
    # 2. 批量跑实验
    for u_idx, user_id in enumerate(selected_users):
        if u_idx == 0:
            print("\nidx | user_id | G_MAE | FT_MAE | EWC_MAE | gain(EWC) | forget(EWC)")
        print(f"[{u_idx+1}/{len(selected_users)}] Processing User: {user_id}")
        idx_sorted = user2idx_sorted[user_id]
        train_idx, _, test_idx = split_user_stream(idx_sorted)
        batch_idx_list = split_train_to_batches(train_idx, K=3)
        
        test_loader = data.DataLoader(CLDataset(signals[test_idx], labels[test_idx]), batch_size=32, shuffle=False)
        batch_loaders = [data.DataLoader(CLDataset(signals[b], labels[b]), batch_size=32, shuffle=False) for b in batch_idx_list]
        
        user_res = {'user_id': user_id}
        
        # 每个用户只load一次（放在 for mode 之前）
        base_model = torch.load(pretrained_model_path, map_location="cpu")
        base_state = {k: v.clone() for k, v in base_model.state_dict().items()}  # 干净权重快照

        for mode in modes:
            # 用 deepcopy 保留结构，再用 state_dict 保证权重是“全局初始态”
            model = copy.deepcopy(base_model).to(device)
            model.load_state_dict(base_state, strict=True)

            # 冻结/解冻
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True

            Settings = {
                'BP_optimizer': "torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)",
                'trainer': "Model_Trainer(model, torch.nn.MSELoss(), BP_optimizer, device, Settings, batch_size=32, num_epochs=5, save_states=False, save_final=False)"
            }
            BP_optimizer = eval(Settings['BP_optimizer'])
            model_trainer = eval(Settings['trainer'])

            # 触发新写的范式 ✅ 关键：静默训练（不刷屏）
            res = model_trainer.Train_CL_Model(
                user_id, batch_loaders, test_loader,
                mode=mode, lambda_ewc=1e-3, head_keywords=layers_to_unfreeze,
                verbose=0, show_progress=False
            )
            # 收录该模式结果
            for k, v in res.items():
                if k not in ['user_id', 'mode']:
                    user_res[f"{mode}_{k}"] = v
                
        # 计算增益 (Gain)
        user_res['gain'] = user_res['global_eval_test_mae'] - user_res['seq_ewc_test_mae']
        g = user_res.get('global_eval_test_mae', float('nan'))
        ft = user_res.get('seq_ft_test_mae', float('nan'))
        ewc = user_res.get('seq_ewc_test_mae', float('nan'))
        forget = user_res.get('seq_ewc_forget', float('nan'))
        gain = g - ewc
        print(f"{u_idx+1:>3d} | {user_id} | {g:>6.2f} | {ft:>6.2f} | {ewc:>7.2f} | {gain:>8.2f} | {forget:>10.2f}")

        all_results.append(user_res)
        
    # 3. 保存和统计结果
    output_dir = './Model_Training/CL_Results'
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, f"summary_{target}.csv"), index=False)
    
    stats = {"mean": df.mean(numeric_only=True).to_dict(), "std": df.std(numeric_only=True).to_dict()}
    with open(os.path.join(output_dir, f"summary_{target}.json"), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nAll Done! Mean Personalization Gain: {stats['mean']['gain']:.4f} mmHg")