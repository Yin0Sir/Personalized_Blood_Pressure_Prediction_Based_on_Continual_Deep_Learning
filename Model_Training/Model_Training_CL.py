import torch
import torch.utils.data as data
import random
import numpy as np
import pandas as pd
import json
import os
import math
import time
from datetime import datetime
import copy
from mat73 import loadmat
from Model_Def.Trainer import Model_Trainer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.load.*weights_only=False.*")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    while True:
        if isinstance(sub, list) and len(sub) == 1:
            sub = sub[0]
            continue
        if isinstance(sub, np.ndarray) and sub.size == 1:
            sub = sub.item()
            continue
        break
    if isinstance(sub, np.ndarray) and sub.dtype.kind in ("U", "S"):
        return "".join(sub.astype(str).flatten().tolist())
    if isinstance(sub, list) and all(isinstance(x, str) for x in sub):
        return "".join(sub)
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
        sub_key = _subject_to_key(sub)
        if sub_key not in user2idx:
            user2idx[sub_key] = []
        user2idx[sub_key].append(i)
        
    user2idx_sorted = {}
    for sub_key, idx_list in user2idx.items():
        idx_array = np.array(idx_list, dtype=np.int64)
        t0_values = time_seq[idx_array, 0, 0] if time_seq.ndim == 3 else time_seq[idx_array, 0]
        sort_args = np.argsort(t0_values)
        user2idx_sorted[sub_key] = idx_array[sort_args]
        
    return signals, labels, user2idx_sorted

def split_user_stream(idx_sorted, train_ratio=0.8, val_ratio=0.1):
    T = len(idx_sorted)
    return (idx_sorted[:math.floor(train_ratio * T)], 
            idx_sorted[math.floor(train_ratio * T):math.floor((train_ratio + val_ratio) * T)], 
            idx_sorted[math.floor((train_ratio + val_ratio) * T):])

def split_train_to_batches(train_idx, K=4):
    T_train = len(train_idx)
    batch_size = T_train // K
    return [train_idx[i * batch_size : (i + 1) * batch_size if i < K - 1 else T_train] for i in range(K)]

UNFREEZE_PRESETS = {
    "head_only": ["final_fc"],
    "head_4": ["final_fc", "resnet.layer4"],
    "head_3_4": ["final_fc", "resnet.layer3", "resnet.layer4"],
    "head_c": ["final_fc", "cornet"],
    "head_c_4": ["final_fc", "cornet", "resnet.layer4"],

}
UNFREEZE_PRESET = "head_c"  # 当前启用哪个组合

def set_trainable_by_prefix(model, prefixes):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(name == pref or name.startswith(pref + ".") for pref in prefixes):
            p.requires_grad = True

if __name__ == '__main__':
    Seed(6)
    target = 'SBP'  # 或 DBP
    mat_path = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset.mat'
    pretrained_model_path = 'PTH/121517/trained_model.pth'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    TimeID = datetime.now().strftime('%H%M%S')
    
    # 1. 数据准备
    signals, labels, user2idx_sorted = load_and_sort_user_data(mat_path, target)
    valid_users = [u for u in user2idx_sorted.keys() if len(user2idx_sorted[u]) >= 300]
    selected_users = random.sample(valid_users, min(200, len(valid_users))) # 测试跑20个人
    
    all_results = []
    modes = ['global_eval', 'seq_ft', 'seq_ewc']
    layers_to_unfreeze = UNFREEZE_PRESETS[UNFREEZE_PRESET]
    
    # 2. 批量跑实验
    for u_idx, user_id in enumerate(selected_users):
        t_user0 = time.time()
        print(f"[{u_idx+1}/{len(selected_users)}] Processing User: {user_id}")
        idx_sorted = user2idx_sorted[user_id]
        train_idx, val_idx, test_idx = split_user_stream(idx_sorted)
        batch_idx_list = split_train_to_batches(train_idx, K=4)
        val_loader  = data.DataLoader(CLDataset(signals[val_idx],  labels[val_idx]),  batch_size=8, shuffle=False)
        test_loader = data.DataLoader(CLDataset(signals[test_idx], labels[test_idx]), batch_size=8, shuffle=False)
        batch_loaders = [data.DataLoader(CLDataset(signals[b], labels[b]), batch_size=8, shuffle=False) for b in batch_idx_list]
        
        user_res = {'user_id': user_id}
        
        # 每个用户只load一次（放在 for mode 之前）
        base_model = torch.load(pretrained_model_path, map_location="cpu")
        base_state = {k: v.clone() for k, v in base_model.state_dict().items()}  # 干净权重快照

        for mode in modes:
            # 用 deepcopy 保留结构，再用 state_dict 保证权重是“全局初始态”
            model = copy.deepcopy(base_model).to(device)
            model.load_state_dict(base_state, strict=True)

            # 冻结/解冻
            set_trainable_by_prefix(model, layers_to_unfreeze)
            if mode == modes[0]:
                user_res['trainable_params'] = count_trainable_params(model)
            # 解冻后立刻打印（建议只跑一次）
            if u_idx == 0 and mode == modes[0]:
                trainable = [n for n, p in model.named_parameters() if p.requires_grad]
                print("UNFREEZE_PRESET:", UNFREEZE_PRESET, "->", layers_to_unfreeze)

            Settings = {
                'BP_optimizer': "torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, betas=(0.9, 0.999), weight_decay=0)",
                'trainer': "Model_Trainer(model, torch.nn.MSELoss(), BP_optimizer, device, Settings, batch_size=8, num_epochs=20, save_states=False, save_final=False, timeid=TimeID)"
            }
            BP_optimizer = eval(Settings['BP_optimizer'])
            model_trainer = eval(Settings['trainer'])

            res = model_trainer.Train_CL_Model(
                user_id, batch_loaders, test_loader, val_loader=val_loader,
                val_check='batch',              # ✅ 每个CL batch后评估一次val
                rollback_to_best=True,          # ✅ 训练结束回滚到val最优
                patience=0,                     # ✅ 0=不早停，只回滚（最保守）
                mode=mode, lambda_ewc=100, trainable_keywords=layers_to_unfreeze,
                verbose=0, show_progress=False
            )
            # 收录该模式结果
            for k, v in res.items():
                if k not in ['user_id', 'mode']:
                    user_res[f"{mode}_{k}"] = v
                
        # 计算增益 (Gain)
        user_res['gain'] = user_res['global_eval_test_mae'] - user_res['seq_ewc_test_mae']
        g_mae  = user_res.get('global_eval_test_mae',  float('nan'))
        g_rmse = user_res.get('global_eval_test_rmse', float('nan'))
        g_r2   = user_res.get('global_eval_test_r2',   float('nan'))

        ft_mae  = user_res.get('seq_ft_test_mae',  float('nan'))
        ft_rmse = user_res.get('seq_ft_test_rmse', float('nan'))
        ft_r2   = user_res.get('seq_ft_test_r2',   float('nan'))

        ewc_mae  = user_res.get('seq_ewc_test_mae',  float('nan'))
        ewc_rmse = user_res.get('seq_ewc_test_rmse', float('nan'))
        ewc_r2   = user_res.get('seq_ewc_test_r2',   float('nan'))

        forget = user_res.get('seq_ewc_forget', float('nan'))

        gain_mae  = g_mae - ewc_mae
        gain_rmse = g_rmse - ewc_rmse
        delta_r2  = ewc_r2 - g_r2

        # 训练参数量（同一用户同一 preset 固定，存一份即可）
        user_res['trainable_params'] = user_res.get('trainable_params', float('nan'))  # 你也可在首次解冻后赋值
        t_user = time.time() - t_user0

        if u_idx == 0:
            print("\nidx | user_id | P(M) | G(MAE/RMSE/R2) | FT(MAE/RMSE/R2) | EWC(MAE/RMSE/R2) | gain_mae | dR2 | forget | t(s)")

        print(
            f"{u_idx+1:>3d} | {user_id} | {user_res.get('trainable_params', float('nan'))/1e6:>4.2f} | "
            f"{g_mae:>5.2f}/{g_rmse:>5.2f}/{g_r2:>6.3f} | "
            f"{ft_mae:>5.2f}/{ft_rmse:>5.2f}/{ft_r2:>6.3f} | "
            f"{ewc_mae:>5.2f}/{ewc_rmse:>5.2f}/{ewc_r2:>6.3f} | "
            f"{gain_mae:>8.2f} | {delta_r2:>6.3f} | {forget:>6.2f} | {t_user:>5.1f}"
        )

        all_results.append(user_res)
        
    # 3. 保存和统计结果
    output_dir = './Model_Training/CL_Results'
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    def series_stats(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return {}
        return {
            "n": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        }

    # 关键列（按你的命名：mode_metric）
    cols = {
        "gain_mae": "gain",
        "forget": "seq_ewc_forget",
        "g_mae": "global_eval_test_mae",
        "g_rmse": "global_eval_test_rmse",
        "g_r2": "global_eval_test_r2",
        "ft_mae": "seq_ft_test_mae",
        "ft_rmse": "seq_ft_test_rmse",
        "ft_r2": "seq_ft_test_r2",
        "ewc_mae": "seq_ewc_test_mae",
        "ewc_rmse": "seq_ewc_test_rmse",
        "ewc_r2": "seq_ewc_test_r2",
        "trainable_params": "trainable_params",
    }

    stats = {"columns": {}, "overall": {}}
    for k, c in cols.items():
        if c in df.columns:
            stats["columns"][k] = series_stats(df[c])

    # 胜率/分组统计（以 MAE 为准）
    if "gain" in df.columns:
        improved = df["gain"] > 0
        stats["overall"]["n_users"] = int(len(df))
        stats["overall"]["improve_rate_mae"] = float(improved.mean())
        stats["overall"]["n_improved"] = int(improved.sum())
        stats["overall"]["n_degraded"] = int((~improved).sum())

        if improved.any():
            stats["overall"]["gain_mean_improved"] = float(df.loc[improved, "gain"].mean())
            stats["overall"]["gain_median_improved"] = float(df.loc[improved, "gain"].median())
        if (~improved).any():
            stats["overall"]["gain_mean_degraded"] = float(df.loc[~improved, "gain"].mean())
            stats["overall"]["gain_median_degraded"] = float(df.loc[~improved, "gain"].median())

    # best / worst user（按 gain）
    if "gain" in df.columns and "user_id" in df.columns and len(df) > 0:
        best_idx = df["gain"].idxmax()
        worst_idx = df["gain"].idxmin()
        stats["overall"]["best_user"] = {
            "user_id": df.loc[best_idx, "user_id"],
            "gain": float(df.loc[best_idx, "gain"]),
            "g_mae": float(df.loc[best_idx, "global_eval_test_mae"]),
            "ewc_mae": float(df.loc[best_idx, "seq_ewc_test_mae"]),
        }
        stats["overall"]["worst_user"] = {
            "user_id": df.loc[worst_idx, "user_id"],
            "gain": float(df.loc[worst_idx, "gain"]),
            "g_mae": float(df.loc[worst_idx, "global_eval_test_mae"]),
            "ewc_mae": float(df.loc[worst_idx, "seq_ewc_test_mae"]),
        }

    # 写 JSON
    df.to_csv(os.path.join(output_dir, f"summary-{target}-{TimeID}.csv"), index=False)
    with open(os.path.join(output_dir, f"summary-{target}-{TimeID}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # 控制台也打印一段简洁统计
    print(f"\nUsers: {stats['overall'].get('n_users', len(df))} | Improve rate(MAE): {stats['overall'].get('improve_rate_mae', float('nan')):.3f}")
    print(f"Gain(MAE) mean±std: {stats['columns'].get('gain_mae', {}).get('mean', float('nan')):.3f} ± {stats['columns'].get('gain_mae', {}).get('std', float('nan')):.3f}")
    print(f"Gain(MAE) median[IQR]: {stats['columns'].get('gain_mae', {}).get('median', float('nan')):.3f} [{stats['columns'].get('gain_mae', {}).get('iqr', float('nan')):.3f}]")
    print(f"Forget mean±std: {stats['columns'].get('forget', {}).get('mean', float('nan')):.3f} ± {stats['columns'].get('forget', {}).get('std', float('nan')):.3f}")