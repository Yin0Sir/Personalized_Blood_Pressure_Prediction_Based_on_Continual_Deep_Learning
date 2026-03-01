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

# 1. 基础辅助函数
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

# 2. 数据处理逻辑
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

    subjects, signals, labels, time_seq = subset['Subject'], subset['Signals'][:, 0:2, :], subset[target_label], subset['Time']

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
        user2idx_sorted[sub_key] = idx_array[np.argsort(t0_values)]
        
    return signals, labels, user2idx_sorted

def split_user_stream(idx_sorted, train_ratio=0.8, val_ratio=0.1):
    T = len(idx_sorted)
    t_end, v_end = math.floor(train_ratio * T), math.floor((train_ratio + val_ratio) * T)
    return idx_sorted[:t_end], idx_sorted[t_end:v_end], idx_sorted[v_end:]

def split_train_to_batches(train_idx, K=4):
    T_train = len(train_idx)
    batch_size = T_train // K
    return [train_idx[i * batch_size : (i + 1) * batch_size if i < K - 1 else T_train] for i in range(K)]

# 3. 模型配置设置
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

# 4. 结果保存与统计模块
def save_and_summarize_results(all_results, output_dir, target, TimeID):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    
    def series_stats(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0: return {}
        return {
            "n": int(s.shape[0]), "mean": float(s.mean()), "std": float(s.std()),
            "median": float(s.median()), "min": float(s.min()), "max": float(s.max()),
            "q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75)),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        }

    # 完整列映射
    cols_to_stat = {
        "gain_mae": "gain", "forget": "seq_ewc_forget",
        "g_mae": "global_eval_test_mae", "g_rmse": "global_eval_test_rmse", "g_r2": "global_eval_test_r2",
        "ft_mae": "seq_ft_test_mae", "ft_rmse": "seq_ft_test_rmse", "ft_r2": "seq_ft_test_r2",
        "ewc_mae": "seq_ewc_test_mae", "ewc_rmse": "seq_ewc_test_rmse", "ewc_r2": "seq_ewc_test_r2",
        "trainable_params": "trainable_params",
    }
    
    stats = {"columns": {}, "overall": {}}
    for k, c in cols_to_stat.items():
        if c in df.columns:
            stats["columns"][k] = series_stats(df[c])

    # 完整胜率与分布统计
    if "gain" in df.columns:
        improved = df["gain"] > 0
        stats["overall"].update({
            "n_users": int(len(df)), "improve_rate_mae": float(improved.mean()),
            "n_improved": int(improved.sum()), "n_degraded": int((~improved).sum())
        })

        if improved.any():
            stats["overall"]["gain_mean_improved"] = float(df.loc[improved, "gain"].mean())
            stats["overall"]["gain_median_improved"] = float(df.loc[improved, "gain"].median())
        if (~improved).any():
            stats["overall"]["gain_mean_degraded"] = float(df.loc[~improved, "gain"].mean())
            stats["overall"]["gain_median_degraded"] = float(df.loc[~improved, "gain"].median())

    # 提取表现最好/最差的用户
    if "gain" in df.columns and "user_id" in df.columns and len(df) > 0:
        best_idx, worst_idx = df["gain"].idxmax(), df["gain"].idxmin()
        for label, idx in zip(["best_user", "worst_user"], [best_idx, worst_idx]):
            stats["overall"][label] = {
                "user_id": df.loc[idx, "user_id"],
                "gain": float(df.loc[idx, "gain"]),
                "g_mae": float(df.loc[idx, "global_eval_test_mae"]),
                "ewc_mae": float(df.loc[idx, "seq_ewc_test_mae"]),
            }

    # 保存
    df.to_csv(os.path.join(output_dir, f"summary-{target}-{TimeID}.csv"), index=False)
    with open(os.path.join(output_dir, f"summary-{target}-{TimeID}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # 打印最终统计摘要
    print(f"\nUsers: {stats['overall'].get('n_users', len(df))} | Improve rate(MAE): {stats['overall'].get('improve_rate_mae', float('nan')):.3f}")
    print(f"Gain(MAE) mean±std: {stats['columns'].get('gain_mae', {}).get('mean', float('nan')):.3f} ± {stats['columns'].get('gain_mae', {}).get('std', float('nan')):.3f}")
    print(f"Gain(MAE) median[IQR]: {stats['columns'].get('gain_mae', {}).get('median', float('nan')):.3f} [{stats['columns'].get('gain_mae', {}).get('iqr', float('nan')):.3f}]")
    print(f"Forget mean±std: {stats['columns'].get('forget', {}).get('mean', float('nan')):.3f} ± {stats['columns'].get('forget', {}).get('std', float('nan')):.3f}")

if __name__ == '__main__':
    Seed(6)
    target = 'SBP'
    mat_path = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset.mat'
    pretrained_model_path = 'PTH/121517/trained_model.pth'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    TimeID = datetime.now().strftime('%H%M%S')
    
    # 初始化数据
    signals, labels, user2idx_sorted = load_and_sort_user_data(mat_path, target)
    valid_users = [u for u in user2idx_sorted.keys() if len(user2idx_sorted[u]) >= 300]
    selected_users = random.sample(valid_users, min(10, len(valid_users)))
    
    all_results = []
    modes = ['global_eval', 'seq_ft', 'seq_ewc']
    layers_to_unfreeze = UNFREEZE_PRESETS[UNFREEZE_PRESET]
    
    base_model = torch.load(pretrained_model_path, map_location="cpu")
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    print("UNFREEZE_PRESET:", UNFREEZE_PRESET, "->", layers_to_unfreeze)
    print("\nidx | user_id | P(M) | G(MAE/RMSE/R2) | FT(MAE/RMSE/R2) | EWC(MAE/RMSE/R2) | gain_mae | dR2 | forget | t(s)")

    # 批量跑实验
    for u_idx, user_id in enumerate(selected_users):
        t_user0 = time.time()
        idx_sorted = user2idx_sorted[user_id]
        train_idx, val_idx, test_idx = split_user_stream(idx_sorted)
        
        batch_loaders = [data.DataLoader(CLDataset(signals[b], labels[b]), batch_size=8, shuffle=False) for b in split_train_to_batches(train_idx, K=4)]
        val_loader  = data.DataLoader(CLDataset(signals[val_idx],  labels[val_idx]),  batch_size=8, shuffle=False)
        test_loader = data.DataLoader(CLDataset(signals[test_idx], labels[test_idx]), batch_size=8, shuffle=False)
        
        user_res = {'user_id': user_id}
        
        for mode in modes:
            model = copy.deepcopy(base_model).to(device)
            model.load_state_dict(base_state, strict=True)
            set_trainable_by_prefix(model, layers_to_unfreeze)

            if mode == modes[0]:
                user_res['trainable_params'] = count_trainable_params(model)

            Settings = {
                'BP_optimizer': "torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, betas=(0.9, 0.999), weight_decay=0)",
                'trainer': "Model_Trainer(model, torch.nn.MSELoss(), BP_optimizer, device, Settings, batch_size=8, num_epochs=20, save_states=False, save_final=False, timeid=TimeID)"
            }
            BP_optimizer = eval(Settings['BP_optimizer'])
            model_trainer = eval(Settings['trainer'])

            res = model_trainer.Train_CL_Model(
                user_id, batch_loaders, test_loader, val_loader=val_loader,
                val_check='batch', rollback_to_best=True, patience=0, 
                mode=mode, lambda_ewc=500, trainable_keywords=layers_to_unfreeze,
                verbose=0, show_progress=False
            )
            for k, v in res.items():
                if k not in ['user_id', 'mode']: user_res[f"{mode}_{k}"] = v
                
        # 计算结果差异
        user_res['gain'] = user_res.get('global_eval_test_mae', 0) - user_res.get('seq_ewc_test_mae', 0)
        g_mae, g_rmse, g_r2 = user_res.get('global_eval_test_mae', float('nan')), user_res.get('global_eval_test_rmse', float('nan')), user_res.get('global_eval_test_r2', float('nan'))
        ft_mae, ft_rmse, ft_r2 = user_res.get('seq_ft_test_mae', float('nan')), user_res.get('seq_ft_test_rmse', float('nan')), user_res.get('seq_ft_test_r2', float('nan'))
        ewc_mae, ewc_rmse, ewc_r2 = user_res.get('seq_ewc_test_mae', float('nan')), user_res.get('seq_ewc_test_rmse', float('nan')), user_res.get('seq_ewc_test_r2', float('nan'))
        
        gain_mae = g_mae - ewc_mae
        delta_r2 = ewc_r2 - g_r2
        forget = user_res.get('seq_ewc_forget', float('nan'))
        # 打印当前用户摘要
        t_user = time.time() - t_user0
        print(f"{u_idx+1:>3d} | {user_id} | {user_res.get('trainable_params', float('nan'))/1e6:>4.2f} | "
              f"{g_mae:>5.2f}/{g_rmse:>5.2f}/{g_r2:>6.3f} | "
              f"{ft_mae:>5.2f}/{ft_rmse:>5.2f}/{ft_r2:>6.3f} | "
              f"{ewc_mae:>5.2f}/{ewc_rmse:>5.2f}/{ewc_r2:>6.3f} | "
              f"{gain_mae:>8.2f} | {delta_r2:>6.3f} | {forget:>6.2f} | {t_user:>5.1f}")
        all_results.append(user_res)
        
    # 保存与统计汇总
    save_and_summarize_results(all_results, './Model_Training/CL_Results', target, TimeID)