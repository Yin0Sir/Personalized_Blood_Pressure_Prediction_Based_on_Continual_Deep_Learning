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
def print_current_gpu_info():
    """打印当前使用的 GPU 信息（逻辑编号、型号、物理索引）"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"当前使用 GPU: 逻辑编号 {current_device}，型号: {device_name}")
        
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            physical_index = visible_devices[current_device]
            print(f"对应物理 GPU 索引: {physical_index}")
    else:
        print("CUDA 不可用，使用 CPU")

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

import torch
import torch.nn as nn
import types

class HighPerformanceAdapter1d(nn.Module):
    def __init__(self, channels, reduction=2): # 降低reduction以提升容量
        super().__init__()
        mid_channels = max(channels // reduction, channels // 2)
        
        # 1. 特征变换路径
        self.transform = nn.Sequential(
            nn.Conv1d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels), # 引入BN稳定深层训练
            nn.GELU(),
            nn.Conv1d(mid_channels, channels, kernel_size=1, bias=False)
        )
        
        # 2. 通道注意力路径 (捕捉个体生理信号的敏感通道)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. 门控标量 (初始化为0，保证初始输出与基模完全一致)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # 将最后一层卷积权重初始化为0
        nn.init.zeros_(self.transform[-1].weight)

    def forward(self, x):
        # 提取个性化特征
        adapted_feat = self.transform(x)
        # 赋予通道注意力
        attn = self.attention(x)
        adapted_feat = adapted_feat * attn
        
        # 门控机制：自适应决定个性化特征的注入比例
        return x + self.gate * adapted_feat

def inject_ultimate_adapters_and_unfreeze_bn(model):
    from Model_Def.DDCCor import BasicBlock
    
    # 1. 动态插入 Adapter 到 Layer3 和 Layer4
    # 注意：我们这里只对深层做插入，提升精度同时控制参数量
    target_layers = [model.resnet.layer3, model.resnet.layer4]
    
    for layer in target_layers:
        for name, module in layer.named_children():
            if isinstance(module, BasicBlock):
                out_channels = module.bn2.num_features
                module.adapter = HighPerformanceAdapter1d(out_channels, reduction=2).to(next(model.parameters()).device)
                
                old_forward = module.forward
                def new_forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    
                    if self.downsample is not None:
                        identity = self.downsample(x)
                        
                    out = self.adapter(out) # 注入高性能 Adapter
                    out += identity
                    out = self.relu(out)
                    return out
                
                module.forward = types.MethodType(new_forward, module)

    # 2. 参数冻结与精确解冻策略
    # 首先冻结全局所有参数
    for p in model.parameters():
        p.requires_grad = False
        
    # 然后解冻我们需要的极致个性化部分
    for name, p in model.named_parameters():
        # (a) 解冻所有插入的 Adapter
        if "adapter" in name:
            p.requires_grad = True
        # (b) 解冻所有的 BatchNorm 层（这是生理信号提点的关键！）
        elif "bn" in name or "norm" in name or isinstance(model.get_submodule(".".join(name.split(".")[:-1])), nn.BatchNorm1d):
            p.requires_grad = True
        # (c) 解冻后端的 CorNet 上下文网络和最终分类头 (由于你有head_c配置)
        elif "cornet" in name or "final_fc" in name:
            p.requires_grad = True

    return model

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

def split_user_stream(idx_sorted, train_ratio=0.9, val_ratio=None, use_val=False):
    """
    划分用户数据流为训练/验证/测试集
    
    Args:
        idx_sorted: 时间排序后的样本索引
        train_ratio: 训练集比例（默认0.9）
        val_ratio: 验证集比例（仅当use_val=True时有效，默认None）
        use_val: 是否使用验证集（False时返回train/test; True时返回train/val/test）
    
    Returns:
        若use_val=False: (train_idx, test_idx)  # 9:1 分割
        若use_val=True:  (train_idx, val_idx, test_idx)  # 8:1:1 分割
    """
    T = len(idx_sorted)
    
    if use_val:
        if val_ratio is None:
            val_ratio = 0.1  # 默认8:1:1分割
        t_end = math.floor(train_ratio * T)
        v_end = math.floor((train_ratio + val_ratio) * T)
        return idx_sorted[:t_end], idx_sorted[t_end:v_end], idx_sorted[v_end:]
    else:
        # 9:1 分割：无验证集
        t_end = math.floor(train_ratio * T)
        return idx_sorted[:t_end], idx_sorted[t_end:]

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
    "adapter": []
}
UNFREEZE_PRESET = "head_c"  # 当前启用哪个组合

EXCLUDED_USERS = {
    "p001625_1",
    "p000205_1",
    "p003834_1",
    "p003021_1",
    "p004536_1",
    "p002998_1",
    "p004339_1",
    "p002496_1",
    "p001764_1",
    "p001092_1",
    "p005122_1",
    "p005255_1",
    "p004112_1",
    "p005934_1",
}

def set_trainable_by_prefix(model, prefixes):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(name == pref or name.startswith(pref + ".") for pref in prefixes):
            p.requires_grad = True

# 样本级总体统计（不分批次，直接拼接所有样本计算整体指标）
def _infer_on_loader(model, loader, device):
    """跑一遍 loader，返回 (y_true, y_pred) 的 1D numpy"""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            p = model(x).detach().cpu().numpy()
            ys.append(y.numpy())
            ps.append(p)
    y_true = np.concatenate(ys, axis=0).reshape(-1).astype(np.float64)
    y_pred = np.concatenate(ps, axis=0).reshape(-1).astype(np.float64)
    return y_true, y_pred

def _r2_np(y_true, y_pred):
    """不依赖 sklearn 的 R2（与 r2_score 等价定义）"""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)

def _pooled_metrics(y_true_list, y_pred_list):
    """输入若干段 y_true/y_pred，拼接后做 pooled(样本级总体) 统计"""
    if len(y_true_list) == 0:
        return {}
    y_true = np.concatenate(y_true_list).reshape(-1).astype(np.float64)
    y_pred = np.concatenate(y_pred_list).reshape(-1).astype(np.float64)
    err = y_true - y_pred

    me = float(np.mean(err))
    sd = float(np.std(err))  # ddof=0，与你 Trainer.py 的 SD() 一致
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = _r2_np(y_true, y_pred)

    return {
        "n_samples": int(err.size),
        "me": me,
        "sd": sd,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

# 4. 结果保存与统计模块
def save_and_summarize_results(all_results, output_dir, target, TimeID, pooled=None):
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

    # === 动态列映射：自适应 mode 数量与名称 ===
    # 约定：每个 mode 的指标以 "{mode}_{metric}" 命名（例如 global_eval_test_mae）, 其中 metric 常见为：test_me/test_sd/test_mae/test_rmse/test_r2/forget 等
    metric_suffixes = ["test_me", "test_sd", "test_mae", "test_rmse", "test_r2", "forget"]

    # 1) 优先使用 pooled 的 key 作为 mode 列表；否则从 df 列名中解析
    if pooled is not None and isinstance(pooled, dict) and len(pooled) > 0:
        modes_infer = list(pooled.keys())
    else:
        modes_infer = []
        if len(df.columns) > 0:
            for c in df.columns:
                for suf in metric_suffixes:
                    tag = "_" + suf
                    if c.endswith(tag):
                        modes_infer.append(c[: -len(tag)])
                        break
        # 去重保持顺序
        seen = set()
        modes_infer = [x for x in modes_infer if not (x in seen or seen.add(x))]
    
    # === [新增] 生成 gain_<mode>（以 base_mode 为参照）===
    base_mode = "global_eval" if "global_eval" in modes_infer else (modes_infer[0] if len(modes_infer) > 0 else None)
    # 只要 base 和目标 mode 都有 test_mae 列，就生成 gain_<mode> = base_mae - mode_mae
    if base_mode is not None:
        base_mae_col = f"{base_mode}_test_mae"
        if base_mae_col in df.columns:
            for m in modes_infer:
                m_mae_col = f"{m}_test_mae"
                if m_mae_col in df.columns:
                    df[f"gain_{m}"] = pd.to_numeric(df[base_mae_col], errors="coerce") - pd.to_numeric(df[m_mae_col], errors="coerce")
                else:
                    df[f"gain_{m}"] = np.nan

    # 2) 构建统计列映射：既统计通用列，也统计每个 mode 的指标列
    cols_to_stat = {
        "trainable_params": "trainable_params",
    }
    for m in modes_infer:
        gcol = f"gain_{m}"
        if gcol in df.columns:
            cols_to_stat[gcol] = gcol
    for mode in modes_infer:
        for suf in metric_suffixes:
            col = f"{mode}_{suf}"
            if col in df.columns:
                cols_to_stat[f"{mode}_{suf}"] = col

    stats = {"columns": {}, "overall": {}}
    for k, c in cols_to_stat.items():
        if c in df.columns:
            stats["columns"][k] = series_stats(df[c])

    # === [替换] 按 gain_<mode> 分别统计 improve_rate 与分布 ===
    stats["overall"].update({"n_users": int(len(df))})
    for m in modes_infer:
        gcol = f"gain_{m}"
        if gcol not in df.columns:
            continue
        g = pd.to_numeric(df[gcol], errors="coerce")
        improved = g > 0
        # 每个 mode 的提升比例（相对 base_mode 的 MAE 改善）
        stats["overall"][f"improve_rate_{m}"] = float(improved.mean()) if improved.notna().any() else float("nan")
        stats["overall"][f"n_improved_{m}"] = int(improved.sum(skipna=True))
        stats["overall"][f"n_degraded_{m}"] = int((~improved & g.notna()).sum())

    # === 对每个 gain_<mode> 提取 best/worst user ===
    if "user_id" in df.columns and len(df) > 0:
        stats["overall"]["best_worst_by_gain"] = {}
        for m in modes_infer:
            gcol = f"gain_{m}"
            if gcol not in df.columns:
                continue
            g = pd.to_numeric(df[gcol], errors="coerce")
            if g.dropna().empty:
                continue
            best_idx = g.idxmax()
            worst_idx = g.idxmin()
            stats["overall"]["best_worst_by_gain"][m] = {
                "best_user": {"user_id": df.loc[best_idx, "user_id"], "gain": float(g.loc[best_idx])},
                "worst_user": {"user_id": df.loc[worst_idx, "user_id"], "gain": float(g.loc[worst_idx])},
            }

    # 保存 [新增] pooled（样本级总体 / micro）统计：把所有用户 test 样本拼起来算
    if pooled is not None:
        stats["overall"]["pooled_test"] = {}
        for mode, buf in pooled.items():
            pm = _pooled_metrics(buf.get("y_true", []), buf.get("y_pred", []))
            if pm:
                stats["overall"]["pooled_test"][mode] = pm
    df.to_csv(os.path.join(output_dir, f"summary-{target}-{TimeID}.csv"), index=False)
    with open(os.path.join(output_dir, f"summary-{target}-{TimeID}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # 打印 用户级（macro）统计：自适应 mode； 打印 gain_<mode> 统计与 improve_rate
    for m in modes_infer:
        key = f"gain_{m}"
        if key not in stats["columns"]:
            continue
        s = stats["columns"][key]
        ir = stats["overall"].get(f"improve_rate_{m}", float("nan"))
        print(
            f"{key} (base={base_mode}): "
            f"mean±std {s.get('mean', float('nan')):.3f} ± {s.get('std', float('nan')):.3f} | "
            f"median[IQR] {s.get('median', float('nan')):.3f} [{s.get('iqr', float('nan')):.3f}] | "
            f"improve_rate {ir:.3f}"
        )
    # forget（若存在）
    for k in list(stats["columns"].keys()):
        if k.endswith("_forget"):
            s = stats["columns"][k]
            print(f"{k} mean±std: {s.get('mean', float('nan')):.3f} ± {s.get('std', float('nan')):.3f}")

    def _colstat(key):
        return stats["columns"].get(key, {})

    # 从 cols_to_stat 中解析 mode 列表（保持稳定）
    mode_list = []
    for key in cols_to_stat.keys():
        for suf in ["test_mae", "test_me", "test_sd", "test_rmse", "test_r2"]:
            tag = "_" + suf
            if key.endswith(tag):
                mode = key[: -len(tag)]
                if mode not in mode_list:
                    mode_list.append(mode)
                break

    for mode in mode_list:
        mae = _colstat(f"{mode}_test_mae")
        me  = _colstat(f"{mode}_test_me")
        sd  = _colstat(f"{mode}_test_sd")
        rmse= _colstat(f"{mode}_test_rmse")
        r2  = _colstat(f"{mode}_test_r2")
        if any(len(x) > 0 for x in [mae, me, sd, rmse, r2]):
            print(
                f"{mode}: "
                f"MAE {mae.get('mean', float('nan')):.3f}±{mae.get('std', float('nan')):.3f} | "
                f"ME {me.get('mean', float('nan')):.3f}±{me.get('std', float('nan')):.3f} | "
                f"SD {sd.get('mean', float('nan')):.3f}±{sd.get('std', float('nan')):.3f} | "
                f"RMSE {rmse.get('mean', float('nan')):.3f}±{rmse.get('std', float('nan')):.3f} | "
                f"R2 {r2.get('mean', float('nan')):.3f}±{r2.get('std', float('nan')):.3f}"
            )

    # pooled（样本级总体 / micro）统计：自适应 mode
    pooled_block = stats["overall"].get("pooled_test", {})
    if pooled_block:
        print("\nPooled test (all samples across all users) [micro]:")
        order = mode_list if len(mode_list) > 0 else list(pooled_block.keys())
        for mode in order:
            pm = pooled_block.get(mode, None)
            if pm is None:
                continue
            print(
                f"{mode}: n={pm['n_samples']} | "
                f"ME={pm['me']:.3f} | SD={pm['sd']:.3f} | RMSE={pm['rmse']:.3f} | MAE={pm['mae']:.3f} | R2={pm['r2']:.3f}"
            )

if __name__ == '__main__':
    Seed(6)
    target = 'SBP'
    if os.name == 'nt':  # Windows
        mat_path = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset.mat'
    elif os.name == 'posix':  # Linux
        mat_path = '/home/zxy233580/projects/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset.mat'
    pretrained_model_path = 'PTH/121517/trained_model.pth' if target == 'SBP' else 'PTH/233948/trained_model.pth'
    
    print_current_gpu_info()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TimeID = datetime.now().strftime('%H%M%S')
    
    # 初始化数据
    signals, labels, user2idx_sorted = load_and_sort_user_data(mat_path, target)
    valid_users = [u for u in user2idx_sorted.keys() if len(user2idx_sorted[u]) >= 300 and u not in EXCLUDED_USERS]
    selected_users = random.sample(valid_users, min(1000, len(valid_users)))
    
    all_results = []
    modes = ['global_eval', 'seq_ft', 'seq_ewc', 'seq_replay', 'seq_hybrid']
    # modes = ['global_eval', 'seq_hybrid']
    pooled = {m: {"y_true": [], "y_pred": []} for m in modes} # 用于后续整体统计的 pooled 结果容器
    layers_to_unfreeze = UNFREEZE_PRESETS[UNFREEZE_PRESET]
    
    base_model = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    print("解冻层:", UNFREEZE_PRESET, "->", layers_to_unfreeze)

    # 动态表头：自适应 modes
    header_parts = ["idx", "user_id", "P(M)"]
    header_parts += [f"{m}(MAE/ME/SD)" for m in modes]
    header_parts += [f"{m}_forget" for m in modes]
    header_parts += ["t(s)"]
    print(" | ".join(header_parts))

    # 批量跑实验
    USE_VAL = False
    for u_idx, user_id in enumerate(selected_users):
        t_user0 = time.time()
        idx_sorted = user2idx_sorted[user_id]
        
        if USE_VAL:
            train_idx, val_idx, test_idx = split_user_stream(idx_sorted, train_ratio=0.9, use_val=True)
            val_loader = data.DataLoader(CLDataset(signals[val_idx], labels[val_idx]), batch_size=8, shuffle=False)
        else:
            train_idx, test_idx = split_user_stream(idx_sorted, train_ratio=0.8, use_val=False)
            val_loader = None  # 无验证集
        
        batch_loaders = [data.DataLoader(CLDataset(signals[b], labels[b]), batch_size=8, shuffle=False) for b in split_train_to_batches(train_idx, K=4)]
        test_loader = data.DataLoader(CLDataset(signals[test_idx], labels[test_idx]), batch_size=8, shuffle=False)
        
        user_res = {'user_id': user_id}
        
        for mode in modes:
            model = copy.deepcopy(base_model).to(device)
            model.load_state_dict(base_state, strict=True)
            
            if UNFREEZE_PRESET == "adapter" or "adapter" in layers_to_unfreeze:
                inject_ultimate_adapters_and_unfreeze_bn(model)
            else:
                set_trainable_by_prefix(model, layers_to_unfreeze)

            if mode == modes[0]:
                user_res['trainable_params'] = count_trainable_params(model)

            Settings = {
                'BP_optimizer': "torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, betas=(0.9, 0.999), weight_decay=0)",
                'trainer': "Model_Trainer(model, torch.nn.MSELoss(), BP_optimizer, device, Settings, batch_size=8, num_epochs=50, save_states=False, save_final=False, timeid=TimeID)"
            }
            BP_optimizer = eval(Settings['BP_optimizer'])
            model_trainer = eval(Settings['trainer'])

            res = model_trainer.Train_CL_Model(
                user_id, batch_loaders, test_loader, val_loader=val_loader,
                val_check='epoch', rollback_to_best=True, patience=0, 
                mode=mode, lambda_ewc=500, trainable_keywords=layers_to_unfreeze,
                buffer_size=64, replay_batch_size=8, alpha_replay=1, # replay 相关参数
                verbose=0, show_progress=False
            )
            for k, v in res.items():
                if k not in ['user_id', 'mode']: user_res[f"{mode}_{k}"] = v
            # 收集 pooled(样本级总体) 的 test 预测
            y_true_m, y_pred_m = _infer_on_loader(model, test_loader, device)
            pooled[mode]["y_true"].append(y_true_m)
            pooled[mode]["y_pred"].append(y_pred_m)

        # 计算结果差异
        def _get(mode, suf, default=float('nan')):
            if mode is None:
                return default
            return user_res.get(f"{mode}_{suf}", default)

        forget_map = {m: _get(m, "forget") for m in modes}
        t_user = time.time() - t_user0

        # 动态行输出：每个 mode 打印 MAE/ME/SD（若缺失则 nan）
        row_parts = [
            f"{u_idx+1:>3d}",
            f"{user_id}",
            f"{user_res.get('trainable_params', float('nan'))/1e6:>4.2f}",
        ]

        for m in modes:
            mae = user_res.get(f"{m}_test_mae", float('nan'))
            me  = user_res.get(f"{m}_test_me", float('nan'))
            sd  = user_res.get(f"{m}_test_sd", float('nan'))
            row_parts.append(f"{mae:>5.2f}/{me:>5.2f}/{sd:>6.3f}")

        for m in modes:
            fgt = forget_map.get(m, float('nan'))
            if isinstance(fgt, (int, float, np.floating)) and np.isfinite(fgt):
                row_parts.append(f"{float(fgt):>6.2f}")
            else:
                row_parts.append(f"{float('nan'):>6.2f}")

        row_parts.append(f"{t_user:>5.1f}")
        print(" | ".join(row_parts))
        all_results.append(user_res)
        
    # 保存与统计汇总
    save_and_summarize_results(all_results, './CL_Results', target, TimeID, pooled=pooled)