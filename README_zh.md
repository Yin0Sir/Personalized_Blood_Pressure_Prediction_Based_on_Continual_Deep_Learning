# 基于持续深度学习的个性化血压预测研究

- [English](README.md) | [简体中文](README_zh.md)

本仓库为东南大学硕士课题《**基于持续深度学习的个性化血压预测研究**》的项目代码。项目围绕 ECG/PPG 双通道时序信号，构建基线血压预测模型（本人其他仓库的项目），并在个体连续数据流上比较多种持续学习个性化策略（Fine-tune / EWC / Replay / Hybrid）。

---

## 1. 项目目标

- **任务类型**：无创血压估计（回归）
- **预测目标**：
  - SBP（收缩压）
  - DBP（舒张压）
- **输入信号**：PulseDB 子集中的 `Signals[:, 0:2, :]`（ECG + PPG）
- **核心问题**：
  1. 训练可泛化的全局基模；
  2. 在用户时序数据流上进行个性化持续学习；
  3. 比较不同持续学习策略在精度与遗忘上的差异。

---

## 2. 方法概览

### 2.1 模型结构（`Model_Def/DDCCor.py`）

主模型为 `DDCCR_Net()`（实际返回 `CombinedCorNetResNet`）：

1. **DualDynamicConv1d**：
   - 对 ECG、PPG 分别做独立动态卷积；
   - 同时对双通道做联合动态卷积；
   - 用可学习权重融合独立特征与联合特征。
2. **ResNet1D Backbone**：
   - `BasicBlock` + 4 个 stage（`[3,4,6,3]`）；
   - 提取时序高层特征。
3. **CorNet 上下文增强模块**：
   - 对输出分布做上下文建模和残差校正。
4. **最终回归头**：
   - `final_fc` 输出单个 BP 值（`num_BP=1`）。

### 2.2 持续学习策略（`Model_Def/Trainer.py` + `Model_Training_CL.py`）

在每个用户时间排序后的训练流上分 batch 顺序学习，支持：

- `global_eval`：不个性化，直接评估全局模型；
- `seq_ft`：顺序微调；
- `seq_ewc`：顺序微调 + EWC 正则；
- `seq_replay`：顺序微调 + 经验回放；
- `seq_hybrid`：EWC + Replay 混合。

并记录：`MAE / ME / SD / RMSE / R2`、遗忘度 `forget`、用户级（macro）与样本级（pooled/micro）统计。

---

## 3. 仓库结构

```text
XXXX/
├── Model_Def/
│   ├── DDCCor.py              # 模型定义：DynamicConv + ResNet + CorNet
│   ├── Trainer.py             # 训练器：常规训练 + 持续学习训练 + EWC/Replay
│   └── __init__.py
├── Model_Training.py          # 全局模型训练脚本（基模训练）
├── Fine_tune.py               # 载入预训练模型后微调
├── Model_Training_CL.py       # 个性化持续学习主实验脚本
├── Model_eval.py              # 模型评估与可视化（Bland-Altman/回归/误差分布）
├── model_test.py              # 模型结构、参数量、FLOPs、前向测试
├── CL_Results/                # 示例实验输出（summary-*.csv/json）
└── README.md
```

---

## 4. 运行环境

建议 Python 3.9+，PyTorch 2.x，CUDA 可选。

### 4.1 主要依赖

- `torch`
- `numpy`
- `pandas`
- `mat73`
- `scikit-learn`
- `progressbar2`
- `tensorboard`
- `matplotlib`
- `seaborn`
- `torchinfo`
- `ptflops`

### 4.2 安装示例

```bash
pip install torch numpy pandas mat73 scikit-learn progressbar2 tensorboard matplotlib seaborn torchinfo ptflops
```

---

## 5. 数据说明与格式

代码默认读取 `.mat` 文件（`mat73.loadmat`），关键字段为：

- `Subset['Signals']`：形状近似为 `[N, C, L]`
  - 当前代码仅使用前两通道：`[:, 0:2, :]`（ECG, PPG）
- `Subset['SBP']`：收缩压标签
- `Subset['DBP']`：舒张压标签
- `Subset['Subject']`、`Subset['Time']`：持续学习中用于按用户和时间排序

> 注意：仓库中的脚本包含硬编码路径（Windows/Linux），请根据本地环境修改数据与模型路径。

---

## 6. 训练与实验流程

### 6.1 全局基模训练

脚本：`Model_Training.py`

- 加载训练/测试子集；
- 构建模型并执行常规监督训练；
- 输出 checkpoint 与最终模型。

```bash
python Model_Training.py
```

训练产物（由 `Trainer.py` 保存）：

- `./<TimeID后6位>/checkpoint_epoch_*.pth`
- `./<TimeID后6位>/trained_model.pth`
- TensorBoard 日志：`./TensorBoard/<TimeID>/...`

### 6.2 预训练模型微调

脚本：`Fine_tune.py`

- 载入 `trained_model.pth`；
- 冻结大部分层，只解冻指定层（如 `final_fc`, `resnet.layer4`, `resnet.layer3`）；
- 在目标数据上继续训练。

```bash
python Fine_tune.py
```

### 6.3 个性化持续学习实验（核心）

脚本：`Model_Training_CL.py`

- 按用户聚合并按时间排序；
- 每位用户按时间切分训练/测试（可选验证）；
- 将训练流分成 K 个顺序批次；
- 对每位用户依次运行多种模式（`global_eval/seq_ft/seq_ewc/seq_replay/seq_hybrid`）；
- 保存用户级汇总与总体统计。

```bash
python Model_Training_CL.py
```

输出：

- `./CL_Results/summary-<target>-<TimeID>.csv`
- `./CL_Results/summary-<target>-<TimeID>.json`
- TensorBoard：`./TensorBoard/CL_Experiments/<TimeID>/<mode>/<user_id>/...`

### 6.4 模型评估与可视化

脚本：`Model_eval.py`

可评估完整模型或 checkpoint，并绘制：

- Bland-Altman 图
- 回归拟合图
- 误差散点图
- 误差核密度分布图

```bash
python Model_eval.py
```

### 6.5 模型结构与复杂度测试

脚本：`model_test.py`

- 打印模型结构；
- 使用 `torchinfo` 查看参数统计；
- 使用 `ptflops` 估算 FLOPs；
- 执行前向传播形状检查。

```bash
python model_test.py
```

---

## 7. 关键指标说明

训练与评估中主要使用：

- **MAE**：平均绝对误差（越低越好）
- **RMSE**：均方根误差（越低越好）
- **ME**：平均误差（偏差）
- **SD**：误差标准差
- **R²**：拟合优度（越高越好）
- **CPE5/CPE10/CPE15**（`Model_eval.py`）：绝对误差落在 5/10/15 mmHg 内的比例
- **forget**（持续学习）：后续批次相对首批性能退化程度

---

## 8. 可配置项（建议优先检查）

在运行前请优先检查并按需修改：

1. **数据路径**：
   - `Model_Training.py`
   - `Fine_tune.py`
   - `Model_Training_CL.py`
   - `Model_eval.py`
2. **目标类型**：`SBP` / `DBP`
3. **持续学习模式与超参数**：
   - `modes`
   - `lambda_ewc`
   - `buffer_size`
   - `replay_batch_size`
   - `alpha_replay`
4. **解冻策略**：`UNFREEZE_PRESET`（如 `head_only`, `head_3_4`, `head_c`, `adapter`）
5. **用户采样与过滤**：`selected_users`、`EXCLUDED_USERS`

---

## 9. 已知注意事项

1. 代码当前默认外部本地路径（如 `D:/Data/...`），需要手动改为你的数据路径。
2. `Model_Training.py` 中实例化函数名写为 `DDCCor.DDCCor_Net()`，而模型文件中实际定义为 `DDCCor.DDCCR_Net()`；运行该脚本前建议统一为 `DDCCor.DDCCR_Net()`。
3. 本仓库未内置统一的 `requirements.txt` 与自动化测试配置（如 `pytest`）。

---

## 10. 复现建议流程

1. 安装依赖并准备 PulseDB 子集 `.mat` 数据；
2. 修改各脚本中的数据路径与模型路径；
3. 先运行 `model_test.py` 检查模型可前向；
4. 运行 `Model_Training.py` 训练全局基模；
5. 运行 `Fine_tune.py` 验证个性化微调效果；
6. 运行 `Model_Training_CL.py` 进行持续学习全量实验；
7. 用 `Model_eval.py` 对选定模型进行图形化评估。

## 语言切换

- [中文版 (Chinese)](README_zh.md)
- [English](README.md)

## 联系与贡献

欢迎提交 Issue 或 PR。请提交数据集描述、复现实验或功能改进建议。
