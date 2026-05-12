# Personalized Blood Pressure Prediction Based on Continual Deep Learning

- [English](README.md) | [简体中文](README_zh.md)

This repository contains the code for the Master's thesis project "*Research on Personalized Blood Pressure Prediction Based on Continual Deep Learning*" from Southeast University. The project focuses on ECG/PPG dual-channel time series signals, builds a baseline blood pressure prediction model (see related repositories), and compares multiple continual learning personalization strategies (Fine-tune / EWC / Replay / Hybrid) on individual continuous data streams.

---

## 1. Project Objectives

- **Task Type**: Non-invasive Blood Pressure Estimation (Regression)
- **Prediction Targets**:
  - SBP (Systolic Blood Pressure)
  - DBP (Diastolic Blood Pressure)
- **Input Signals**: `Signals[:, 0:2, :]` from PulseDB subset (ECG + PPG)
- **Core Problems**:
  1. Train a generalizable global base model;
  2. Perform personalized continual learning on user time-series data streams;
  3. Compare differences in accuracy and forgetting across different continual learning strategies.

---

## 2. Method Overview

### 2.1 Model Architecture (`Model_Def/DDCCor.py`)

The main model is `DDCCR_Net()` (actually returns `CombinedCorNetResNet`):

1. **DualDynamicConv1d**:
   - Independent dynamic convolution for ECG and PPG respectively;
   - Joint dynamic convolution for dual channels;
   - Learnable weights fuse independent and joint features.
2. **ResNet1D Backbone**:
   - `BasicBlock` + 4 stages (`[3,4,6,3]`);
   - Extract high-level temporal features.
3. **CorNet Context Enhancement Module**:
   - Contextual modeling and residual correction of output distribution.
4. **Final Regression Head**:
   - `final_fc` outputs a single BP value (`num_BP=1`).

### 2.2 Continual Learning Strategies (`Model_Def/Trainer.py` + `Model_Training_CL.py`)

Sequentially learn batch-by-batch on each user's time-ordered training stream with support for:

- `global_eval`: No personalization, direct evaluation of global model;
- `seq_ft`: Sequential fine-tuning;
- `seq_ewc`: Sequential fine-tuning + EWC regularization;
- `seq_replay`: Sequential fine-tuning + Experience replay;
- `seq_hybrid`: EWC + Replay hybrid.

Records: `MAE / ME / SD / RMSE / R2`, forgetting degree `forget`, and user-level (macro) and sample-level (pooled/micro) statistics.

---

## 3. Repository Structure

```text
XXXX/
├── Model_Def/
│   ├── DDCCor.py              # Model definition: DynamicConv + ResNet + CorNet
│   ├── Trainer.py             # Trainer: standard training + continual learning + EWC/Replay
│   └── __init__.py
├── Model_Training.py          # Global model training script (base model training)
├── Fine_tune.py               # Fine-tune script after loading pretrained model
├── Model_Training_CL.py       # Main personalized continual learning experiment script
├── Model_eval.py              # Model evaluation and visualization (Bland-Altman/Regression/Error distribution)
├── model_test.py              # Model structure, parameters, FLOPs, forward test
├── CL_Results/                # Example experiment outputs (summary-*.csv/json)
└── README.md
```

---

## 4. Running Environment

Python 3.9+ is recommended, PyTorch 2.x, CUDA optional.

### 4.1 Main Dependencies

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

### 4.2 Installation Example

```bash
pip install torch numpy pandas mat73 scikit-learn progressbar2 tensorboard matplotlib seaborn torchinfo ptflops
```

---

## 5. Data Description and Format

The code reads `.mat` files by default (`mat73.loadmat`), with key fields:

- `Subset['Signals']`: Shape approximately `[N, C, L]`
  - Current code uses only the first two channels: `[:, 0:2, :]` (ECG, PPG)
- `Subset['SBP']`: Systolic blood pressure labels
- `Subset['DBP']`: Diastolic blood pressure labels
- `Subset['Subject']`, `Subset['Time']`: Used for sorting by user and time in continual learning

> Note: Scripts in the repository contain hardcoded paths (Windows/Linux); please modify data and model paths according to your local environment.

---

## 6. Training and Experiment Workflow

### 6.1 Global Base Model Training

Script: `Model_Training.py`

- Load training/test subsets;
- Build model and execute standard supervised training;
- Output checkpoint and final model.

```bash
python Model_Training.py
```

Training artifacts (saved by `Trainer.py`):

- `./<Last 6 digits of TimeID>/checkpoint_epoch_*.pth`
- `./<Last 6 digits of TimeID>/trained_model.pth`
- TensorBoard logs: `./TensorBoard/<TimeID>/...`

### 6.2 Pretrained Model Fine-tuning

Script: `Fine_tune.py`

- Load `trained_model.pth`;
- Freeze most layers, unfreeze specified layers (e.g., `final_fc`, `resnet.layer4`, `resnet.layer3`);
- Continue training on target data.

```bash
python Fine_tune.py
```

### 6.3 Personalized Continual Learning Experiments (Core)

Script: `Model_Training_CL.py`

- Aggregate by user and sort by time;
- Split training/test for each user by time (optional validation);
- Divide training stream into K sequential batches;
- Run multiple modes sequentially for each user (`global_eval/seq_ft/seq_ewc/seq_replay/seq_hybrid`);
- Save user-level summaries and overall statistics.

```bash
python Model_Training_CL.py
```

Outputs:

- `./CL_Results/summary-<target>-<TimeID>.csv`
- `./CL_Results/summary-<target>-<TimeID>.json`
- TensorBoard: `./TensorBoard/CL_Experiments/<TimeID>/<mode>/<user_id>/...`

### 6.4 Model Evaluation and Visualization

Script: `Model_eval.py`

Evaluate complete models or checkpoints and plot:

- Bland-Altman diagrams
- Regression fit plots
- Error scatter plots
- Error kernel density distribution plots

```bash
python Model_eval.py
```

### 6.5 Model Structure and Complexity Testing

Script: `model_test.py`

- Print model structure;
- View parameter statistics using `torchinfo`;
- Estimate FLOPs using `ptflops`;
- Perform forward propagation shape check.

```bash
python model_test.py
```

---

## 7. Key Metrics Description

Main metrics used in training and evaluation:

- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **ME**: Mean Error (bias)
- **SD**: Error Standard Deviation
- **R²**: Coefficient of Determination (higher is better)
- **CPE5/CPE10/CPE15** (`Model_eval.py`): Proportion where absolute error falls within 5/10/15 mmHg
- **forget** (continual learning): Degree of performance degradation in subsequent batches relative to first batch

---

## 8. Configurable Items (Check Before Running)

Before running, prioritize checking and modifying if necessary:

1. **Data Paths**:
   - `Model_Training.py`
   - `Fine_tune.py`
   - `Model_Training_CL.py`
   - `Model_eval.py`
2. **Target Type**: `SBP` / `DBP`
3. **Continual Learning Modes and Hyperparameters**:
   - `modes`
   - `lambda_ewc`
   - `buffer_size`
   - `replay_batch_size`
   - `alpha_replay`
4. **Unfreezing Strategy**: `UNFREEZE_PRESET` (e.g., `head_only`, `head_3_4`, `head_c`, `adapter`)
5. **User Sampling and Filtering**: `selected_users`, `EXCLUDED_USERS`

---

## 9. Known Considerations

1. The code currently uses external local paths by default (e.g., `D:/Data/...`), which need to be manually changed to your data paths.
2. In `Model_Training.py`, the instantiation function name is written as `DDCCor.DDCCor_Net()`, but the actual definition in the model file is `DDCCor.DDCCR_Net()`; it's recommended to unify this to `DDCCor.DDCCR_Net()` before running the script.
3. This repository does not include a unified `requirements.txt` and automated test configuration (e.g., `pytest`).

---

## 10. Recommended Reproduction Workflow

1. Install dependencies and prepare PulseDB subset `.mat` data;
2. Modify data and model paths in each script;
3. Run `model_test.py` first to verify the model can forward;
4. Run `Model_Training.py` to train the global base model;
5. Run `Fine_tune.py` to validate personalized fine-tuning effects;
6. Run `Model_Training_CL.py` for full-scale continual learning experiments;
7. Use `Model_eval.py` for graphical evaluation of selected models.

---

## Language Support

- [中文版 (Chinese)](README_zh.md)
- English

## Contact & Contribution

Issues or PRs are welcome. Please submit dataset descriptions, reproduction experiments, or feature improvement suggestions.
