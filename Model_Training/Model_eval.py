import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mat73 import loadmat
from Model_Def import ResNet, DDCCor

def Seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, Input, Label):
        self.Input = Input
        self.Label = Label
    def __len__(self):
        return len(self.Input)
    def __getitem__(self, idx):
        return self.Input[idx, :], self.Label[[idx]]

def Build_Dataset(Path, Label):
    Data = loadmat(Path)
    # 获取ECG和PPG信号通道
    return Dataset(Data['Subset']['Signals'][:, 0:2, :], Data['Subset'][Label])

def Evaluate_Model_checkpoint(checkpoint_path, target):
    # 根据目标选择文件和分析字段
    if target == 'SBP':
        Evaluate_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset.mat'
        xlim_range = [45, 200]
        ylim_range = [-75, 75]
        xlabel_bland = 'Average SBP (mmHg)'
        ylabel_bland = 'SBP Diff (mmHg)'
        xlabel_reg = 'Actual SBP (mmHg)'
        ylabel_reg = 'Estimated SBP (mmHg)'
        xlabel_density = 'Mean Error (SBP)'
    elif target == 'DBP':
        Evaluate_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset.mat'
        xlim_range = [20, 120]
        ylim_range = [-50, 50]
        xlabel_bland = 'Average DBP (mmHg)'
        ylabel_bland = 'DBP Diff (mmHg)'
        xlabel_reg = 'Actual DBP (mmHg)'
        ylabel_reg = 'Estimated DBP (mmHg)'
        xlabel_density = 'Mean Error (DBP)'
    else:
        raise ValueError("Invalid target. Please use 'SBP' or 'DBP'.")

    # 加载测试数据
    Test_Data = Build_Dataset(Evaluate_File, target)
    test_loader = torch.utils.data.DataLoader(Test_Data, batch_size=32, shuffle=False)

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # 恢复模型状态
    model = DDCCor.DDCCR_Net()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = model.float()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    errors = all_targets - all_predictions
    abs_errors = np.abs(errors)
    mean_values = (all_targets + all_predictions) / 2
    lower_bound = np.percentile(errors, 2.5)  # 置信区间的左边界
    upper_bound = np.percentile(errors, 97.5) # 置信区间的右边界
    slope, intercept = np.polyfit(all_targets, all_predictions, 1)  # 一阶多项式拟合
    pearson_corr = np.corrcoef(all_targets, all_predictions)[0, 1]  # Pearson相关系数

    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    me = np.mean(errors)
    sd = np.std(errors)
    cpe5 = np.sum(abs_errors < 5) / len(abs_errors) * 100
    cpe10 = np.sum(abs_errors < 10) / len(abs_errors) * 100
    cpe15 = np.sum(abs_errors < 15) / len(abs_errors) * 100

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, ME: {me:.4f}, SD: {sd:.4f}, "
          f"CPE5: {cpe5:.2f}%, CPE10: {cpe10:.2f}%, CPE15: {cpe15:.2f}%")
    
    # 可视化：Bland-Altman 图
    plt.figure(figsize=(6, 6))
    plt.scatter(mean_values, errors, c='dodgerblue', s=10, alpha=0.4)
    plt.axhline(me, color='red', linestyle='-', label='Mean')
    plt.axhline(me + 1.96 * sd, color='green', linestyle='--', label='+1.96 SD')
    plt.axhline(me - 1.96 * sd, color='green', linestyle='--', label='-1.96 SD')
    plt.xlim(xlim_range)
    plt.ylim(ylim_range)
    plt.title('Bland-Altman')
    plt.xlabel(xlabel_bland)
    plt.ylabel(ylabel_bland)
    plt.legend()
    plt.grid(False)
    plt.show()

    # 可视化：回归对比分析图
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_predictions, c='dodgerblue', s=10, alpha=0.4)
    plt.plot(all_targets, slope * np.array(all_targets) + intercept, color='black', lw=2, label=f'Fit line (r={pearson_corr:.2f})')
    plt.xlim(xlim_range)
    plt.ylim(xlim_range)
    plt.title('Regression')
    plt.xlabel(xlabel_reg)
    plt.ylabel(ylabel_reg)
    plt.legend()
    plt.grid(False)
    plt.show()

    # 可视化：密度误差图
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, errors, c='dodgerblue', s=10, alpha=0.4)
    plt.axhline(me, color='red', linestyle='-', label='Mean')
    plt.axhline(me + 1.96 * sd, color='green', linestyle='--', label='+1.96 SD')
    plt.axhline(me - 1.96 * sd, color='green', linestyle='--', label='-1.96 SD')
    plt.xlim(xlim_range)
    plt.ylim(ylim_range)
    plt.title("Density Error")
    plt.xlabel(xlabel_reg)
    plt.ylabel(ylabel_bland)
    plt.legend()
    plt.grid(False)
    plt.show()

    # 可视化：误差概率分布图
    plt.figure(figsize=(6, 6))
    sns.kdeplot(errors, fill=True, color="purple", alpha=0.4, label=f'KDE')
    plt.axvspan(lower_bound, upper_bound, color='gray', alpha=0.3, label=f'95% CI')
    plt.axvline(lower_bound, color='red', linestyle='--', label=f'{lower_bound:.2f}')
    plt.axvline(upper_bound, color='green', linestyle='--', label=f'{upper_bound:.2f}')
    plt.title('Density Error')
    plt.xlabel(xlabel_density)
    plt.ylabel('Density')
    plt.xlim(ylim_range)
    plt.legend()
    plt.grid(False)
    plt.show()

def Evaluate_Model(path, target):
    # 根据目标选择文件和分析字段
    if target == 'SBP':
        # Evaluate_File = 'D:\Data\PulseDB\FN_Subset_Files\FN_VitalDB_test_Subset.mat'
        Evaluate_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset.mat'
        xlim_range = [45, 200]
        ylim_range = [-75, 75]
        xlabel_bland = 'Average SBP (mmHg)'
        ylabel_bland = 'SBP Diff (mmHg)'
        xlabel_reg = 'Actual SBP (mmHg)'
        ylabel_reg = 'Estimated SBP (mmHg)'
        xlabel_density = 'Mean Error (SBP)'
    elif target == 'DBP':
        Evaluate_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset.mat'
        xlim_range = [20, 120]
        ylim_range = [-50, 50]
        xlabel_bland = 'Average DBP (mmHg)'
        ylabel_bland = 'DBP Diff (mmHg)'
        xlabel_reg = 'Actual DBP (mmHg)'
        ylabel_reg = 'Estimated DBP (mmHg)'
        xlabel_density = 'Mean Error (DBP)'
    else:
        raise ValueError("Invalid target. Please use 'SBP' or 'DBP'.")

    # 加载测试数据
    Test_Data = Build_Dataset(Evaluate_File, target)
    test_loader = torch.utils.data.DataLoader(Test_Data, batch_size=32, shuffle=False)

    model = torch.load(path)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = model.float()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    errors = all_targets - all_predictions
    abs_errors = np.abs(errors)
    mean_values = (all_targets + all_predictions) / 2
    lower_bound = np.percentile(errors, 2.5)  # 置信区间的左边界
    upper_bound = np.percentile(errors, 97.5)  # 置信区间的右边界
    slope, intercept = np.polyfit(all_targets, all_predictions, 1)  # 一阶多项式拟合

    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    me = np.mean(errors)
    sd = np.std(errors)
    cpe5 = np.sum(abs_errors < 5) / len(abs_errors) * 100
    cpe10 = np.sum(abs_errors < 10) / len(abs_errors) * 100
    cpe15 = np.sum(abs_errors < 15) / len(abs_errors) * 100

    count_within_CI = np.sum((errors >= (me - 1.96 * sd)) & (errors <= (me + 1.96 * sd)))
    PW_CI = (count_within_CI / len(errors)) * 100

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, ME: {me:.4f}, SD: {sd:.4f}, "
          f"CPE5: {cpe5:.2f}%, CPE10: {cpe10:.2f}%, CPE15: {cpe15:.2f}%")

    # 创建一行四个图的布局
    fig, axes = plt.subplots(1, 4, figsize=(8, 2), dpi=300)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 6

    # Bland-Altman 图
    axes[0].scatter(mean_values, errors, c='dodgerblue', s=10, alpha=0.4)
    axes[0].axhline(me, color='red', linestyle='-', label='Mean')
    axes[0].axhline(me + 1.96 * sd, color='green', linestyle='--', label='+1.96 SD')
    axes[0].axhline(me - 1.96 * sd, color='green', linestyle='--', label='-1.96 SD')
    axes[0].set_xlim(xlim_range)
    axes[0].set_ylim(ylim_range)
    axes[0].set_title(f'PW-CI: {PW_CI:.2f}%', fontsize=6, fontweight='bold')
    axes[0].set_xlabel(xlabel_bland, fontsize=6, fontweight='bold')
    axes[0].set_ylabel(ylabel_bland, fontsize=6, fontweight='bold')
    axes[0].legend()
    axes[0].tick_params(direction='in', axis='both', which='major', labelsize=5) # 设置刻度朝内
    axes[0].grid(False)

    # 回归对比分析图
    axes[1].scatter(all_targets, all_predictions, c='dodgerblue', s=10, alpha=0.4)
    axes[1].plot(all_targets, slope * np.array(all_targets) + intercept, color='red')
    axes[1].set_xlim(xlim_range)
    axes[1].set_ylim(xlim_range)
    axes[1].set_title(f'R²: {r2:.2f}', fontsize=6, fontweight='bold')
    axes[1].set_xlabel(xlabel_reg, fontsize=6, fontweight='bold')
    axes[1].set_ylabel(ylabel_reg, fontsize=6, fontweight='bold')
    axes[1].legend(frameon=False)  # 去掉图例的小框
    axes[1].tick_params(direction='in', axis='both', which='major', labelsize=5) # 设置刻度朝内
    axes[1].grid(False)

    # 密度误差图
    axes[2].scatter(all_targets, errors, c='dodgerblue', s=10, alpha=0.4)
    axes[2].axhline(me, color='red', linestyle='-', label='Mean')
    axes[2].axhline(me + 1.96 * sd, color='green', linestyle='--', label='+1.96 SD')
    axes[2].axhline(me - 1.96 * sd, color='green', linestyle='--', label='-1.96 SD')
    axes[2].set_xlim(xlim_range)
    axes[2].set_ylim(ylim_range)
    axes[2].set_title(f'Mean: {me:.2f}, SD: {sd:.2f}', fontsize=6, fontweight='bold')
    axes[2].set_xlabel(xlabel_reg, fontsize=6, fontweight='bold')
    axes[2].set_ylabel(ylabel_bland, fontsize=6, fontweight='bold')
    axes[2].legend()
    axes[2].tick_params(direction='in', axis='both', which='major', labelsize=5) # 设置刻度朝内
    axes[2].grid(False)

    # 误差概率分布图
    sns.kdeplot(errors, fill=True, color="purple", alpha=0.4, label=f'KDE', ax=axes[3])
    axes[3].axvspan(lower_bound, upper_bound, color='gray', alpha=0.3, label=f'95% CI')
    axes[3].axvline(lower_bound, color='red', linestyle='--', label=f'{lower_bound:.2f}')
    axes[3].axvline(upper_bound, color='green', linestyle='--', label=f'{upper_bound:.2f}')
    # axes[3].set_title('Density Error')
    axes[3].set_xlabel(xlabel_density, fontsize=6, fontweight='bold')
    axes[3].set_ylabel('Density', fontsize=6, fontweight='bold')
    axes[3].set_xlim(ylim_range)
    axes[3].legend()
    axes[3].tick_params(direction='in', axis='both', which='major', labelsize=5) # 设置刻度朝内
    axes[3].grid(False)

    annotations = ['(a)', '(b)', '(c)', '(d)']

    for i, ax in enumerate(axes):
        # 在左上角添加编号
        ax.text(
            -0.1, 1.1,  # 设置文本位置，坐标是以轴的坐标系为基准
            annotations[i],  # 编号文本
            transform=ax.transAxes,  # 使用轴的坐标系
            fontsize=8,  # 文本字体大小
            fontweight='bold',  # 文本字体加粗
            va='top',  # 垂直对齐到顶部
            ha='left'  # 水平对齐到左边
        )
        ax.spines['top'].set_linewidth(0.6)  # 顶部边框
        ax.spines['bottom'].set_linewidth(0.6)  # 底部边框
        ax.spines['left'].set_linewidth(0.6)  # 左侧边框
        ax.spines['right'].set_linewidth(0.6)  # 右侧边框
        
    # 调整布局
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    Seed(6)
    # Evaluate_Model('PTH/134858/trained_model.pth', 'SBP')
    Evaluate_Model('PTH/210655/trained_model.pth', 'DBP')
    # Evaluate_Model_checkpoint('PTH/122040/checkpoint_epoch_75.pth', 'SBP')