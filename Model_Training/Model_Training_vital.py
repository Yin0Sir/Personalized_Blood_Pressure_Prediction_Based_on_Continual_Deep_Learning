import torch
import random
import numpy as np
from mat73 import loadmat
import torch.utils.data as data
from Model_Def.Trainer import Model_Trainer
from Model_Def.TIM import ResNet, CorNet, DDC, DDCCor, models, DC, DCCR, DDCCR_without_mse
from Model_Def.EMBC import CF_Basic_l, CF_Basic_s, CFNet, DesCor

def Seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Dataset(data.Dataset):
    def __init__(self, Input, Label):
        self.Input = Input
        self.Label = Label
    def __len__(self):
        return len(self.Input)
    def __getitem__(self, idx):
        return self.Input[idx, :], self.Label[[idx]]

def Build_Dataset(Path, Label):
    Data = loadmat(Path)
    # 获取前两个通道，即ECG和PPG信号
    return Dataset(Data['Subset']['Signals'][:, 0:2, :], Data['Subset'][Label])
    # return Dataset(Data['Subset_new']['Signals'][:, 0:2, :], Data['Subset_new'][Label])

Train_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_Train_Subset.mat'
Test_CalBased_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset.mat'
# Test_CalFree_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset.mat'

# Train_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_Train_Subset_1000.mat'
# Test_CalBased_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset_1000.mat'

Train_Data = Build_Dataset(Train_File, 'SBP')
Test_Data = Build_Dataset(Test_CalBased_File, 'SBP')

if __name__ == '__main__':
    Seed(6)
    # model = CF_Basic_s.Resnet34_1D()
    model = DDCCR_without_mse.DDCCR_Net()
    Seed(6)
    # 准备要记录的设置
    Settings = {'BP_optimizer': 'torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0)',
                'trainer': 'Model_Trainer(model,torch.nn.MSELoss(),BP_optimizer,device,Settings,batch_size=32,num_epochs=100,save_states=True,save_final=True)'}
    # 设置训练装置
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(torch.cuda.get_device_name(0))
    model.to(device)
    # 实例化优化器和模型训练器
    BP_optimizer = eval(Settings['BP_optimizer'])
    model_trainer = eval(Settings['trainer'])
    # 设置训练集和对比下的两个设置集
    model_trainer.Set_Dataset(Train_Data, {'Test_Data': Test_Data})
    model_trainer.Train_Model()