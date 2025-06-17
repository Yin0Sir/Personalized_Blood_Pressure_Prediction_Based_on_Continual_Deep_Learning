import torch
import random
import numpy as np
from mat73 import loadmat
import torch.utils.data as data
from Model_Def.Trainer import Model_Trainer
from Model_Def.EMBC import CF_Basic_s

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
    return Dataset(Data['Subset']['Signals'][:, 0:2, :], Data['Subset'][Label])

Train_File1 = 'D:/Data/PulseDB/Supplementary_Subset_Files/Data_Task1.mat'
Train_File2 = 'D:/Data/PulseDB/Supplementary_Subset_Files/Data_Task2.mat'
Train_File3 = 'D:/Data/PulseDB/Supplementary_Subset_Files/Data_Task3.mat'
Train_File4 = 'D:/Data/PulseDB/Supplementary_Subset_Files/Data_Task4.mat'
Test_File = 'D:/Data/PulseDB/Supplementary_Subset_Files/Data_Task5.mat'

Train_Data1 = Build_Dataset(Train_File1, 'DBP')
Train_Data2 = Build_Dataset(Train_File2, 'DBP')
Train_Data3 = Build_Dataset(Train_File3, 'DBP')
Train_Data4 = Build_Dataset(Train_File4, 'DBP')
Test_Data = Build_Dataset(Test_File, 'DBP')

if __name__ == '__main__':
    Seed(6)
    model = CF_Basic_s.Resnet34_1D()
    Settings = {'BP_optimizer': 'torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)',
                'trainer': 'Model_Trainer(model,torch.nn.MSELoss(),BP_optimizer,device,Settings,batch_size=32,num_epochs=40,save_states=True,save_final=True)'}

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(torch.cuda.get_device_name(0))
    model.to(device)
    # 实例化优化器和模型训练器
    BP_optimizer = eval(Settings['BP_optimizer'])
    model_trainer = eval(Settings['trainer'])

    # 构造阶段数据列表
    stage_trainsets = [Train_Data1, Train_Data2, Train_Data3, Train_Data4]
    stage_testsets = [
    {"Test_Data": Test_Data},
    {"Test_Data": Test_Data},
    {"Test_Data": Test_Data},
    {"Test_Data": Test_Data}]

    # 调用多阶段训练方法
    model_trainer.Train_Model_MultiStage(stage_trainsets, stage_testsets)
