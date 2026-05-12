import torch
import torch.utils.data as data
import random
import numpy as np
from mat73 import loadmat
from Model_Def.Trainer import Model_Trainer

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

Train_File = 'D:/Data/PulseDB/FN_Subset_Files/FN_MIMIC_train_Subset.mat' # MIMIC VitalDB
Test_File = 'D:/Data/PulseDB/FN_Subset_Files/FN_MIMIC_test_Subset.mat'

Train_Data = Build_Dataset(Train_File, 'DBP')
Test_Data = Build_Dataset(Test_File, 'DBP')

if __name__ == '__main__':
    Seed(6)
    Settings = {
        'BP_optimizer': "torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)",
        'trainer': "Model_Trainer(model, torch.nn.MSELoss(), BP_optimizer, device, Settings, batch_size=32, num_epochs=50, save_states=True, save_final=True)"
    }

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    pretrained_model_path = 'PTH/121517/trained_model.pth'
    print(f"Loading pretrained model from {pretrained_model_path}")
    model = torch.load(pretrained_model_path, map_location=device)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    layers_to_unfreeze = ["final_fc","resnet.layer4", "resnet.layer3"]  # 综合最好的结果
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True

    BP_optimizer = eval(Settings['BP_optimizer'])
    model_trainer = eval(Settings['trainer'])
    model_trainer.Set_Dataset(Train_Data, {'Evaluation': Test_Data})

    model_trainer.Train_Model()
