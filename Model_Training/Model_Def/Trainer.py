from datetime import datetime
import time
import torch
import torch.utils.data as data
import os
import numpy as np
import progressbar as PB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter as SW
from io import StringIO
import sys

# R-Squared
def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Mean error
def ME(y_true, y_pred):
    return np.mean(y_true-y_pred)

# Standard deviation of error
def SD(y_true, y_pred):
    return np.std(y_true-y_pred)

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

widgets = [
    PB.Bar(),
    PB.Counter(),
    ' ',
    PB.Percentage(),
    ' ',
    PB.DynamicMessage('Batch_BP_Loss'),
    ' ',
    PB.ETA()
]


class Model_Trainer:
    def __init__(self, model, criterion_BP, optimizer_BP, device, settings_yml, batch_size=32, num_epochs=100, save_states=False, save_final=False):

        self.Model_Running = model.to(device)
        self.Model_BestTest = []
        self.BP_Loss_Fun = criterion_BP
        self.Optimizer_BP = optimizer_BP
        self.Num_Epoch = num_epochs
        self.Train_Batchsize = batch_size
        self.Device = device
        self.Save_States = save_states
        self.Save_Final = save_final
        self.YMLSettings = settings_yml

    def Model_Info(self):
        model = self.Model_Running
        print('-' * 10)
        print('Model Structure:')
        print(model)
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable parameters: {}'.format(num))
        print('Settings')
        for item, setting in self.YMLSettings.items():
            print(item, ':', setting)
        print('-' * 10)

    def Set_Dataset(self, train_set, test_set=[]):
        self.Train_Set = train_set
        self.Test_Set_List = test_set
    
    def Train_Model_MultiStage(self, stage_trainsets: list, stage_testsets: list):
        TimeID = datetime.now().strftime('%Y_%m%d_%H%M%S')
        ModelID = TimeID[-6:]
        Writer = SW(os.path.join('Model_Training/TensorBoard', TimeID))
        save_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        print('ModelID: '+ModelID)
        self.Model_Info()
        sys.stdout = save_stdout
        Writer.add_text('Model', result.getvalue().replace('\n', '     \n'))
            
        print(f'========== Multi-Stage Training: Total {len(stage_trainsets)} stages ==========')
        self.Model_Info()

        for stage_id, (train_set, test_set_dict) in enumerate(zip(stage_trainsets, stage_testsets), 1):
            print(f'\n=== Stage {stage_id} Training ===')

            # 动态赋值当前阶段的训练集与测试集
            self.Train_Set = train_set
            self.Test_Set_List = test_set_dict

            # --- 下面是复制你原 Train_Model() 中的代码块 ---
            Start_Epoch = 1
            batchcounter = 1
            batchrecordcounter = 1
            
            # 为训练和测试集设置数据加载器
            Train = data.DataLoader(self.Train_Set, self.Train_Batchsize, shuffle=True, drop_last=False)
            Test_Names = []
            Test_List = []
            for name, testdata in self.Test_Set_List.items():
                Test_Names.append(name)
                Test_List.append(data.DataLoader(testdata, batch_size=128))

            Start_Time = time.time()
            # 设置键盘中断，这样当训练过程中断时，模型仍然可以保存到文件
            Interrupt = False
            Train_Batch = self.Train_Batch

            for Epoch in range(Start_Epoch, Start_Epoch+self.Num_Epoch):
                try:
                    print('Epoch {}/{}'.format(Epoch, Start_Epoch+self.Num_Epoch-1))
                    print('-' * 10)

                    # 训练阶段
                    Epoch_BP_Train_Loss = []
                    k = 0
                    Epoch_BP_Preds = []
                    Epoch_BP_Labels = []
                    with PB.ProgressBar(widgets=widgets, max_value=len(Train)) as bar:
                        for inputs, BP_labels in Train:
                            # 累积批次训练输出
                            BP_loss, BP_Outputs = Train_Batch(inputs, BP_labels)
                            Epoch_BP_Labels.append(BP_labels.cpu().detach().numpy())
                            Epoch_BP_Preds.append(BP_Outputs.cpu().detach().numpy())
                            # 进度条更新
                            bar.update(k, Batch_BP_Loss=BP_loss)
                            k += 1
                            batchcounter += 1
                            # 将每批的训练损失保存到 TensorBoard
                            if not batchcounter % 100:
                                Writer.add_scalar('Batch_BP_Loss', BP_loss, batchrecordcounter)
                                batchrecordcounter += 1  # 每次计数为100批次
                        
                        # 在每个时期结束时，计算训练集上的误差指标
                        Epoch_BP_Labels = np.concatenate(Epoch_BP_Labels, axis=0)
                        Epoch_BP_Preds = np.concatenate(Epoch_BP_Preds, axis=0)
                        Epoch_Train_R2 = R2(Epoch_BP_Labels, Epoch_BP_Preds)
                        Epoch_Train_ME = ME(Epoch_BP_Labels, Epoch_BP_Preds)
                        Epoch_Train_SD = SD(Epoch_BP_Labels, Epoch_BP_Preds)
                        Epoch_Train_RMSE = RMSE(Epoch_BP_Labels, Epoch_BP_Preds)
                        Epoch_Train_MAE = MAE(Epoch_BP_Labels, Epoch_BP_Preds)

                        # 计算训练损失
                        Epoch_BP_Train_Loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_BP_Labels), torch.from_numpy(Epoch_BP_Preds))
                        
                        # 打印训练错误指标的摘要
                        print('Epoch BP Training Loss: {:e} R2: {}'.format(Epoch_BP_Train_Loss, Epoch_Train_R2))
                        
                        # 保存检查点
                        if self.Save_States:
                            self.Save_Checkpoint(ModelID, TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=False)

                    # 将训练结果写入 TensorBoard
                    Writer_Loss_Dict = {'Train_BP': Epoch_BP_Train_Loss}
                    Writer_R2_Dict = {'Train': Epoch_Train_R2}
                    Writer_ME_Dict = {'Train': Epoch_Train_ME}
                    Writer_SD_Dict = {'Train': Epoch_Train_SD}
                    Writer_RMSE_Dict = {'Train': Epoch_Train_RMSE}
                    Writer_MAE_Dict = {'Train': Epoch_Train_MAE}

                    # 测试阶段
                    # 在每个测试集上使用当前模型运行测试
                    for name, Test in zip(Test_Names, Test_List):
                        Test_Name = name
                        Epoch_Test_Loss = []
                        Epoch_Preds = []
                        Epoch_Labels = []
                        # 批量累积预测
                        for inputs, labels in Test:
                            Loss_Per_Batch, Outputs = self.Test_Batch(inputs, labels)
                            Epoch_Test_Loss.append(Loss_Per_Batch)
                            Epoch_Labels.append(labels.cpu().detach().numpy())
                            Epoch_Preds.append(Outputs.cpu().detach().numpy())
                        # 计算误差指标
                        Epoch_Labels = np.concatenate(Epoch_Labels, axis=0)
                        Epoch_Preds = np.concatenate(Epoch_Preds, axis=0)

                        Epoch_Test_Loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_Labels), torch.from_numpy(Epoch_Preds))
                        
                        Epoch_Test_R2 = R2(Epoch_Labels, Epoch_Preds)
                        Epoch_Test_ME = ME(Epoch_Labels, Epoch_Preds)
                        Epoch_Test_SD = SD(Epoch_Labels, Epoch_Preds)
                        Epoch_Test_RMSE = RMSE(Epoch_Labels, Epoch_Preds)
                        Epoch_Test_MAE = MAE(Epoch_Labels, Epoch_Preds)
                        
                        # 打印摘要
                        print('Epoch '+Test_Name + ' Loss: {:e}, R2: {}, ME: {}, SD: {}, RMSE: {}, MAE: {}'.format(Epoch_Test_Loss, Epoch_Test_R2, Epoch_Test_ME, Epoch_Test_SD, Epoch_Test_RMSE, Epoch_Test_MAE))
                        
                        # 写入 TensorBoard
                        Writer_Loss_Dict.update({Test_Name: Epoch_Test_Loss})
                        Writer_R2_Dict.update({Test_Name: Epoch_Test_R2})
                        Writer_ME_Dict.update({Test_Name: Epoch_Test_ME})
                        Writer_SD_Dict.update({Test_Name: Epoch_Test_SD})
                        Writer_RMSE_Dict.update({Test_Name: Epoch_Test_RMSE})
                        Writer_MAE_Dict.update({Test_Name: Epoch_Test_MAE})

                    Writer.add_scalars('Loss', Writer_Loss_Dict, Epoch)
                    Writer.add_scalars('R2', Writer_R2_Dict, Epoch)
                    Writer.add_scalars('ME', Writer_ME_Dict, Epoch)
                    Writer.add_scalars('SD', Writer_SD_Dict, Epoch)
                    Writer.add_scalars('RMSE', Writer_RMSE_Dict, Epoch)
                    Writer.add_scalars('MAE', Writer_MAE_Dict, Epoch)
                
                # 如果训练因键盘中断而手动停止
                except KeyboardInterrupt:
                    print('Earlystopped by interrupt at epoch {:d}'.format(Epoch))
                    Interrupt = True
                    break
            Writer.close()
            time_elapsed = time.time() - Start_Time
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            # 保存每一阶段模型
            self.Save_Checkpoint(f"{ModelID}_stage{stage_id}", TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=True)
            if Interrupt:
                raise KeyboardInterrupt
            
        Writer.close()
        print('Multi-Stage Training complete')

    def Train_Model(self):
        TimeID = datetime.now().strftime('%Y_%m%d_%H%M%S')
        ModelID = TimeID[-6:]
        
        Start_Epoch = 1
        batchcounter = 1
        batchrecordcounter = 1
  
        # Print model info to command line
        Writer = SW(os.path.join('Model_Training/TensorBoard', TimeID))
        print('ModelID: '+ModelID)
        self.Model_Info()
        # 将模型信息打印为字符串，以便可以在 TensorBoard 中记录
        
        save_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        print('ModelID: '+ModelID)
        self.Model_Info()
        sys.stdout = save_stdout
        Writer.add_text('Model', result.getvalue().replace('\n', '     \n'))
        
        # 为训练和测试集设置数据加载器
        Train = data.DataLoader(self.Train_Set, self.Train_Batchsize, shuffle=True, drop_last=False)
        Test_Names = []
        Test_List = []
        for name, testdata in self.Test_Set_List.items():
            Test_Names.append(name)
            Test_List.append(data.DataLoader(testdata, batch_size=128))

        Start_Time = time.time()
        # 设置键盘中断，这样当训练过程中断时，模型仍然可以保存到文件
        Interrupt = False

        Train_Batch = self.Train_Batch

        for Epoch in range(Start_Epoch, Start_Epoch+self.Num_Epoch):
            try:
                print('Epoch {}/{}'.format(Epoch, Start_Epoch+self.Num_Epoch-1))
                print('-' * 10)

                # 训练阶段
                Epoch_BP_Train_Loss = []
                k = 0
                Epoch_BP_Preds = []
                Epoch_BP_Labels = []
                with PB.ProgressBar(widgets=widgets, max_value=len(Train)) as bar:
                    for inputs, BP_labels in Train:
                        # 累积批次训练输出
                        BP_loss, BP_Outputs = Train_Batch(inputs, BP_labels)
                        Epoch_BP_Labels.append(BP_labels.cpu().detach().numpy())
                        Epoch_BP_Preds.append(BP_Outputs.cpu().detach().numpy())
                        # 进度条更新
                        bar.update(k, Batch_BP_Loss=BP_loss)
                        k += 1
                        batchcounter += 1
                        # 将每批的训练损失保存到 TensorBoard
                        if not batchcounter % 100:
                            Writer.add_scalar('Batch_BP_Loss', BP_loss, batchrecordcounter)
                            batchrecordcounter += 1  # 每次计数为100批次
                    
                    # 在每个时期结束时，计算训练集上的误差指标
                    Epoch_BP_Labels = np.concatenate(Epoch_BP_Labels, axis=0)
                    Epoch_BP_Preds = np.concatenate(Epoch_BP_Preds, axis=0)
                    Epoch_Train_R2 = R2(Epoch_BP_Labels, Epoch_BP_Preds)
                    Epoch_Train_ME = ME(Epoch_BP_Labels, Epoch_BP_Preds)
                    Epoch_Train_SD = SD(Epoch_BP_Labels, Epoch_BP_Preds)
                    Epoch_Train_RMSE = RMSE(Epoch_BP_Labels, Epoch_BP_Preds)
                    Epoch_Train_MAE = MAE(Epoch_BP_Labels, Epoch_BP_Preds)

                    # 计算训练损失
                    Epoch_BP_Train_Loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_BP_Labels), torch.from_numpy(Epoch_BP_Preds))
                    
                    # 打印训练错误指标的摘要
                    print('Epoch BP Training Loss: {:e} R2: {}'.format(Epoch_BP_Train_Loss, Epoch_Train_R2))
                    
                    # 保存检查点
                    if self.Save_States:
                        self.Save_Checkpoint(ModelID, TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=False)

                # 将训练结果写入 TensorBoard
                Writer_Loss_Dict = {'Train_BP': Epoch_BP_Train_Loss}
                Writer_R2_Dict = {'Train': Epoch_Train_R2}
                Writer_ME_Dict = {'Train': Epoch_Train_ME}
                Writer_SD_Dict = {'Train': Epoch_Train_SD}
                Writer_RMSE_Dict = {'Train': Epoch_Train_RMSE}
                Writer_MAE_Dict = {'Train': Epoch_Train_MAE}

                # 测试阶段
                # 在每个测试集上使用当前模型运行测试
                for name, Test in zip(Test_Names, Test_List):
                    Test_Name = name
                    Epoch_Test_Loss = []
                    Epoch_Preds = []
                    Epoch_Labels = []
                    # 批量累积预测
                    for inputs, labels in Test:
                        Loss_Per_Batch, Outputs = self.Test_Batch(inputs, labels)
                        Epoch_Test_Loss.append(Loss_Per_Batch)
                        Epoch_Labels.append(labels.cpu().detach().numpy())
                        Epoch_Preds.append(Outputs.cpu().detach().numpy())
                    # 计算误差指标
                    Epoch_Labels = np.concatenate(Epoch_Labels, axis=0)
                    Epoch_Preds = np.concatenate(Epoch_Preds, axis=0)

                    Epoch_Test_Loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_Labels), torch.from_numpy(Epoch_Preds))
                    
                    Epoch_Test_R2 = R2(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_ME = ME(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_SD = SD(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_RMSE = RMSE(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_MAE = MAE(Epoch_Labels, Epoch_Preds)
                    
                    # 打印摘要
                    print('Epoch '+Test_Name + ' Loss: {:e}, R2: {}, ME: {}, SD: {}, RMSE: {}, MAE: {}'.format(Epoch_Test_Loss, Epoch_Test_R2, Epoch_Test_ME, Epoch_Test_SD, Epoch_Test_RMSE, Epoch_Test_MAE))
                    
                    # 写入 TensorBoard
                    Writer_Loss_Dict.update({Test_Name: Epoch_Test_Loss})
                    Writer_R2_Dict.update({Test_Name: Epoch_Test_R2})
                    Writer_ME_Dict.update({Test_Name: Epoch_Test_ME})
                    Writer_SD_Dict.update({Test_Name: Epoch_Test_SD})
                    Writer_RMSE_Dict.update({Test_Name: Epoch_Test_RMSE})
                    Writer_MAE_Dict.update({Test_Name: Epoch_Test_MAE})

                Writer.add_scalars('Loss', Writer_Loss_Dict, Epoch)
                Writer.add_scalars('R2', Writer_R2_Dict, Epoch)
                Writer.add_scalars('ME', Writer_ME_Dict, Epoch)
                Writer.add_scalars('SD', Writer_SD_Dict, Epoch)
                Writer.add_scalars('RMSE', Writer_RMSE_Dict, Epoch)
                Writer.add_scalars('MAE', Writer_MAE_Dict, Epoch)
               
            # 如果训练因键盘中断而手动停止
            except KeyboardInterrupt:
                print('Earlystopped by interrupt at epoch {:d}'.format(Epoch))
                Interrupt = True
                break
        Writer.close()
        time_elapsed = time.time() - Start_Time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # 保存模型
        self.Save_Checkpoint(ModelID, TimeID, Epoch,batchcounter, batchrecordcounter, savemodel=True)
        if Interrupt:
            raise KeyboardInterrupt

    # 训练模式下批次的前向传播
    def Train_Batch(self, inputs, BP_labels):
        self.Model_Running.train()
        inputs = inputs.float().to(self.Device)
        BP_labels = BP_labels.float().to(self.Device)

        self.Model_Running.zero_grad()
        BP_outputs = self.Model_Running(inputs)
        BP_loss = self.BP_Loss_Fun(BP_outputs, BP_labels)
        BP_loss_report = BP_loss.item()
        BP_loss.backward()
        self.Optimizer_BP.step()

        return BP_loss_report, BP_outputs
    
    # 测试模式下批次的前向传播（推理）
    def Test_Batch(self, inputs, labels):

        self.Model_Running.eval()
        inputs = inputs.float().to(self.Device)
        labels = labels.float().to(self.Device)
        with torch.no_grad():
            # 测试时，仅显示重建损失
            BP_outputs = self.Model_Running(inputs)
            loss = self.BP_Loss_Fun(BP_outputs, labels)
        return loss.item(), BP_outputs
    
    # 保存检查点
    def Save_Checkpoint(self, modelID, timeID, epoch, batchcounter, batchrecordcounter, savemodel=False):
        # 为每个纪元保存一个字典
        foldername = modelID
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
        torch.save({'model_id': modelID,
                    'time_id': timeID,
                    'epoch': epoch,
                    'model_state_dict': self.Model_Running.state_dict(),
                    'optimizer_state_dict': self.Optimizer_BP.state_dict(),
                    'batchcounter': batchcounter,
                    'batchrecordcounter': batchrecordcounter,
                    }, os.path.join(foldername, 'checkpoint_epoch_{}.pth'.format(epoch)))
        if savemodel:
            torch.save(self.Model_Running, os.path.join(foldername, 'trained_model.pth'))
