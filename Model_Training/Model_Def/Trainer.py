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

    # 针对 DataLoader 维度的整体评估 (复用你原有的指标计算)
    def Evaluate_Loader(self, loader):
        self.Model_Running.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.float().to(self.Device)
                preds = self.Model_Running(inputs)
                all_preds.append(preds.cpu().detach().numpy())
                all_labels.append(labels.numpy())
                
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算各种指标
        loss = self.BP_Loss_Fun(torch.from_numpy(all_preds), torch.from_numpy(all_labels)).item()
        r2 = R2(all_labels, all_preds)
        me = ME(all_labels, all_preds)
        sd = SD(all_labels, all_preds)
        rmse = RMSE(all_labels, all_preds)
        mae = MAE(all_labels, all_preds)
        return loss, r2, me, sd, rmse, mae

    # 持续学习/EWC 主训练范式
    def Train_CL_Model(self, user_id, batch_loaders, test_loader, mode='seq_ewc', lambda_ewc=1e-3, head_keywords=["final_fc"],
        verbose=0,            # 0: 不打印; 1: 仅打印开始/结束; 2: 打印每个CL batch摘要
        show_progress=False,  # True 才显示 progressbar
):
        TimeID = datetime.now().strftime('%Y_%m%d_%H%M%S')

        # TensorBoard 仍然保留（不影响控制台）
        Writer = SW(os.path.join('Model_Training/TensorBoard/CL_Experiments', TimeID + f"_{mode}"))

        if verbose >= 1:
            print(f"[{user_id}] {mode} ...")

        result_dict = {'user_id': user_id, 'mode': mode}

        # 1) global_eval：只评估一次
        if mode == 'global_eval':
            _, r2, _, _, rmse, mae = self.Evaluate_Loader(test_loader)
            result_dict.update({'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2})
            if verbose >= 1:
                print(f"[{user_id}] {mode} done | MAE {mae:.4f} RMSE {rmse:.4f} R2 {r2:.4f}")
            Writer.close()
            return result_dict

        # 2) seq_ft / seq_ewc：顺序训练
        filter_fn = lambda name: any(kw in name for kw in head_keywords)
        ewc = EWCRegularizer(self.Model_Running, filter_fn, self.Device)
        err_b1_history = []
        global_step = 1

        for k, loader in enumerate(batch_loaders):
            # 统计该 CL batch 的平均训练 loss（不刷进度条）
            batch_losses = []

            for Epoch in range(1, self.Num_Epoch + 1):  # Num_Epoch 视作 epochs_per_batch
                self.Model_Running.train()

                if show_progress:
                    with PB.ProgressBar(widgets=widgets, max_value=len(loader)) as bar:
                        for i, (inputs, labels) in enumerate(loader):
                            inputs, labels = inputs.float().to(self.Device), labels.float().to(self.Device)

                            self.Optimizer_BP.zero_grad()
                            outputs = self.Model_Running(inputs)
                            loss = self.BP_Loss_Fun(outputs, labels)

                            if mode == 'seq_ewc' and k > 0:
                                loss += lambda_ewc * ewc.ewc_loss()

                            loss.backward()
                            self.Optimizer_BP.step()

                            batch_losses.append(loss.item())

                            if not global_step % 5:
                                Writer.add_scalar(f'CL_Loss/Batch_{k+1}', loss.item(), global_step)
                            global_step += 1

                            bar.update(i, Batch_BP_Loss=loss.item())
                else:
                    for (inputs, labels) in loader:
                        inputs, labels = inputs.float().to(self.Device), labels.float().to(self.Device)

                        self.Optimizer_BP.zero_grad()
                        outputs = self.Model_Running(inputs)
                        loss = self.BP_Loss_Fun(outputs, labels)

                        if mode == 'seq_ewc' and k > 0:
                            loss += lambda_ewc * ewc.ewc_loss()

                        loss.backward()
                        self.Optimizer_BP.step()

                        batch_losses.append(loss.item())

                        if not global_step % 5:
                            Writer.add_scalar(f'CL_Loss/Batch_{k+1}', loss.item(), global_step)
                        global_step += 1

            # 每个 CL batch 结束：评估 batch1 MAE 作为遗忘监控
            _, _, _, _, _, b1_mae = self.Evaluate_Loader(batch_loaders[0])
            err_b1_history.append(b1_mae)

            # EWC：每段后更新快照+Fisher
            if mode == 'seq_ewc':
                ewc.consolidate()
                ewc.estimate_fisher(loader, self.BP_Loss_Fun)

            if verbose >= 2:
                avg_loss = float(np.mean(batch_losses)) if len(batch_losses) else 0.0
                print(f"  - CL batch {k+1}/{len(batch_loaders)} | avg_loss {avg_loss:.3f} | b1_mae {b1_mae:.3f}")

        # 3) 最终 test 评估
        _, test_r2, _, _, test_rmse, test_mae = self.Evaluate_Loader(test_loader)
        result_dict.update({'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2})

        # 遗忘度：最后一次b1_mae - 第一次b1_mae
        result_dict['forget'] = (err_b1_history[-1] - err_b1_history[0]) if len(err_b1_history) >= 2 else 0.0

        if verbose >= 1:
            print(f"[{user_id}] {mode} done | MAE {test_mae:.4f} | Forget {result_dict['forget']:.4f}")

        Writer.close()
        return result_dict

# 持续学习 EWC 正则化器
class EWCRegularizer:
    def __init__(self, model, param_filter_fn, device='cuda'):
        self.model = model
        self.param_filter_fn = param_filter_fn
        self.device = device
        self.theta_star = {}
        self.fisher = {}

    def consolidate(self):
        """保存当前模型参数的快照"""
        self.theta_star = {}
        for name, param in self.model.named_parameters():
            if self.param_filter_fn(name) and param.requires_grad:
                self.theta_star[name] = param.data.clone().detach()

    def estimate_fisher(self, dataloader, loss_fn, max_batches=50):
        """估计 Fisher 信息矩阵"""
        self.fisher = {}
        for name, param in self.model.named_parameters():
            if self.param_filter_fn(name) and param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data).to(self.device)

        self.model.train()
        batch_count = 0
        for inputs, targets in dataloader:
            if batch_count >= max_batches: break
            inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if self.param_filter_fn(name) and param.requires_grad:
                    if param.grad is not None:
                        self.fisher[name] += param.grad.data ** 2
            batch_count += 1
            
        for name in self.fisher:
            self.fisher[name] /= float(batch_count if batch_count > 0 else 1)
        self.model.zero_grad()

    def ewc_loss(self):
        """计算 EWC 正则化损失"""
        loss = 0.0
        for name, param in self.model.named_parameters():
            if self.param_filter_fn(name) and param.requires_grad:
                if name in self.fisher and name in self.theta_star:
                    loss += torch.sum(self.fisher[name] * (param - self.theta_star[name]) ** 2)
        return loss