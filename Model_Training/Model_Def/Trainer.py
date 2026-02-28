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

def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def ME(y_true, y_pred):
    return np.mean(y_true-y_pred)

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
    def __init__(self, model, criterion_BP, optimizer_BP, device, settings_yml, batch_size=32, num_epochs=100, save_states=False, save_final=False, timeid=None):

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
        self.TimeID = timeid if timeid is not None else datetime.now().strftime('%H%M%S')

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
        TimeID = self.TimeID
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
                    # loss expects (preds, labels)
                    Epoch_BP_Train_Loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_BP_Preds), torch.from_numpy(Epoch_BP_Labels))
                    
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

                    # loss expects (preds, labels)
                    Epoch_Test_Loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_Preds), torch.from_numpy(Epoch_Labels))
                    
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
        
        loss = self.BP_Loss_Fun(torch.from_numpy(all_preds), torch.from_numpy(all_labels)).item()
        r2 = R2(all_labels, all_preds)
        me = ME(all_labels, all_preds)
        sd = SD(all_labels, all_preds)
        rmse = RMSE(all_labels, all_preds)
        mae = MAE(all_labels, all_preds)
        return loss, r2, me, sd, rmse, mae

    # 持续学习 EWC 主训练范式
    def Train_CL_Model(self, user_id, batch_loaders, test_loader, val_loader=None,
        mode='seq_ewc', lambda_ewc=1e-3, trainable_keywords=None, head_keywords=None,
        val_check='batch',         # ✅ 'batch' 或 'epoch'
        rollback_to_best=True,     # ✅ 是否回滚到val最佳
        patience=0,                # ✅ 0=不早停；>0支持早停
        min_delta=0.0,             # ✅ val改进阈值
        verbose=0,            # 0: 不打印; 1: 仅打印开始/结束; 2: 打印每个CL batch摘要
        show_progress=False,  # True 才显示 progressbar
):
        TimeID = self.TimeID
        Writer = SW(os.path.join(os.path.join('Model_Training/TensorBoard/CL_Experiments', TimeID), f"_{mode}"))

        if verbose >= 1:
            print(f"[{user_id}] {mode} ...")

        result_dict = {'user_id': user_id, 'mode': mode}

        if mode == 'global_eval':
            loss, r2, me, sd, rmse, mae = self.Evaluate_Loader(test_loader)
            result_dict.update({
                'test_loss': loss,
                'test_r2': r2,
                'test_me': me,
                'test_sd': sd,
                'test_rmse': rmse,
                'test_mae': mae,
            })
            Writer.close()
            return result_dict
        
        if trainable_keywords is None:
            trainable_keywords = head_keywords if head_keywords is not None else ["final_fc"]

        # 2) seq_ft / seq_ewc：顺序训练
        filter_fn = lambda name: any(name == pref or name.startswith(pref + ".") for pref in trainable_keywords)
        ewc = EWCRegularizer(self.Model_Running, filter_fn, self.Device)
        global_step = 1
        cl_batch_loss_mean = []
        b1_mae_hist = []
        all_train_losses = []
        # 验证集+初始化早停标志和最佳模型跟踪
        best_val_mae = float('inf')
        best_state = None
        best_tag = None
        no_improve = 0
        stopped_early = False

        def _eval_val_and_maybe_update(tag: str):
            """返回 True 表示触发 early-stop（仅 patience>0 时可能触发）"""
            nonlocal best_val_mae, best_state, best_tag, no_improve
            if val_loader is None:
                return False

            v_loss, v_r2, v_me, v_sd, v_rmse, v_mae = self.Evaluate_Loader(val_loader)
            result_dict[f'val_mae_{tag}'] = float(v_mae)

            if (v_mae + min_delta) < best_val_mae:
                best_val_mae = float(v_mae)
                best_tag = tag
                best_state = {k: v.detach().cpu().clone() for k, v in self.Model_Running.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            return (patience > 0 and no_improve >= patience)
        
        for k, loader in enumerate(batch_loaders):
            # 统计该 CL batch 的平均训练 loss（不刷进度条）
            batch_losses = []
            # 状态传递的重置：每进入一个新任务，清空上一任务的动量残留，但保持模型参数传递
            self.Optimizer_BP = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.Model_Running.parameters()), 
                lr=1e-4, 
                weight_decay=0
            )
            for Epoch in range(1, self.Num_Epoch + 1):  # Num_Epoch 视作 epochs_per_batch
                self.Model_Running.train()
                # collect per-epoch predictions/labels for metrics
                Epoch_BP_Labels = []
                Epoch_BP_Preds = []

                if show_progress:
                    with PB.ProgressBar(widgets=widgets, max_value=len(loader)) as bar:
                        for i, (inputs, labels) in enumerate(loader):
                            inputs, labels = inputs.float().to(self.Device), labels.float().to(self.Device)

                            self.Optimizer_BP.zero_grad()
                            outputs = self.Model_Running(inputs)
                            # collect preds/labels for this epoch
                            Epoch_BP_Labels.append(labels.cpu().detach().numpy())
                            Epoch_BP_Preds.append(outputs.cpu().detach().numpy())
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
                        # collect preds/labels for this epoch
                        Epoch_BP_Labels.append(labels.cpu().detach().numpy())
                        Epoch_BP_Preds.append(outputs.cpu().detach().numpy())
                        loss = self.BP_Loss_Fun(outputs, labels)

                        if mode == 'seq_ewc' and k > 0:
                            loss += lambda_ewc * ewc.ewc_loss()

                        loss.backward()
                        self.Optimizer_BP.step()

                        batch_losses.append(loss.item())

                        if not global_step % 5:
                            Writer.add_scalar(f'CL_Loss/Batch_{k+1}', loss.item(), global_step)
                        global_step += 1

                # 每个 epoch 后的评估和早停判断
                if (val_loader is not None) and (val_check == 'epoch'):
                    if _eval_val_and_maybe_update(tag=f'after_b{k+1}_e{Epoch}'):
                        stopped_early = True
                        break

            # epoch 循环若提前停止，则直接跳出 batch 循环
            if stopped_early:
                break
            # 每个 CL batch 结束：评估 batch1 MAE 作为遗忘监控
            avg_loss_k = float(np.mean(batch_losses)) if len(batch_losses) else 0.0
            cl_batch_loss_mean.append(avg_loss_k)
            all_train_losses.extend(batch_losses)
            _, _, _, _, _, b1_mae = self.Evaluate_Loader(batch_loaders[0])
            b1_mae_hist.append(float(b1_mae))
            # 将每个 epoch 结束时的预测和标签合并，计算所有指标
            Epoch_BP_Labels = np.concatenate(Epoch_BP_Labels, axis=0)
            Epoch_BP_Preds = np.concatenate(Epoch_BP_Preds, axis=0)

            # 计算各项指标
            Epoch_Train_R2 = R2(Epoch_BP_Labels, Epoch_BP_Preds)
            Epoch_Train_ME = ME(Epoch_BP_Labels, Epoch_BP_Preds)
            Epoch_Train_SD = SD(Epoch_BP_Labels, Epoch_BP_Preds)
            Epoch_Train_RMSE = RMSE(Epoch_BP_Labels, Epoch_BP_Preds)
            Epoch_Train_MAE = MAE(Epoch_BP_Labels, Epoch_BP_Preds)

            # 将训练结果记录到 TensorBoard
            Writer.add_scalars(f'Metrics/Batch_{k+1}', {
                'R2': Epoch_Train_R2,
                'ME': Epoch_Train_ME,
                'SD': Epoch_Train_SD,
                'RMSE': Epoch_Train_RMSE,
                'MAE': Epoch_Train_MAE
            }, global_step)

            # EWC：每段后更新快照+Fisher
            if mode == 'seq_ewc':
                ewc.consolidate()
                ewc.estimate_fisher(loader, self.BP_Loss_Fun)
            # 每个 CL batch 结束后，如果设置了 val_check='batch'，则评估一次验证集并判断是否早停
            if (val_loader is not None) and (val_check == 'batch'):
                if _eval_val_and_maybe_update(tag=f'after_batch{k+1}'):
                    stopped_early = True
                    break
            if verbose >= 2:
                avg_loss = float(np.mean(batch_losses)) if len(batch_losses) else 0.0
                print(f"  - CL batch {k+1}/{len(batch_loaders)} | avg_loss {avg_loss:.3f} | b1_mae {b1_mae:.3f}")
            if stopped_early:
                break
        # ==================== Priority-1: rollback to best val checkpoint ====================
        if rollback_to_best and (best_state is not None):
            self.Model_Running.load_state_dict(best_state, strict=True)
            result_dict['val_best_mae'] = float(best_val_mae)
            result_dict['val_best_at'] = best_tag
        else:
            result_dict['val_best_mae'] = float('nan')
            result_dict['val_best_at'] = None

        # 3) 最终 test 评估
        test_loss, test_r2, test_me, test_sd, test_rmse, test_mae = self.Evaluate_Loader(test_loader)
        result_dict.update({
            'test_loss': test_loss,
            'test_r2': test_r2,
            'test_me': test_me,
            'test_sd': test_sd,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
        })

        # 额外：CL过程统计（尽可能全）
        result_dict['cl_train_loss_mean'] = float(np.mean(all_train_losses)) if len(all_train_losses) else 0.0
        for i, v in enumerate(cl_batch_loss_mean, 1):
            result_dict[f'cl_batch{i}_loss_mean'] = float(v)

        # b1 mae 序列（用于遗忘、漂移诊断）
        for i, v in enumerate(b1_mae_hist, 1):
            result_dict[f'b1_mae_after_batch{i}'] = float(v)

        result_dict['b1_mae_first'] = float(b1_mae_hist[0]) if len(b1_mae_hist) else 0.0
        result_dict['b1_mae_last']  = float(b1_mae_hist[-1]) if len(b1_mae_hist) else 0.0

        # 遗忘度（你原来逻辑）
        result_dict['forget'] = (b1_mae_hist[-1] - b1_mae_hist[0]) if len(b1_mae_hist) >= 2 else 0.0

        if verbose >= 1:
            print(f"[{user_id}] {mode} done | MAE {test_mae:.4f} | Forget {result_dict['forget']:.4f}")

        Writer.close()
        return result_dict

# 持续学习 EWC 正则化器 # ===== Replace your EWCRegularizer in Trainer.py with this version =====
class EWCRegularizer:
    def __init__(self, model, param_filter_fn, device='cuda'):
        self.model = model
        self.param_filter_fn = param_filter_fn
        self.device = device
        self.theta_star = {}
        self.fisher = {}   # online fisher (EMA)

    def consolidate(self):
        """保存当前任务结束后的最优参数快照 θ*"""
        self.theta_star = {}
        for name, param in self.model.named_parameters():
            if self.param_filter_fn(name) and param.requires_grad:
                self.theta_star[name] = param.detach().clone()

    def estimate_fisher(
        self,
        dataloader,
        loss_fn,
        max_batches=200,      # ✅ 建议从 50 提高到 200（或按任务大小比例）
        gamma=0.95,           # ✅ Online EWC: F <- gamma*F + F_new
        damping=1e-8,         # ✅ 防止 fisher 为 0 导致正则无效
        use_eval_mode=True,   # ✅ eval 模式估 fisher，减少 BN/Dropout 噪声
    ):
        """估计当前任务的 Fisher 对角阵，并与历史 Fisher 做 EMA 累积"""
        # 1) 初始化本任务 fisher_new
        fisher_new = {}
        for name, param in self.model.named_parameters():
            if self.param_filter_fn(name) and param.requires_grad:
                fisher_new[name] = torch.zeros_like(param, device=self.device)

        # 2) 估 Fisher 时，尽量别让 BN 更新 running stats
        was_training = self.model.training
        if use_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        batch_count = 0
        for inputs, targets in dataloader:
            if batch_count >= max_batches:
                break
            inputs = inputs.float().to(self.device)
            targets = targets.float().to(self.device)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if self.param_filter_fn(name) and param.requires_grad and param.grad is not None:
                    fisher_new[name] += (param.grad.detach() ** 2)

            batch_count += 1

        denom = float(batch_count if batch_count > 0 else 1.0)
        for name in fisher_new:
            fisher_new[name] = fisher_new[name] / denom
            fisher_new[name] = fisher_new[name] + damping  # ✅ 避免全 0

        # 3) Online EWC: 与历史 fisher 做 EMA 累积
        if len(self.fisher) == 0:
            self.fisher = fisher_new
        else:
            for name in fisher_new:
                if name in self.fisher:
                    self.fisher[name] = gamma * self.fisher[name] + fisher_new[name]
                else:
                    self.fisher[name] = fisher_new[name]

        # 4) 还原训练状态
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        self.model.zero_grad(set_to_none=True)

    def ewc_loss(self):
        """EWC penalty: Σ_i F_i (θ_i - θ*_i)^2"""
        loss = 0.0
        for name, param in self.model.named_parameters():
            if self.param_filter_fn(name) and param.requires_grad:
                if name in self.fisher and name in self.theta_star:
                    loss += torch.sum(self.fisher[name] * (param - self.theta_star[name]) ** 2)
        return loss