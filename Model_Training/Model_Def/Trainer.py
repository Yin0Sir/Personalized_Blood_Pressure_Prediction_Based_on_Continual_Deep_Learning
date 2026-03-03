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
                inputs, labels = inputs.float().to(self.Device), labels.float().to(self.Device)
                preds = self.Model_Running(inputs)
                all_preds.append(preds.cpu().detach().numpy())
                all_labels.append(labels.cpu().detach().numpy())
                
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
        mode='seq_ewc', lambda_ewc=0, trainable_keywords=None, head_keywords=None,
        val_check='epoch',         # ✅ 'batch' 或 'epoch'
        rollback_to_best=True,     # ✅ 是否回滚到val最佳
        patience=0,                # ✅ 0=不早停；>0支持早停
        min_delta=0.0,             # ✅ val改进阈值
        buffer_size=200, replay_batch_size=8, alpha_replay=1.0, # [新增] Replay 参数
        verbose=0,            # 0: 不打印; 1: 仅打印开始/结束; 2: 打印每个CL batch摘要
        show_progress=False,  # True 才显示 progressbar
):
        TimeID = self.TimeID
        tb_dir = os.path.join('Model_Training', 'TensorBoard', 'CL_Experiments', TimeID, mode, str(user_id))
        Writer = SW(tb_dir)

        # 将模型信息写入 TensorBoard（与 Train_Model 格式一致）
        save_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        print('ModelID: '+TimeID[-6:])
        self.Model_Info()
        sys.stdout = save_stdout
        Writer.add_text('Model', result.getvalue().replace('\n', '     \n'))

        if verbose >= 1: print(f"[{user_id}] {mode} ...")
        result_dict = {'user_id': user_id, 'mode': mode}

        # 模式1: 全局直接评估 (不训练)
        if mode == 'global_eval':
            loss, r2, me, sd, rmse, mae = self.Evaluate_Loader(test_loader)
            result_dict.update({'test_loss': loss, 'test_r2': r2, 'test_me': me, 'test_sd': sd, 'test_rmse': rmse, 'test_mae': mae})
            Writer.close()
            return result_dict
        
        # 模式2: 持续学习 (seq_ft / seq_ewc)
        # [新增] 解析当前模式需要启用哪些机制
        use_ewc = mode in ['seq_ewc', 'seq_hybrid']
        use_replay = mode in ['seq_replay', 'seq_hybrid']

        # [新增] 初始化 EWC 和 Replay Buffer
        if use_ewc:
            trainable_keywords = trainable_keywords or head_keywords or ["final_fc"]
            filter_fn = lambda name: any(name == pref or name.startswith(pref + ".") for pref in trainable_keywords)
            ewc = EWCRegularizer(self.Model_Running, filter_fn, self.Device)
            
        if use_replay:
            replay_buffer = ReplayBuffer(capacity=buffer_size, device=self.Device)

        global_step = 1
        cl_batch_loss_mean, b1_mae_hist, all_train_losses = [], [], []
        # 全局最佳验证状态（保留用于统计，可选）
        best_val_mae, best_state, best_tag, no_improve = float('inf'), None, None, 0
        stopped_early = False

        # ----- per‑batch variables (reset for each CL batch) -----
        batch_best_mae = float('inf')
        batch_best_state = None
        batch_best_tag = None
        batch_no_improve = 0

        def _eval_val_and_maybe_update(tag: str, batch_scope: bool = False):
            """验证+早停逻辑。

            * 如果 batch_scope=True，还会更新当前 CL batch 的局部最佳。
            * 返回值 only reflects global patience condition (保持原行为)。
            """
            nonlocal best_val_mae, best_state, best_tag, no_improve
            nonlocal batch_best_mae, batch_best_state, batch_best_tag, batch_no_improve
            if val_loader is None:
                return False
            v_loss, v_r2, v_me, v_sd, v_rmse, v_mae = self.Evaluate_Loader(val_loader)
            result_dict[f'val_mae_{tag}'] = float(v_mae)

            # update global best
            if (v_mae + min_delta) < best_val_mae:
                best_val_mae, best_tag, no_improve = float(v_mae), tag, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.Model_Running.state_dict().items()}
            else:
                no_improve += 1

            # update batch‑specific best if requested
            if batch_scope:
                if (v_mae + min_delta) < batch_best_mae:
                    batch_best_mae, batch_best_tag, batch_no_improve = float(v_mae), tag, 0
                    batch_best_state = {k: v.detach().cpu().clone() for k, v in self.Model_Running.state_dict().items()}
                else:
                    batch_no_improve += 1

            return (patience > 0 and batch_no_improve >= patience)
        
        # --- 遍历 CL Batches ---
        for k, loader in enumerate(batch_loaders):
            # reset per-batch best tracking before training this new task/batch
            batch_best_mae = float('inf')
            batch_best_state = None
            batch_best_tag = None
            batch_no_improve = 0
            batch_losses = []
            
            for Epoch in range(1, self.Num_Epoch + 1):
                self.Model_Running.train()
                Epoch_BP_Labels, Epoch_BP_Preds = [], []
                
                # 配置进度条或普通迭代器
                iterator = PB.ProgressBar(widgets=widgets, max_value=len(loader))(loader) if show_progress else loader
                
                for i, (inputs, labels) in enumerate(iterator):
                    inputs, labels = inputs.float().to(self.Device), labels.float().to(self.Device)
                    self.Optimizer_BP.zero_grad()
                    outputs = self.Model_Running(inputs)
                    
                    Epoch_BP_Labels.append(labels.cpu().detach().numpy())
                    Epoch_BP_Preds.append(outputs.cpu().detach().numpy())

                    loss = self.BP_Loss_Fun(outputs, labels)

                    # 2. [新增] 加上 EWC 约束惩罚 (仅从第二个任务开始)
                    if use_ewc and k > 0:
                        loss += lambda_ewc * ewc.ewc_loss()

                    # 3. [新增] 加上 Replay Loss (仅从第二个任务开始)
                    if use_replay and k > 0:
                        buf_inputs, buf_labels = replay_buffer.sample(replay_batch_size)
                        if buf_inputs is not None:
                            buf_inputs = buf_inputs.float().to(self.Device)
                            buf_labels = buf_labels.float().to(self.Device)
                            buf_outputs = self.Model_Running(buf_inputs)
                            replay_loss = self.BP_Loss_Fun(buf_outputs, buf_labels)
                            loss += alpha_replay * replay_loss

                    loss.backward()
                    self.Optimizer_BP.step()
                    batch_losses.append(loss.item())

                    if not global_step % 5:
                        Writer.add_scalar(f'CL_Loss/Batch_{k+1}', loss.item(), global_step)
                    global_step += 1
                    
                    if show_progress: iterator.update(i, Batch_BP_Loss=loss.item())

                # 计算并记录此 epoch 的训练集指标（按 epoch 作为 x 轴）
                if len(Epoch_BP_Labels) > 0:
                    try:
                        Epoch_BP_Labels_arr = np.concatenate(Epoch_BP_Labels, axis=0)
                        Epoch_BP_Preds_arr = np.concatenate(Epoch_BP_Preds, axis=0)
                        E_R2 = R2(Epoch_BP_Labels_arr, Epoch_BP_Preds_arr)
                        E_ME = ME(Epoch_BP_Labels_arr, Epoch_BP_Preds_arr)
                        E_SD = SD(Epoch_BP_Labels_arr, Epoch_BP_Preds_arr)
                        E_RMSE = RMSE(Epoch_BP_Labels_arr, Epoch_BP_Preds_arr)
                        E_MAE = MAE(Epoch_BP_Labels_arr, Epoch_BP_Preds_arr)
                        E_loss = self.BP_Loss_Fun(torch.from_numpy(Epoch_BP_Preds_arr), torch.from_numpy(Epoch_BP_Labels_arr))

                        Writer.add_scalars('Loss', {'Train_BP': float(E_loss)}, Epoch)
                        Writer.add_scalars('R2', {'Train': float(E_R2)}, Epoch)
                        Writer.add_scalars('ME', {'Train': float(E_ME)}, Epoch)
                        Writer.add_scalars('SD', {'Train': float(E_SD)}, Epoch)
                        Writer.add_scalars('RMSE', {'Train': float(E_RMSE)}, Epoch)
                        Writer.add_scalars('MAE', {'Train': float(E_MAE)}, Epoch)

                        # 为每个 CL batch 创建独立面板（例如在 TensorBoard 中按 Batch_1/LOSS 等分组）
                        batch_tag = f'Batch_{k+1}'
                        Writer.add_scalars(f'{batch_tag}/Loss', {'Train_BP': float(E_loss)}, Epoch)
                        Writer.add_scalars(f'{batch_tag}/R2', {'Train': float(E_R2)}, Epoch)
                        Writer.add_scalars(f'{batch_tag}/ME', {'Train': float(E_ME)}, Epoch)
                        Writer.add_scalars(f'{batch_tag}/SD', {'Train': float(E_SD)}, Epoch)
                        Writer.add_scalars(f'{batch_tag}/RMSE', {'Train': float(E_RMSE)}, Epoch)
                        Writer.add_scalars(f'{batch_tag}/MAE', {'Train': float(E_MAE)}, Epoch)
                    except Exception:
                        pass

                # Epoch级别的早停检查（同时更新 batch 内最佳）
                if val_loader and val_check == 'epoch' and _eval_val_and_maybe_update(f'after_b{k+1}_e{Epoch}', batch_scope=True):
                    stopped_early = True; break

            # 每个 CL batch 结束：评估 batch1 MAE 作为遗忘监控
            all_train_losses.extend(batch_losses)
            cl_batch_loss_mean.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
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
            # 将训练结果以规范化字典写入 TensorBoard（与 Train_Model 的格式一致）
            Writer_Loss_Dict = {'Train_BP': float(np.mean(batch_losses)) if batch_losses else 0.0}
            Writer_R2_Dict = {'Train': float(Epoch_Train_R2)}
            Writer_ME_Dict = {'Train': float(Epoch_Train_ME)}
            Writer_SD_Dict = {'Train': float(Epoch_Train_SD)}
            Writer_RMSE_Dict = {'Train': float(Epoch_Train_RMSE)}
            Writer_MAE_Dict = {'Train': float(Epoch_Train_MAE)}
            # 以全局步数为 x 轴记录
            Writer.add_scalars('Loss', Writer_Loss_Dict, global_step)
            Writer.add_scalars('R2', Writer_R2_Dict, global_step)
            Writer.add_scalars('ME', Writer_ME_Dict, global_step)
            Writer.add_scalars('SD', Writer_SD_Dict, global_step)
            Writer.add_scalars('RMSE', Writer_RMSE_Dict, global_step)
            Writer.add_scalars('MAE', Writer_MAE_Dict, global_step)

            # EWC：每段后更新快照+Fisher  任务结束后更新 EWC 参数 和 Buffer
            if use_ewc:
                ewc.consolidate()
                ewc.estimate_fisher(loader, self.BP_Loss_Fun)
                
            if use_replay:
                # 每个任务结束后，均匀分配 Buffer 容量，提取新数据存入
                # 比如总共 K 个任务，每个任务存 capacity // K 个样本
                samples_to_add = max(1, buffer_size // len(batch_loaders))
                replay_buffer.add_data(loader, samples_to_add)

            # 每个 CL batch 结束后，如果设置了 val_check='batch'，则评估一次验证集并判断是否早停
            if val_loader and val_check == 'batch' and _eval_val_and_maybe_update(f'after_batch{k+1}', batch_scope=True):
                stopped_early = True; break

            # batch结束时对模型进行本批最佳回滚
            if rollback_to_best and batch_best_state is not None:
                # load the best parameters seen during this batch
                self.Model_Running.load_state_dict(batch_best_state, strict=True)
                result_dict[f'batch{ k+1 }_best_mae'] = float(batch_best_mae)
                result_dict[f'batch{ k+1 }_best_at'] = batch_best_tag

            if verbose >= 2:
                avg_loss = float(np.mean(batch_losses)) if len(batch_losses) else 0.0
                print(f"  - CL batch {k+1}/{len(batch_loaders)} | avg_loss {avg_loss:.3f} | b1_mae {b1_mae:.3f}")

            if stopped_early: break

        # ==================== 最终回滚/统计 ====================
        # 训练过程中每个 CL batch 已在 batch 末尾回滚到该 batch 内最佳。
        # 这里不再重新加载全局 best_state，以免覆盖最后一个 batch 的状态。
        if rollback_to_best:
            # 仍然保留全局验证最优统计信息供结果输出
            if best_state is not None:
                result_dict['val_best_mae'] = float(best_val_mae)
                result_dict['val_best_at'] = best_tag
            else:
                result_dict['val_best_mae'] = float('nan')
                result_dict['val_best_at'] = None
        else:
            result_dict['val_best_mae'] = float('nan')
            result_dict['val_best_at'] = None

        # 3) 最终 test 评估
        test_loss, test_r2, test_me, test_sd, test_rmse, test_mae = self.Evaluate_Loader(test_loader)
        result_dict.update({'test_loss': test_loss, 'test_r2': test_r2, 'test_me': test_me, 'test_sd': test_sd, 'test_rmse': test_rmse, 'test_mae': test_mae})

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

# 记忆缓冲区类 (Replay Buffer)
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.inputs = []
        self.labels = []
    
    def add_data(self, loader, num_samples):
        """从当前 loader 中随机提取 num_samples 个样本加入 Buffer"""
        new_inputs, new_labels = [], []
        for x, y in loader:
            new_inputs.append(x.float())
            new_labels.append(y.float())
        new_inputs = torch.cat(new_inputs, dim=0)
        new_labels = torch.cat(new_labels, dim=0)

        # 随机采样
        indices = torch.randperm(len(new_inputs))[:num_samples]
        self.inputs.append(new_inputs[indices].to(self.device))
        self.labels.append(new_labels[indices].to(self.device))

        # 拼接并限制容量 (Reservoir Sampling 的简单替代方案)
        all_inputs = torch.cat(self.inputs, dim=0)
        all_labels = torch.cat(self.labels, dim=0)

        if len(all_inputs) > self.capacity:
            # 如果超出容量，随机丢弃旧数据，保留 capacity 大小
            indices = torch.randperm(len(all_inputs))[:self.capacity]
            all_inputs = all_inputs[indices]
            all_labels = all_labels[indices]

        self.inputs = [all_inputs]
        self.labels = [all_labels]

    def sample(self, batch_size):
        """随机采样一个 batch 用于回放"""
        if len(self.inputs) == 0 or len(self.inputs[0]) == 0:
            return None, None
        inputs_tensor = self.inputs[0]
        labels_tensor = self.labels[0]
        curr_size = len(inputs_tensor)
        batch_size = min(batch_size, curr_size)
        
        indices = torch.randperm(curr_size)[:batch_size]
        return inputs_tensor[indices], labels_tensor[indices]

# 持续学习 EWC 正则化器 
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
            loss_fn(self.model(inputs), targets).backward()

            for name, param in self.model.named_parameters():
                if self.param_filter_fn(name) and param.requires_grad and param.grad is not None:
                    fisher_new[name] += (param.grad.detach() ** 2)
            batch_count += 1

        denom = float(max(batch_count, 1.0))
        for name in fisher_new:
            fisher_new[name] = (fisher_new[name] / denom) + damping  # ✅ 避免全 0

        # 3) Online EWC: 与历史 fisher 做 EMA 累积
        if len(self.fisher) == 0:
            self.fisher = fisher_new
        else:
            for name in fisher_new:
                self.fisher[name] = gamma * self.fisher.get(name, 0) + fisher_new[name]

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