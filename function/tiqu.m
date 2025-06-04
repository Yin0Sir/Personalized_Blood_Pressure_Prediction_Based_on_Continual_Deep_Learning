% 1. 加载用户编号列表
load('293nums.mat');  % 假设变量名为 selected_users，类型为 cell array

% 2. 加载两个原始数据集
load('D:\Data\PulseDB\Supplementary_Subset_Files\VitalDB_Train_Subset.mat');
Subset1 = Subset;  % 保存副本
clear Subset;

load('D:\Data\PulseDB\Supplementary_Subset_Files\VitalDB_CalBased_Test_Subset.mat');   % 变量名：Subset
Subset2 = Subset;
clear Subset;

% 3. 合并两个结构体数组
Subset_All.Subject = [Subset1.Subject; Subset2.Subject];
Subset_All.Signals = cat(1, Subset1.Signals, Subset2.Signals);
Subset_All.SBP     = [Subset1.SBP; Subset2.SBP];
Subset_All.DBP     = [Subset1.DBP; Subset2.DBP];
Subset_All.Age     = [Subset1.Age; Subset2.Age];
Subset_All.Gender  = [Subset1.Gender; Subset2.Gender];
Subset_All.Height  = [Subset1.Height; Subset2.Height];
Subset_All.Weight  = [Subset1.Weight; Subset2.Weight];
Subset_All.BMI     = [Subset1.BMI; Subset2.BMI];
Subset_All.Time    = cat(1, Subset1.Time, Subset2.Time);

% 4. 根据用户编号筛选
all_subjects = Subset_All.Subject;
selected_idx = ismember(all_subjects, selected_users);

% 5. 构建筛选后的新结构体
Subset_293Users.Subject = Subset_All.Subject(selected_idx);
Subset_293Users.Signals = Subset_All.Signals(selected_idx, :, :);
Subset_293Users.SBP     = Subset_All.SBP(selected_idx);
Subset_293Users.DBP     = Subset_All.DBP(selected_idx);
Subset_293Users.Age     = Subset_All.Age(selected_idx);
Subset_293Users.Gender  = Subset_All.Gender(selected_idx);
Subset_293Users.Height  = Subset_All.Height(selected_idx);
Subset_293Users.Weight  = Subset_All.Weight(selected_idx);
Subset_293Users.BMI     = Subset_All.BMI(selected_idx);
Subset_293Users.Time    = Subset_All.Time(selected_idx, :, :);

% 6. 保存到新文件
save('VitalDB_Subset_293Users.mat', 'Subset_293Users', '-v7.3');

disp('✅ 成功保存为 VitalDB_Subset_293Users.mat。');