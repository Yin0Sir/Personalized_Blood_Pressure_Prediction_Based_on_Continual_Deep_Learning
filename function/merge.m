
% 合并两个 PulseDB 子集结构体：Sorted_Subset_293Users 与 VitalDB_CalFree_Test_Subset
% Author: ChatGPT
% Date: 2025-06-04

clear; clc;

% 加载两个数据集（变量名都为 Subset）
load("D:\Data\PulseDB\Supplementary_Subset_Files\Sorted_Subset_293Users.mat");      % 包含 293 个用户的数据
Subset1 = sorted_subset;
load("D:\Data\PulseDB\Supplementary_Subset_Files\VitalDB_CalFree_Test_Subset.mat"); % 原测试集数据
Subset2 = Subset;

% 合并字段
Merged_Subset.Subject = [Subset1.Subject; Subset2.Subject];
Merged_Subset.Signals = cat(1, Subset1.Signals, Subset2.Signals);
Merged_Subset.SBP     = [Subset1.SBP; Subset2.SBP];
Merged_Subset.DBP     = [Subset1.DBP; Subset2.DBP];
Merged_Subset.Age     = [Subset1.Age; Subset2.Age];
Merged_Subset.Gender  = [Subset1.Gender; Subset2.Gender];
Merged_Subset.Height  = [Subset1.Height; Subset2.Height];
Merged_Subset.Weight  = [Subset1.Weight; Subset2.Weight];
Merged_Subset.BMI     = [Subset1.BMI; Subset2.BMI];
Merged_Subset.Time    = cat(1, Subset1.Time, Subset2.Time);

% 保存结果
save('task2.mat', 'Merged_Subset', '-v7.3');
disp('✅ 已成功合并两个结构体数据并保存');
