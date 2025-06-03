rng(42); % 固定随机种子，确保每次随机划分一致
data = load('D:\Data\PulseDB\MIMIC_Info_Files\MIMIC_CalFree_Test_Info.mat'); % 替换为你的 MAT 文件名
fieldNames = fieldnames(data);
structData = data.(fieldNames{1}); % 动态访问第一个字段

% 获取唯一的 Subj_Name
uniqueNames = unique({structData.Subj_Name}); % 提取所有唯一的 Subj_Name

% 初始化两个分组
FN_MIMIC_train_Info = [];
FN_MIMIC_test_Info = [];

% 遍历每个 Subj_Name，进行分组
for i = 1:length(uniqueNames)
    % 获取当前 Subj_Name 的数据
    currentName = uniqueNames{i};
    indices = strcmp({structData.Subj_Name}, currentName);
    currentData = structData(indices);
    
    % 随机打乱数据顺序
    randIndices = randperm(length(currentData)); % 随机排列索引
    currentData = currentData(randIndices); % 按随机顺序重排数据
    
    % 按照 1:9 分割
    splitIdx = round(length(currentData) * 0.1); % 计算 10% 的数据索引
    FN_MIMIC_train_Info = [FN_MIMIC_train_Info; currentData(1:splitIdx)]; % 随机 10% 数据分到 group1
    FN_MIMIC_test_Info = [FN_MIMIC_test_Info; currentData(splitIdx+1:end)]; % 剩余随机 90% 数据分到 group2
end

% 保存到两个 MAT 文件
save('D:\Data\PulseDB\FN_Info_Files\FN_MIMIC_train_Info.mat', 'FN_MIMIC_train_Info','-v7.3');
save('D:\Data\PulseDB\FN_Info_Files\FN_MIMIC_test_Info.mat', 'FN_MIMIC_test_Info','-v7.3');

disp('数据已按照随机 10% 和 90% 分割并保存为两个 MAT 文件。');