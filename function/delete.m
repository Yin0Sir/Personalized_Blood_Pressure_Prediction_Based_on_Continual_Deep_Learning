load('selected_users.mat');  % 变量名应为 selected_users (cell array 或 string array)
if isstruct(selected_users)
    selected_users = selected_users.selected_users;  % 如果是 struct 中的字段
end
selected_users = string(selected_users);  % 确保是 string 格式

% 2. 加载两个原始数据集
subsetFiles = {
    'VitalDB_CalBased_Test_Subset.mat', ...
    'VitalDB_Train_Subset.mat'
};

outputFiles = {
    'VitalDB_CalBased_Test_Subset_Removed293.mat', ...
    'VitalDB_Train_Subset_Removed293.mat'
};

for idx = 1:2
    % 读取原始 subset
    load(subsetFiles{idx}, 'Subset');
    
    % 将 Subject 字段转换为 string 数组以便比较
    all_subjects = string(Subset.Subject);
    
    % 找到不属于 selected_users 的索引
    keep_idx = ~ismember(all_subjects, selected_users);

    % 生成新子集（剔除293用户）
    Subset_new.Subject = Subset.Subject(keep_idx);
    Subset_new.Signals = Subset.Signals(keep_idx, :, :);
    Subset_new.SBP = Subset.SBP(keep_idx);
    Subset_new.DBP = Subset.DBP(keep_idx);
    Subset_new.Age = Subset.Age(keep_idx);
    Subset_new.Gender = Subset.Gender(keep_idx);
    Subset_new.Height = Subset.Height(keep_idx);
    Subset_new.Weight = Subset.Weight(keep_idx);
    Subset_new.BMI = Subset.BMI(keep_idx);
    Subset_new.Time = Subset.Time(keep_idx, :, :);

    % 保存新数据集
    save(outputFiles{idx}, 'Subset_new', '-v7.3');
    fprintf('✅ %s saved as %s\n', subsetFiles{idx}, outputFiles{idx});
end
