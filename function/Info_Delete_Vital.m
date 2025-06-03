% 加载 MAT 文件
data = load('D:\Data\PulseDB\Info_Files\Train_Info.mat'); % 替换为你的 MAT 文件路径和名称
fieldNames = fieldnames(data);
structData = data.(fieldNames{1}); % 动态访问第一个字段

% 找到 Source 字段中包含 'VitalDB' 的行
rowsToDelete = strcmp({structData.Source}, 'VitalDB');

% 删除这些行
MIMIC_Train_Info = structData(~rowsToDelete); % 保留非 'VitalDB' 的行

% 保存过滤后的数据
save('D:\Data\PulseDB\MIMIC_Info_Files\MIMIC_Train_Info.mat', 'MIMIC_Train_Info','-v7.3'); % 替换为你想保存的新文件名

disp('已成功删除 Source 字段为 "VitalDB" 的行，并保存为新的 MAT 文件。');
