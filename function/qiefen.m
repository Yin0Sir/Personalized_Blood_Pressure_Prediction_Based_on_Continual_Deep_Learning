load('D:\Data\PulseDB\Supplementary_Subset_Files\task2.mat');  % 加载数据
data = Merged_Subset;

subjects = unique(data.Subject);
task_num = 5;

% 初始化五个空结构体
split_data = cell(task_num, 1);
for i = 1:task_num
    split_data{i} = structfun(@(x) [], data, 'UniformOutput', false);
end

for i = 1:length(subjects)
    sid = subjects{i};
    
    % 找到该被试所有样本索引
    idx = find(strcmp(data.Subject, sid));
    
    % 按照每个样本的 Time(:,1) 排序
    [~, sort_idx] = sort(arrayfun(@(i) data.Time(i,1), idx));
    sorted_idx = idx(sort_idx);
    
    % 划分成 5 份
    n = length(sorted_idx);
    chunk_size = floor(n / task_num);
    
    for k = 1:task_num
        start_idx = (k-1)*chunk_size + 1;
        end_idx = (k<task_num) * (k*chunk_size) + (k==task_num)*n;
        this_idx = sorted_idx(start_idx:end_idx);
        
        % 添加到对应任务中
        fields = fieldnames(data);
        for f = 1:length(fields)
            field = fields{f};
        
            % 判断是否是 Signals 字段
            if strcmp(field, 'Signals')
                % 三维拼接，保留 n × 3 × 1250 格式
                split_data{k}.(field) = cat(1, split_data{k}.(field), data.(field)(this_idx, :, :));
            else
                % 普通拼接（例如 Subject 是 cell，其他是 double）
                split_data{k}.(field) = [split_data{k}.(field); data.(field)(this_idx, :)];
            end
        end
    end
end

% 保存结果
for k = 1:task_num
    filename = sprintf('Data_Task%d.mat', k);
    Subset = split_data{k};
    save(filename, 'Subset', '-v7.3');
end
