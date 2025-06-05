function sortedStruct = sort_struct_by_field(structVar, fieldName)
% 按结构体中的指定字段对整个结构体排序
% 输入：
%   structVar - 原始结构体
%   fieldName - 要排序的字段名（字符串）
% 输出：
%   sortedStruct - 按指定字段排序后的结构体

    % 获取排序索引
    fieldData = structVar.(fieldName);

    % 如果是 cell 类型（如 Subject/Gender），用 sort
    if iscell(fieldData)
        [~, sortIdx] = sort(fieldData);
    else
        % 否则假设为数值型
        [~, sortIdx] = sort(fieldData);
    end

    % 初始化输出结构体
    sortedStruct = struct();

    % 遍历每个字段，按 sortIdx 重排
    fields = fieldnames(structVar);
    for i = 1:numel(fields)
        f = fields{i};
        data = structVar.(f);
        if ndims(data) == 2
            sortedStruct.(f) = data(sortIdx, :);
        elseif ndims(data) == 3
            sortedStruct.(f) = data(sortIdx, :, :);
        else
            error('Unsupported data dimension for field %s', f);
        end
    end
end