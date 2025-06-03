% 加载 .mat 文件
data1="D:/Data/PulseDB/Subset_Files/Train_Subset";
data2='D:/Data/PulseDB/Subset_Files/CalBased_Test_Subset';
data3='D:/Data/PulseDB/Subset_Files/CalFree_Test_Subset';
data4='D:/Data/PulseDB/Subset_Files/AAMI_Test_Subset';
data5='D:/Data/PulseDB/Subset_Files/AAMI_Cal_Subset';


Add_Derivatives_To_Subset(data1,'D:/Data/PulseDB/Subset_Files/Train_Subset1')
% Add_Derivatives_To_Subset(data2,'D:/Data/PulseDB/Subset_Files/CalBased_Test_Subset1')
% Add_Derivatives_To_Subset(data3,'D:/Data/PulseDB/Subset_Files/CalFree_Test_Subset1')
% Add_Derivatives_To_Subset(data4,'D:/Data/PulseDB/Subset_Files/AAMI_Test_Subset1')
% Add_Derivatives_To_Subset(data5,'D:/Data/PulseDB/Subset_Files/AAMI_Cal_Subset1')

function Add_Derivatives_To_Subset(MAT_File_Path, Save_Name)
    % Load the MAT file containing the 'Subset' structure
    load(MAT_File_Path, 'Subset');
    
    Len = size(Subset.Signals, 1);
    % Pre-allocate new Signals array (Len x 7 x 1250)
    New_Signals = zeros(Len, 7, 1250);
    
    % Loop through each subject to calculate the derivatives
    for i = 1:Len
        % Get the original signals (ECG_F and PPG_F)
        ECG_F = Subset.Signals(i, 1, :);
        PPG_F = Subset.Signals(i, 2, :);
        
        % 计算一阶和二阶导数
        ECG_F_first = [ECG_F(1); diff(squeeze(ECG_F))];  % First derivative of ECG
        ECG_F_second = [ECG_F(1); ECG_F_first(1); diff(ECG_F_first)];  % Second derivative of ECG
        % 处理导数的大小，使其与原始信号的大小一致（1250）
        ECG_F_second = ECG_F_second(1:end-1);  % 删除最后一个元素，使长度为 1250

        PPG_F_first = [PPG_F(1); diff(squeeze(PPG_F))];  % First derivative of PPG
        PPG_F_second = [PPG_F(1); PPG_F_first(1); diff(PPG_F_first)];  % Second derivative of PPG
        % 处理 PPG_F_second 的大小，确保它与原始信号大小一致
        PPG_F_second = PPG_F_second(1:end-1);  % 删除最后一个元素，使长度为 1250

        % Assign the original and derivative signals to the new Signals array
        New_Signals(i, 1, :) = ECG_F;  % Original ECG
        New_Signals(i, 2, :) = PPG_F;  % Original PPG
        New_Signals(i, 3, :) = Subset.Signals(i, 3, :);  % Original ABP
        New_Signals(i, 4, :) = ECG_F_first;  % First derivative of ECG
        New_Signals(i, 5, :) = ECG_F_second;  % Second derivative of ECG
        New_Signals(i, 6, :) = PPG_F_first;  % First derivative of PPG
        New_Signals(i, 7, :) = PPG_F_second;  % Second derivative of PPG
    end
    
    % Save the updated data structure with new 'Signals' (7 channels)
    Subset.Signals = New_Signals;
    save(Save_Name, 'Subset', '-v7.3');
end
