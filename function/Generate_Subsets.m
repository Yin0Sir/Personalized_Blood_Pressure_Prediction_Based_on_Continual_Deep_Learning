clear;clc;

%% 生成 PulseDB 的训练、校准和测试子集
% 注意：数据的身高、体重和 BMI 字段仅对segements 从 VitalDB 数据集中分离出来，并且对于 VitalDB 数据集中的段，
% 将是 NaN 的 MIMIC-III 匹配子集，因为这些 infraamtion 不包括在原始 MIMIC-III 匹配子集。

%% 加载文件 （移动硬盘上）
MIMIC_Path="E:/Dataset/PulseDB/Segment_Files/PulseDB_MIMIC";
Vital_Path="E:/Dataset/PulseDB/Segment_Files/PulseDB_Vital";

%% 加载Pulse信息文件
% Train_Info='E:/Dataset/PulseDB/Info_Files/Train_Info';
% CalBased_Test_Info='E:/Dataset/PulseDB/Info_Files/CalBased_Test_Info';
% CalFree_Test_Info='E:/Dataset/PulseDB/Info_Files/CalFree_Test_Info';
% AAMI_Test_Info='E:/Dataset/PulseDB/Info_Files/AAMI_Test_Info';
% AAMI_Cal_Info='E:/Dataset/PulseDB/Info_Files/AAMI_Cal_Info';

%% 加载MIMIC信息文件
MIMIC_Train_Info='D:/Data/PulseDB/MIMIC_Info_Files/MIMIC_Train_Info';
MIMIC_CalBased_Test_Info='D:/Data/PulseDB/MIMIC_Info_Files/MIMIC_CalBased_Test_Info';
MIMIC_CalFree_Test_Info='D:/Data/PulseDB/MIMIC_Info_Files/MIMIC_CalFree_Test_Info';
MIMIC_AAMI_Test_Info='D:/Data/PulseDB/MIMIC_Info_Files/MIMIC_AAMI_Test_Info';
MIMIC_AAMI_Cal_Info='D:/Data/PulseDB/MIMIC_Info_Files/MIMIC_AAMI_Cal_Info';

%% 加载Vital信息文件
% VitalDB_Train_Info='D:/Data/PulseDB/Supplementary_Info_Files/VitalDB_Train_Info';
% VitalDB_CalBased_Test_Info='D:/Data/PulseDB/Supplementary_Info_Files/VitalDB_CalBased_Test_Info';
% VitalDB_CalFree_Test_Info='D:/Data/PulseDB/Supplementary_Info_Files/VitalDB_CalFree_Test_Info';
% VitalDB_AAMI_Test_Info='D:/Data/PulseDB/Supplementary_Info_Files/VitalDB_AAMI_Test_Info';
% VitalDB_AAMI_Cal_Info='D:/Data/PulseDB/Supplementary_Info_Files/VitalDB_AAMI_Cal_Info';

%% 加载FN信息文件
% FN_MIMIC_train_Info='D:/Data/PulseDB/FN_Info_Files/FN_MIMIC_train_Info';
% FN_MIMIC_test_Info='D:/Data/PulseDB/FN_Info_Files/FN_MIMIC_test_Info';
% FN_VitalDB_train_Info='D:/Data/PulseDB/FN_Info_Files/FN_VitalDB_train_Info';
% FN_VitalDB_test_Info='D:/Data/PulseDB/FN_Info_Files/FN_VitalDB_test_Info';
% FN_train_Info='D:/Data/PulseDB/FN_Info_Files/FN_train_Info';
% FN_test_Info='D:/Data/PulseDB/FN_Info_Files/FN_test_Info';

%% 生成Pulse子集
% Generate_Subset(MIMIC_Path,Vital_Path,Train_Info,'E:/Dataset/PulseDB/Subset_Files/Train_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,CalBased_Test_Info,'E:/Dataset/PulseDB/Subset_Files/CalBased_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,CalFree_Test_Info,'E:/Dataset/PulseDB/Subset_Files/CalFree_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,AAMI_Test_Info,'E:/Dataset/PulseDB/Subset_Files/AAMI_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,AAMI_Cal_Info,'E:/Dataset/PulseDB/Subset_Files/AAMI_Cal_Subset')

%% 生成MIMIC子集
% Generate_Subset(MIMIC_Path,Vital_Path,MIMIC_Train_Info,'D:/Data/PulseDB/MIMIC_Subset_Files/MIMIC_Train_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,MIMIC_CalBased_Test_Info,'D:/Data/PulseDB/MIMIC_Subset_Files/MIMIC_CalBased_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,MIMIC_CalFree_Test_Info,'D:/Data/PulseDB/MIMIC_Subset_Files/MIMIC_CalFree_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,MIMIC_AAMI_Test_Info,'D:/Data/PulseDB/MIMIC_Subset_Files/MIMIC_AAMI_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,MIMIC_AAMI_Cal_Info,'D:/Data/PulseDB/MIMIC_Subset_Files/MIMIC_AAMI_Cal_Subset')

%% 生成Vital子集
% Generate_Subset(MIMIC_Path,Vital_Path,VitalDB_Train_Info,'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_Train_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,VitalDB_CalBased_Test_Info,'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalBased_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,VitalDB_CalFree_Test_Info,'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_CalFree_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,VitalDB_AAMI_Test_Info,'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_AAMI_Test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,VitalDB_AAMI_Cal_Info,'D:/Data/PulseDB/Supplementary_Subset_Files/VitalDB_AAMI_Cal_Subset')

%% 生成微调子集（CalFree1:9）
% Generate_Subset(MIMIC_Path,Vital_Path,FN_MIMIC_train_Info,'D:/Data/PulseDB/FN_Subset_Files/FN_MIMIC_train_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,FN_MIMIC_test_Info,'D:/Data/PulseDB/FN_Subset_Files/FN_MIMIC_test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,FN_VitalDB_train_Info,'D:/Data/PulseDB/FN_Subset_Files/FN_VitalDB_train_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,FN_VitalDB_test_Info,'D:/Data/PulseDB/FN_Subset_Files/FN_VitalDB_test_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,FN_train_Info,'D:/Data/PulseDB/FN_Subset_Files/FN_train_Subset')
% Generate_Subset(MIMIC_Path,Vital_Path,FN_test_Info,'D:/Data/PulseDB/FN_Subset_Files/FN_test_Subset')

%% Function
function Generate_Subset(MIMIC_Path, Vital_Path,Info_File_Path, Save_Name)
% Retrieve segments from files using the Info file
Info=load(Info_File_Path);
Field=fieldnames(Info);
Info=Info.(Field{1});
Len=numel(Info);

% Pre-allocate memory
Subset.Subject=cell(Len,1);
Subset.Signals=zeros(Len,3,1250);
Subset.SBP=NaN(Len,1);
Subset.DBP=NaN(Len,1);
Subset.Age=NaN(Len,1);
Subset.Gender=cell(Len,1);
Subset.Height=NaN(Len,1);
Subset.Weight=NaN(Len,1);
Subset.BMI=NaN(Len,1);
Subset.Time=zeros(Len,1,1250);

% Locate unique subjects in the Info file, load subject-by-subject
Subjects=unique({Info.Subj_Name});

pos=1;
f=waitbar(0,['Gathering Data For: ', Save_Name]);
f.Children.Title.Interpreter = 'none';
for i=1:numel(Subjects)
    waitbar(i/numel(Subjects),f)
    
    Subj_Name=Subjects{i};
    Subj_ID=Subj_Name(1:7);
    Source=str2double(Subj_Name(end));
    if Source==0
        Segment_Path=MIMIC_Path;
    elseif Source==1
        Segment_Path=Vital_Path;
    end
    
    Segments_File=load(fullfile(Segment_Path,Subj_ID));
    Subj_Segments=Segments_File.Subj_Wins;
    Selected_IDX=[Info(strcmp({Info.Subj_Name},Subj_Name)).Subj_SegIDX]; % All selected segments belonging to this subject
    
    for j=Selected_IDX
        Segment=Subj_Segments(j);
        Subset.Subject{pos}=Subj_Name;
        Subset.Signals(pos,:,:)=[Segment.ECG_F,Segment.PPG_F,Segment.ABP_Raw]';
        Subset.SBP(pos)=Segment.SegSBP;
        Subset.DBP(pos)=Segment.SegDBP;
        Subset.Age(pos)=Segment.Age;
        Subset.Gender{pos}=Segment.Gender;
        if Source==1 %Record information for VitalDB subjects
            Subset.Height(pos)=Segment.Height;
            Subset.Weight(pos)=Segment.Weight;
            Subset.BMI(pos)=Segment.BMI;
            Subset.Time(pos,:,:)=[Segment.T]';
        end
        pos=pos+1;
    end
    
end
waitbar(1,f,'Saving File')
save(Save_Name, 'Subset','-v7.3')
delete(f)
end