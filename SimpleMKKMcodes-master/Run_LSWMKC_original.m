% Run_LSWMKC_original.m
% 完全按 LSWMKC 原文/demo 设置：alpha_range=2.^[0:10]、neibour=5、myNMIACCV2（50次取max）
% 各 alpha 运行 numTrials 次，输出 mean±std；最后取最佳 alpha 的均值
% 数据可选：(1) LSWMKC 的 datasets/XXX_Kmatrix  (2) MPS 数据 + preprocess

clear
clc
warning off;

%% 路径：仅使用 LSWMKC，保证全部为原代码
lswmkcDir = 'd:\matlab\LSWMKC-main';   % 或 fullfile(fileparts(mfilename('fullpath')), '..', 'LSWMKC-main')
if ~isfolder(lswmkcDir)
    error('请设置 lswmkcDir 为 LSWMKC-main 根目录。');
end
addpath(genpath(lswmkcDir));

%% 原文参数
global neibour
neibour = 3;    % initial neighbours
alpha_range = 2.^[0:1:10];
numTrials = 10; % 每个 alpha 运行次数，输出均值±标准差

%% 数据来源：'lswmkc' 用 LSWMKC 自带 _Kmatrix；'mps' 用 MPS 数据 + preprocess（与 Compare 脚本同 KH）
dataSource = 'mps';   % 'lswmkc' | 'mps'

if strcmpi(dataSource, 'lswmkc')
    dataName = 'Yale';   % 或 YALE 等，需存在 datasets/dataName_Kmatrix.mat
    dataPath = fullfile(lswmkcDir, 'datasets', [dataName, '_Kmatrix.mat']);
    if ~isfile(dataPath)
        error('未找到 %s，请改用 dataSource=''mps'' 或提供 LSWMKC datasets 下的 _Kmatrix 文件。', dataPath);
    end
    load(dataPath, 'KH', 'Y');
    Y(Y == -1) = 2;
    numclass = length(unique(Y));
    KH = kcenter(KH);
    KH = knorm(KH);
    fprintf('数据: LSWMKC %s, n=%d, numclass=%d, numker=%d\n', dataName, size(KH,1), numclass, size(KH,3));
else
    % 与 Compare_SimpleMKKM_vs_MPS 同源：MPS 数据 + preprocess
    mpsDir = 'd:\matlab\MPS-main';
    if ~isfolder(mpsDir)
        mpsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'MPS-main');
    end
    if ~isfolder(mpsDir)
        error('请设置 mpsDir（MPS-main 根目录）以使用 dataSource=''mps''。');
    end
    addpath(mpsDir);
    addpath(fullfile(mpsDir, 'FormulationKernels'));
    addpath(fullfile(mpsDir, 'function'));
    addpath(fullfile(mpsDir, 'ClusteringMeasure'));
    addpath(fullfile(mpsDir, 'new'));
    dataPath = fullfile(mpsDir, 'Datasets', 'Multi_view-Datasets');
    dataFile = fullfile(dataPath, 'ORL.mat');   % 或 BBCSport.mat
    if ~isfile(dataFile)
        error('未找到 %s，请修改 dataFile。', dataFile);
    end
    load(dataFile, 'fea', 'gt');
    num_cluster = length(unique(gt));
    dim_c = 5;
    [KH, ~, ~] = preprocess(fea, num_cluster, dim_c);
    KH = kcenter(KH);
    KH = knorm(KH);
    Y = gt(:);
    u = unique(Y);
    u(u < 1) = [];
    for i = 1:length(u), Y(Y == u(i)) = i; end
    numclass = num_cluster;
    fprintf('数据: MPS preprocess %s, n=%d, numclass=%d, numker=%d\n', dataFile, size(KH,1), numclass, size(KH,3));
end

%% 按原文：每个 alpha 跑 numTrials 次（每次 myNMIACCV2 内部 50 次取 max），输出 mean±std
acc_all = zeros(numTrials, length(alpha_range));
nmi_all = zeros(numTrials, length(alpha_range));
pur_all = zeros(numTrials, length(alpha_range));
rand_all = zeros(numTrials, length(alpha_range));
for alpha_indx = 1:length(alpha_range)
    alpha = alpha_range(alpha_indx);
    [Kstar, ~, ~, ~, ~] = Graph_main(KH, alpha);
    Kstar = kcenter(Kstar);
    Kstar = knorm(Kstar);
    [H, ~] = eigs(Kstar, numclass, 'la');
    for tr = 1:numTrials
        rng(tr);
        [res_mean, ~] = myNMIACCV2(H, Y, numclass);   % 原版：内部 50 次取 max
        acc_all(tr, alpha_indx) = res_mean(1);
        nmi_all(tr, alpha_indx) = res_mean(2);
        pur_all(tr, alpha_indx) = res_mean(3);
        rand_all(tr, alpha_indx) = res_mean(4);
    end
    fprintf('alpha=2^%d: ACC mean=%.4f±%.4f,  NMI mean=%.4f±%.4f,  Pur mean=%.4f±%.4f,  Rand mean=%.4f±%.4f\n', ...
        alpha_indx-1, mean(acc_all(:,alpha_indx)), std(acc_all(:,alpha_indx)), ...
        mean(nmi_all(:,alpha_indx)), std(nmi_all(:,alpha_indx)), ...
        mean(pur_all(:,alpha_indx)), std(pur_all(:,alpha_indx)), ...
        mean(rand_all(:,alpha_indx)), std(rand_all(:,alpha_indx)));
end

%% 取 mean(ACC) 最大的 alpha 作为最佳，输出其均值±标准差
[~, max_indx] = max(mean(acc_all, 1));
fprintf('\n===== LSWMKC 原文设置（各 alpha %d 次取均值）=====\n', numTrials);
fprintf('最佳 alpha=2^%d (下标 %d)：ACC mean=%.4f±%.4f,  NMI mean=%.4f±%.4f,  Pur mean=%.4f±%.4f,  Rand mean=%.4f±%.4f\n', ...
    max_indx-1, max_indx, ...
    mean(acc_all(:,max_indx)), std(acc_all(:,max_indx)), ...
    mean(nmi_all(:,max_indx)), std(nmi_all(:,max_indx)), ...
    mean(pur_all(:,max_indx)), std(pur_all(:,max_indx)), ...
    mean(rand_all(:,max_indx)), std(rand_all(:,max_indx)));
fprintf('即 ACC=%.2f±%.2f%%\n', mean(acc_all(:,max_indx))*100, std(acc_all(:,max_indx))*100);
