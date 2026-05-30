% Demo_with_fea_gt.m
% 仅有 fea 和 gt 时：用 MPS 方式构造 KH，再跑 MKKM / SimpleMKKM
% KH 构造：fea2KH_MPS（与 MPS FormulationKernels/preprocess 一致，每视图 5 核：Gaussian/Polynomial/Linear/Sigmoid/InvPloyPlus）

clear
clc
warning off;

path = './';
addpath(genpath(path));

%% ========== 准备 fea 和 gt ==========
% fea: cell，每元素为 n x d_v 的视图矩阵
load('3sources_Kmatrix.mat', 'fea', 'gt');
if ~exist('fea', 'var') || ~exist('gt', 'var')
    error('请先定义 fea (cell 多视图) 和 gt (n x 1)，或 load 你的数据。');
end

%% 用 MPS 方式构造多核矩阵 KH
[KH, Y] = fea2KH_MPS(fea, gt);

%% 参数
numclass = length(unique(Y));
Y(Y<1) = numclass;
options.seuildiffsigma = 1e-5;
options.goldensearch_deltmax = 1e-3;
options.goldensearchmax = 1e-8;
options.numericalprecision = 1e-16;
options.firstbasevariable = 'first';
options.nbitermax = 500;
options.seuil = 0;
options.seuilitermax = 10;
options.miniter = 0;
options.threshold = 1e-4;
numTrials = 20;

%% ---- MKKM ----
tic;
[H_normalized0, Sigma0, obj0] = mkkmeans_train(KH, numclass);
acc0 = zeros(numTrials, 1);
nmi0 = zeros(numTrials, 1);
for t = 1:numTrials
    rng(t);
    [res_mean0, ~] = myNMIACCV2(H_normalized0, Y, numclass);
    acc0(t) = res_mean0(1);
    nmi0(t) = res_mean0(2);
end
timecost0 = toc;

%% ---- SimpleMKKM（warm start）----
options.Sigma0 = Sigma0;
tic;
[H_normalized, Sigma, obj] = simpleMKKM(KH, numclass, options);
acc_sm = zeros(numTrials, 1);
nmi_sm = zeros(numTrials, 1);
for t = 1:numTrials
    rng(t);
    [res_mean, ~] = myNMIACCV2(H_normalized, Y, numclass);
    acc_sm(t) = res_mean(1);
    nmi_sm(t) = res_mean(2);
end
timecost = toc;

fprintf('===== KH 由 fea2KH_MPS 构造（MPS 方式），%d 轮 =====\n', numTrials);
fprintf('MKKM:      ACC best=%.4f  mean=%.4f±%.4f,  NMI best=%.4f  mean=%.4f±%.4f;  时间=%.2fs\n', max(acc0), mean(acc0), std(acc0), max(nmi0), mean(nmi0), std(nmi0), timecost0);
fprintf('SimpleMKKM: ACC best=%.4f  mean=%.4f±%.4f,  NMI best=%.4f  mean=%.4f±%.4f;  时间=%.2fs\n', max(acc_sm), mean(acc_sm), std(acc_sm), max(nmi_sm), mean(nmi_sm), std(nmi_sm), timecost);
