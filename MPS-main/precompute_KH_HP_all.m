%% precompute_KH_HP_all.m
% 统一预先生成 MPS / MR-MPS 所有数据集的核集合 KH / HP，并保存为 .mat，
% 供 Compare_end.m 等实验脚本直接读取，避免重复 preprocess。
%
% 使用方法：
%   1) 在 MATLAB 中 cd 到 MPS-main 目录
%   2) 运行：run('precompute_KH_HP_all.m')
%
% 说明：
%   - dim_c 固定为 5，与 MPS / MR-MPS 主实验一致
%   - 结果保存到 MPS-main/MR_MPS/KH_cache/<dataset>_KH_HP_dim5.mat
%   - Compare_end.m 将优先从该目录读取 KH / HP

close all; clear; clc; warning off;

baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
addpath(fullfile(baseDir, 'ClusteringMeasure'));
addpath(fullfile(baseDir, 'FormulationKernels'));
addpath(fullfile(baseDir, 'function'));

dataPath = fullfile(baseDir, 'Datasets', 'Multi_view-Datasets');
cacheDir = fullfile(baseDir, 'MR_MPS', 'KH_cache');
if ~exist(cacheDir, 'dir'), mkdir(cacheDir); end

datasetName = {'MSRC_v1','BBCSport','BRCA','WikipediaArticles','3sources','ORL'};
dim_c = 5;

fprintf('==== 预计算 KH / HP 并缓存到 %s ====\n', cacheDir);

for idx = 1:numel(datasetName)
    name = datasetName{idx};
    dataFile = fullfile(dataPath, [name '.mat']);
    if ~isfile(dataFile)
        warning('未找到数据文件：%s，跳过。', dataFile);
        continue;
    end

    cacheFile = fullfile(cacheDir, sprintf('%s_KH_HP_dim%d.mat', name, dim_c));
    if isfile(cacheFile)
        fprintf('[%s] 已存在缓存，跳过生成。\n', name);
        continue;
    end

    fprintf('[%s] 加载数据并生成 KH / HP...\n', name);
    load(dataFile, 'fea', 'gt');
    gt = gt(:);
    num_cluster = length(unique(gt));

    tic;
    [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c);
    t_prep = toc;
    fprintf('    完成：num_cluster=%d, num_kernel=%d, 用时 %.2fs\n', num_cluster, num_kernel, t_prep);

    num_cluster_cache = num_cluster; %#ok<NASGU>
    save(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c', '-v7.3');
    fprintf('    已保存到：%s\n', cacheFile);
end

fprintf('==== KH / HP 预计算完成。====\n');

