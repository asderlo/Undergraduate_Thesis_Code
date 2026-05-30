% M2m
% 基于 MR_main_2stage.m 的两阶段搜索结果设计消融实验（不改动原代码，纯新增脚本）
%
% 消融项（默认都跑）：
%  - full：2-stage 最优 (alpha, sel_k, eta) + 对偶动态流形
%  - w/o MS：去掉多尺度，仅保留第 1 个尺度（dim_c=1）
%  - w/o KS：去掉核选择（sel_k = num_kernel，即所有核都选中）
%  - w/o MR：去掉全部流形项（static/dynamic 都关闭）
%  - w/o Dual：仅保留静态流形项，关闭对偶动态流形
%
% 输出：每个数据集写入 MR_MPS/ablation_2stage_results/*.csv 与 *.mat

close all; clear; clc; warning off;

%% ===================== Path =====================
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end
addpath(scriptDir);  % main_lx.m
addpath(fullfile(scriptDir, 'ClusteringMeasure'));
addpath(fullfile(scriptDir, 'FormulationKernels'));
addpath(fullfile(scriptDir, 'function'));
addpath(fullfile(scriptDir, 'MR_MPS')); % save_best_params_3stage 可用（若你需要）

%% ===================== 数据集设置 =====================
dataPath = fullfile(scriptDir, 'Datasets', 'Multi_view-Datasets');
datasetName = {'MSRC_v1','BBCSport','BRCA',...
    '3sources','WebKB','Prokaryotic'};

% run_all_datasets = false;
run_all_datasets = true;

dataIndex = 1;
if run_all_datasets
    dataIndices = 1:numel(datasetName);
else
    dataIndices = dataIndex;
end

%% ===================== 固定最优参数（来自 2-stage 调优结果）=====================
param_map = struct();
param_map.MSRC_v1 = struct('alpha', 0.20, 'sel_k', 20, 'eta', 1);
param_map.BBCSport = struct('alpha', 0.90, 'sel_k', 5, 'eta', 1);
param_map.BRCA = struct('alpha', 0.20, 'sel_k', 16, 'eta', 100);
param_map.x3sources = struct('alpha', 0.80, 'sel_k', 6, 'eta', 1);
param_map.WebKB = struct('alpha', 0.60, 'sel_k', 14, 'eta', 100);
param_map.Prokaryotic = struct('alpha', 0.70, 'sel_k', 2, 'eta', 1);

%% ===================== MR-MPS 开关（主文件统一配置） =====================
opts_base = struct();
opts_base.kNN_graph = 15;
opts_base.warmup_T = 3;
opts_base.static_term_on = true;
opts_base.dual_dynamic_on = true;
opts_base.dual_dynamic_gamma = 0.5;
opts_base.dual_dynamic_sigma = 0; % 0: 自适应估计 sigma
opts_base.maxIter = 50;
gamma_list = 0:0.1:1;
repeat_times_gamma_search = 1;

% 仅保留当前 main_lx 实际使用的开关，移除未生效配置

%% ===================== 消融评估设置 =====================
repeat_times_eval = 1;
dim_c = 5;
run_ablation = false; % true: full+各消融；false: 仅运行 full

outDir = fullfile(scriptDir, 'MR_MPS', 'ablation_2stage_results');
if ~exist(outDir, 'dir'), mkdir(outDir); end
kernelCacheDir = fullfile(scriptDir, 'MR_MPS', 'KH_cache');
if ~exist(kernelCacheDir, 'dir'), mkdir(kernelCacheDir); end

%% ===================== 主循环 =====================
wb = [];
try
    wb = waitbar(0, '准备开始...', 'Name', 'MR ablation (2-stage)');
    cleanupWb = onCleanup(@() safe_close_waitbar(wb));
catch
    wb = [];
end

% 进度总量：按“数据集×(预处理+gamma搜索+configs×evalRep)”估计
if run_ablation
    numConfigs = 5;
else
    numConfigs = 1; % 仅 full
end
gammaSearchWork = numel(gamma_list) * repeat_times_gamma_search;
evalWork = numConfigs * repeat_times_eval;
totalWork = numel(dataIndices) * (gammaSearchWork + evalWork + 1); % +1 预处理/保存等
doneWork = 0;
lastWbUpdate = tic;
wb_update_every_s = 0.15;

for di = 1:numel(dataIndices)
    idx = dataIndices(di);
    dname = datasetName{idx};
    dataFile = fullfile(dataPath, [dname '.mat']);
    if ~isfile(dataFile)
        warning('跳过：数据文件不存在: %s', dataFile);
        continue;
    end

    fprintf('\n========== [%d/%d] Ablation (2-stage) on %s ==========\n', di, numel(dataIndices), dname);
    load(dataFile, 'fea', 'gt');
    gt = gt(:);
    num_cluster = length(unique(gt));

    % 与 campare 一致：优先读取预处理缓存；无缓存再生成并写回
    cacheFile = fullfile(kernelCacheDir, sprintf('%s_KH_HP_dim%d.mat', dname, dim_c));
    if isfile(cacheFile)
        fprintf('从缓存读取 KH / HP: %s\n', cacheFile);
        S = load(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c');
        KH = S.KH;
        HP = S.HP;
        num_kernel = S.num_kernel;
        if isfield(S, 'num_cluster_cache')
            num_cluster = S.num_cluster_cache;
        end
    else
        fprintf('缓存不存在，执行 preprocess 并写入缓存: %s\n', cacheFile);
        [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c);
        num_cluster_cache = num_cluster; 
        save(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c', '-v7.3');
    end
    doneWork = doneWork + 1;
    lastWbUpdate = maybe_update_waitbar(wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
        sprintf('[%d/%d] %s: KH/HP ready', di, numel(dataIndices), dname));
    dkey = matlab.lang.makeValidName(dname);
    if ~isfield(param_map, dkey)
        warning('跳过：未配置固定参数的数据集: %s', dname);
        continue;
    end
    best_param = param_map.(dkey);
    alpha_opt = best_param.alpha;
    selk_opt  = max(1, min(best_param.sel_k, num_kernel));
    eta_opt   = best_param.eta;
    fprintf('固定参数: alpha=%.2f, sel_k=%d, eta=%.3g\n', alpha_opt, selk_opt, eta_opt);
    [gamma_opt, gamma_acc, doneWork] = grid_search_dual_gamma(KH, HP, num_kernel, dim_c, num_cluster, gt, ...
        alpha_opt, selk_opt, eta_opt, opts_base, gamma_list, repeat_times_gamma_search, ...
        wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, sprintf('[%d/%d] %s: GammaSearch', di, numel(dataIndices), dname));
    fprintf('dual_dynamic_gamma 搜索最优: gamma=%.2f, ACC=%.4f\n', gamma_opt, gamma_acc);

    % ============ 组装各消融设置 ============
    configs = {};
    configs{end+1} = struct('name','full',      'dim_c',dim_c, 'sel_k',selk_opt,      'eta',eta_opt, 'static_on',true,  'dual_on',true,  'dual_gamma',gamma_opt);
    if run_ablation
        configs{end+1} = struct('name','woms',      'dim_c',1,     'sel_k',selk_opt,      'eta',eta_opt, 'static_on',true,  'dual_on',true,  'dual_gamma',gamma_opt);
        configs{end+1} = struct('name','woks',      'dim_c',dim_c, 'sel_k',num_kernel,    'eta',eta_opt, 'static_on',true,  'dual_on',true,  'dual_gamma',gamma_opt);
        configs{end+1} = struct('name','womr',      'dim_c',dim_c, 'sel_k',selk_opt,      'eta',0,       'static_on',false, 'dual_on',false, 'dual_gamma',0);
        configs{end+1} = struct('name','wodual',    'dim_c',dim_c, 'sel_k',selk_opt,      'eta',eta_opt, 'static_on',true,  'dual_on',false, 'dual_gamma',0);
    end

    % ============ 逐配置评估 ============
    results = struct();
    for ci = 1:numel(configs)
        cfg = configs{ci};
        fprintf('--- Eval %s ---\n', cfg.name);

        if cfg.dim_c == dim_c
            HP_use = HP;
        else
            % w/o MS：只保留第一尺度
            HP_use = keep_first_scale_HP(HP);
        end

        opts = opts_base;
        opts.static_term_on = cfg.static_on;
        opts.dual_dynamic_on = cfg.dual_on;
        opts.dual_dynamic_gamma = cfg.dual_gamma;
        % 评估阶段建议给足迭代次数，减少“早停”差异
        opts.maxIter = 50;

        runs = zeros(repeat_times_eval, 4); % [ACC NMI ARI Fscore]
        for rep = 1:repeat_times_eval
            rng(rep); % 固定复现
            [F, ~] = main_lx(KH, HP_use, num_kernel, cfg.dim_c, num_cluster, ...
                alpha_opt, cfg.sel_k, cfg.eta, opts);
            res = my_nmi_acc(real(F), gt, num_cluster);
            runs(rep, :) = [res(1,7), res(1,4), res(1,5), res(1,1)];

            doneWork = doneWork + 1;
            lastWbUpdate = maybe_update_waitbar(wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
                sprintf('[%d/%d] %s: %s (%d/%d)', di, numel(dataIndices), dname, cfg.name, rep, repeat_times_eval));
        end

        results.(cfg.name).mean = mean(runs, 1);
        results.(cfg.name).std  = std(runs, 0, 1);
        results.(cfg.name).runs = runs;
        results.(cfg.name).cfg  = cfg;

        fprintf('%s: ACC=%.2f±%.2f%%, NMI=%.2f±%.2f%%, ARI=%.2f±%.2f%%, F=%.2f±%.2f%%\n', ...
            cfg.name, ...
            results.(cfg.name).mean(1)*100, results.(cfg.name).std(1)*100, ...
            results.(cfg.name).mean(2)*100, results.(cfg.name).std(2)*100, ...
            results.(cfg.name).mean(3)*100, results.(cfg.name).std(3)*100, ...
            results.(cfg.name).mean(4)*100, results.(cfg.name).std(4)*100);
    end

    % ============ 保存 ============
    outMat = fullfile(outDir, sprintf('%s_ablation_2stage.mat', dname));
    save(outMat, 'dname', 'alpha_opt', 'selk_opt', 'eta_opt', 'results', '-v7.3');

    outCsv = fullfile(outDir, sprintf('%s_ablation_2stage.csv', dname));
    fid = fopen(outCsv, 'w');
    fprintf(fid, 'dataset,config,alpha,sel_k,eta,static_on,dual_on,dual_gamma,ACC_mean,ACC_std,NMI_mean,NMI_std,ARI_mean,ARI_std,F_mean,F_std\n');
    fns = fieldnames(results);
    for ii = 1:numel(fns)
        nm = fns{ii};
        rr = results.(nm);
        fprintf(fid, '%s,%s,%.8g,%d,%.8g,%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
            dname, nm, alpha_opt, rr.cfg.sel_k, rr.cfg.eta, rr.cfg.static_on, rr.cfg.dual_on, rr.cfg.dual_gamma, ...
            rr.mean(1), rr.std(1), rr.mean(2), rr.std(2), rr.mean(3), rr.std(3), rr.mean(4), rr.std(4));
    end
    fclose(fid);

    fprintf('已保存: %s\n', outCsv);
end

function HP_single = keep_first_scale_HP(HP)
num_kernel = numel(HP);
HP_single = cell(num_kernel, 1);
for v = 1:num_kernel
    Hp_full = HP{v};
    HP_single{v} = cell(1, 1);
    HP_single{v}{1} = Hp_full{1};
end
end

function [best_gamma, best_acc, doneWork] = grid_search_dual_gamma(KH, HP, num_kernel, dim_c, num_cluster, gt, ...
    alpha, sel_k, eta, opts_base, gamma_list, repeat_times, ...
    wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, wbPrefix)
best_acc = -inf;
best_gamma = opts_base.dual_dynamic_gamma;
fprintf('开始 dual\\_dynamic\\_gamma 网格搜索（alpha=%.2f, sel\\_k=%d, eta=%.3g）\n', alpha, sel_k, eta);
for ig = 1:numel(gamma_list)
    gamma = gamma_list(ig);
    acc_runs = zeros(repeat_times, 1);
    for r = 1:repeat_times
        rng(r);
        opts = opts_base;
        opts.dual_dynamic_gamma = gamma;
        [F, ~] = main_lx(KH, HP, num_kernel, dim_c, num_cluster, alpha, sel_k, eta, opts);
        res = my_nmi_acc(real(F), gt, num_cluster);
        acc_runs(r) = res(1,7);
        doneWork = doneWork + 1;
        lastWbUpdate = maybe_update_waitbar(wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
            sprintf('%s gamma=%.2f (%d/%d)', wbPrefix, gamma, r, repeat_times));
    end
    acc = mean(acc_runs);
    acc_std = std(acc_runs, 0, 1);
    fprintf('  gamma=%.2f -> ACC=%.4f (std=%.4f)\n', gamma, acc, acc_std);
    if acc > best_acc
        best_acc = acc;
        best_gamma = gamma;
    end
end
fprintf('gamma 搜索完成：best gamma=%.2f, best ACC=%.4f\n', best_gamma, best_acc);
end

function [res] = my_nmi_acc(F_normalized, gt, num_cluster)
% 与 campare_end.m 中一致：多次 kmeans 取均值/方差
F_normalized = real(F_normalized);
max_rep = 20;
res_rep = zeros(max_rep, 8);
F_normalized = F_normalized ./ repmat(sqrt(sum(F_normalized.^2,2)),1,size(F_normalized,2));

for rep = 1:max_rep
    pre = kmeans(F_normalized, num_cluster, ...
        'maxiter', 100, 'replicates', 20, 'emptyaction', 'singleton');
    res_rep(rep,:) = Clustering8Measure(gt, pre);
end
res = [mean(res_rep); std(res_rep)];
end

%% ===================== waitbar helpers =====================
function lastWbUpdate = maybe_update_waitbar(wb, doneWork, totalWork, lastWbUpdate, every_s, msg)
if isempty(wb) || ~ishandle(wb)
    return;
end
if toc(lastWbUpdate) < every_s
    return;
end
frac = min(max(doneWork / max(totalWork, 1), 0), 1);
try
    waitbar(frac, wb, sprintf('%s\n%.1f%% (%d/%d)', msg, 100*frac, doneWork, totalWork));
catch
end
lastWbUpdate = tic;
end

function safe_close_waitbar(wb)
try
    if ~isempty(wb) && ishandle(wb)
        close(wb);
    end
catch
end
end

