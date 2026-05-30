% MR_ablation_2stage_static.m
% 对应当前 main_lx_end.m：只保留静态图流形项
%
% 消融项：
%  - full：搜索 alpha, sel_k, eta，启用静态流形项
%  - womr：去掉全部流形项，eta=0
%
% 输出：
%  每个数据集写入 MR_MPS/ablation_2stage_static_results/*.csv 与 *.mat

close all; clear; clc; warning off;

%% ===================== Path =====================
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end

addpath(scriptDir);  % main_lx_end.m
addpath(fullfile(scriptDir, 'ClusteringMeasure'));
addpath(fullfile(scriptDir, 'FormulationKernels'));
addpath(fullfile(scriptDir, 'function'));
addpath(fullfile(scriptDir, 'MR_MPS'));

%% ===================== 数据集设置 =====================
dataPath = fullfile(scriptDir, 'Datasets', 'Multi_view-Datasets');

datasetName = { ...
    'MSRC_v1', ...
    'BBCSport', ...
    'BRCA', ...
    '3sources', ...
    'WebKB', ...
    'Prokaryotic'};

run_all_datasets = true;
dataIndex = 1;

if run_all_datasets
    dataIndices = 4:numel(datasetName);
else
    dataIndices = dataIndex;
end

%% ===================== 2-stage 搜索设置 =====================
repeat_times_search = 1;

alpha_list = 0.1:0.1:0.9;
sel_ratio_list = 0.1:0.1:0.9;

eta_list_coarse = [0.01, 0.1, 1, 10, 100];
eta_stage1 = 1;

%% ===================== main_lx_end 参数 =====================
opts_base = struct();

opts_base.kNN_graph = 15;
opts_base.maxIter = 50;
opts_base.tol = 1e-6;
opts_base.irls_eps = 1e-8;
opts_base.static_term_on = true;

%% ===================== 消融评估设置 =====================
repeat_times_eval = 10;
dim_c = 5;

run_ablation = true;

outDir = fullfile(scriptDir, 'MR_MPS', 'ablation_2stage_static_results');
if ~exist(outDir, 'dir'), mkdir(outDir); end

kernelCacheDir = fullfile(scriptDir, 'MR_MPS', 'KH_cache');
if ~exist(kernelCacheDir, 'dir'), mkdir(kernelCacheDir); end

%% ===================== 进度条 =====================
wb = [];
try
    wb = waitbar(0, '准备开始...', 'Name', 'MR ablation static only');
    cleanupWb = onCleanup(@() safe_close_waitbar(wb));
catch
    wb = [];
end

if run_ablation
    numConfigs = 2;
else
    numConfigs = 1;
end

if run_ablation
    stage1Work = numel(alpha_list) * numel(sel_ratio_list) * repeat_times_search * numConfigs;

    stage2Work = ...
        numel(eta_list_coarse) * repeat_times_search + ...   % full
        1 * repeat_times_search;                              % womr
else
    stage1Work = numel(alpha_list) * numel(sel_ratio_list) * repeat_times_search;
    stage2Work = numel(eta_list_coarse) * repeat_times_search;
end

evalWork = numConfigs * repeat_times_eval;

totalWork = numel(dataIndices) * (stage1Work + stage2Work + evalWork + 1);
doneWork = 0;

lastWbUpdate = tic;
wb_update_every_s = 0.15;

%% ===================== 主循环 =====================
for di = 1:numel(dataIndices)

    idx = dataIndices(di);
    dname = datasetName{idx};
    dataFile = fullfile(dataPath, [dname '.mat']);

    if ~isfile(dataFile)
        warning('跳过：数据文件不存在: %s', dataFile);
        continue;
    end

    fprintf('\n========== [%d/%d] Static Ablation on %s ==========\n', ...
        di, numel(dataIndices), dname);

    load(dataFile, 'fea', 'gt');
    gt = gt(:);
    num_cluster = length(unique(gt));

    %% ============ 读取或生成 KH / HP ============
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

        save(cacheFile, 'KH', 'HP', 'num_kernel', ...
            'num_cluster_cache', 'dim_c', '-v7.3');
    end

    doneWork = doneWork + 1;

    lastWbUpdate = maybe_update_waitbar( ...
        wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
        sprintf('[%d/%d] %s: KH/HP ready', di, numel(dataIndices), dname));

    %% ============ sel_k 候选 ============
    sel_list = ceil(sel_ratio_list * num_kernel);
    sel_list = max(1, min(sel_list, num_kernel));
    sel_list = unique(sel_list);

    %% ============ 消融配置 ============
    configs = {};

    configs{end+1} = struct( ...
        'name', 'full', ...
        'dim_c', dim_c, ...
        'eta_on', true, ...
        'static_term_on', true);

    if run_ablation
        configs{end+1} = struct( ...
            'name', 'womr', ...
            'dim_c', dim_c, ...
            'eta_on', false, ...
            'static_term_on', false);
    end

    %% ============ 逐配置搜索与评估 ============
    results = struct();

    for ci = 1:numel(configs)

        cfg = configs{ci};

        fprintf('--- Search + Eval %s ---\n', cfg.name);

        [best, doneWork, lastWbUpdate] = search_best_for_config( ...
            KH, HP, num_kernel, cfg.dim_c, num_cluster, gt, ...
            alpha_list, sel_list, eta_stage1, eta_list_coarse, ...
            opts_base, cfg, repeat_times_search, ...
            wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
            sprintf('[%d/%d] %s: %s', di, numel(dataIndices), dname, cfg.name));

        fprintf('%s 最优: alpha=%.2f, sel_k=%d, eta=%.3g, ACC=%.4f\n', ...
            cfg.name, best.alpha, best.sel_k, best.eta, best.ACC);

        opts = opts_for_config(opts_base, cfg);
        opts.maxIter = 50;

        if cfg.eta_on
            eta_eval = best.eta;
        else
            eta_eval = 0;
        end

        runs = zeros(repeat_times_eval, 4); % [ACC NMI ARI Fscore]

        for rep = 1:repeat_times_eval

            rng(rep);

            [F, ~] = main_lx_end(KH, HP, num_kernel, cfg.dim_c, num_cluster, ...
                best.alpha, best.sel_k, eta_eval, opts);

            res = my_nmi_acc(real(F), gt, num_cluster);
            runs(rep, :) = [res(1,7), res(1,4), res(1,5), res(1,1)];

            doneWork = doneWork + 1;

            lastWbUpdate = maybe_update_waitbar( ...
                wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
                sprintf('[%d/%d] %s: %s (%d/%d)', ...
                di, numel(dataIndices), dname, cfg.name, rep, repeat_times_eval));
        end

        results.(cfg.name).mean = mean(runs, 1);
        results.(cfg.name).std  = std(runs, 0, 1);
        results.(cfg.name).runs = runs;
        results.(cfg.name).cfg  = cfg;
        results.(cfg.name).best = best;

        fprintf('%s: ACC=%.2f±%.2f%%, NMI=%.2f±%.2f%%, ARI=%.2f±%.2f%%, F=%.2f±%.2f%%\n', ...
            cfg.name, ...
            results.(cfg.name).mean(1)*100, results.(cfg.name).std(1)*100, ...
            results.(cfg.name).mean(2)*100, results.(cfg.name).std(2)*100, ...
            results.(cfg.name).mean(3)*100, results.(cfg.name).std(3)*100, ...
            results.(cfg.name).mean(4)*100, results.(cfg.name).std(4)*100);
    end

    %% ============ 保存 ============
    outMat = fullfile(outDir, sprintf('%s_ablation_2stage_static.mat', dname));

    search_grid = struct( ...
        'alpha_list', alpha_list, ...
        'sel_ratio_list', sel_ratio_list, ...
        'sel_list', sel_list, ...
        'eta_list', eta_list_coarse);

    save(outMat, 'dname', 'search_grid', 'run_ablation', 'results', '-v7.3');

    outCsv = fullfile(outDir, sprintf('%s_ablation_2stage_static.csv', dname));

    fid = fopen(outCsv, 'w');

    if fid < 0
        warning('无法写入 CSV 文件: %s', outCsv);
    else
        fprintf(fid, ...
            'dataset,config,alpha,sel_k,eta,static_term_on,search_ACC,ACC_mean,ACC_std,NMI_mean,NMI_std,ARI_mean,ARI_std,F_mean,F_std\n');

        fns = fieldnames(results);

        for ii = 1:numel(fns)
            nm = fns{ii};
            rr = results.(nm);

            fprintf(fid, ...
                '%s,%s,%.8g,%d,%.8g,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
                dname, nm, ...
                rr.best.alpha, ...
                rr.best.sel_k, ...
                rr.best.eta, ...
                rr.cfg.static_term_on, ...
                rr.best.ACC, ...
                rr.mean(1), rr.std(1), ...
                rr.mean(2), rr.std(2), ...
                rr.mean(3), rr.std(3), ...
                rr.mean(4), rr.std(4));
        end

        fclose(fid);
    end

    fprintf('已保存: %s\n', outCsv);
end

%% ===================== 2-stage helpers =====================
function [best, doneWork, lastWbUpdate] = search_best_for_config( ...
    KH, HP, num_kernel, dim_c, num_cluster, gt, ...
    alpha_list, sel_list, eta_stage1, eta_list, ...
    opts_base, cfg, repeat_times, ...
    wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, wbPrefix)

if cfg.eta_on
    stage1_eta = eta_stage1;
else
    stage1_eta = 0;
end

opts_stage1 = opts_for_config(opts_base, cfg);

best1.ACC = -inf;
best1.alpha = NaN;
best1.sel_k = NaN;

%% Stage 1: search alpha and sel_k
for ia = 1:numel(alpha_list)

    alpha = alpha_list(ia);

    for ik = 1:numel(sel_list)

        sel_k = sel_list(ik);

        acc_runs = zeros(repeat_times, 1);

        for r = 1:repeat_times

            rng(r);

            [F, ~] = main_lx_end(KH, HP, num_kernel, dim_c, num_cluster, ...
                alpha, sel_k, stage1_eta, opts_stage1);

            res = my_nmi_acc(real(F), gt, num_cluster);
            acc_runs(r) = res(1,7);

            doneWork = doneWork + 1;

            lastWbUpdate = maybe_update_waitbar( ...
                wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
                sprintf('%s Stage1 alpha=%.2f sel_k=%d (%d/%d)', ...
                wbPrefix, alpha, sel_k, r, repeat_times));
        end

        acc = mean(acc_runs);

        if acc > best1.ACC
            best1.ACC = acc;
            best1.alpha = alpha;
            best1.sel_k = sel_k;
        end
    end
end

%% Stage 2: search eta
if cfg.eta_on
    eta_candidates = eta_list;
else
    eta_candidates = 0;
end

best.ACC = -inf;
best.alpha = best1.alpha;
best.sel_k = best1.sel_k;
best.eta = NaN;
best.stage1_ACC = best1.ACC;

opts_stage2 = opts_for_config(opts_base, cfg);

for ie = 1:numel(eta_candidates)

    eta = eta_candidates(ie);

    acc_runs = zeros(repeat_times, 1);

    for r = 1:repeat_times

        rng(r);

        [F, ~] = main_lx_end(KH, HP, num_kernel, dim_c, num_cluster, ...
            best1.alpha, best1.sel_k, eta, opts_stage2);

        res = my_nmi_acc(real(F), gt, num_cluster);
        acc_runs(r) = res(1,7);

        doneWork = doneWork + 1;

        lastWbUpdate = maybe_update_waitbar( ...
            wb, doneWork, totalWork, lastWbUpdate, wb_update_every_s, ...
            sprintf('%s Stage2 eta=%.3g (%d/%d)', ...
            wbPrefix, eta, r, repeat_times));
    end

    acc = mean(acc_runs);

    if acc > best.ACC
        best.ACC = acc;
        best.eta = eta;
    end
end

end

function opts = opts_for_config(opts_base, cfg)

opts = opts_base;
opts.static_term_on = cfg.static_term_on;

end

function [res] = my_nmi_acc(F_normalized, gt, num_cluster)

F_normalized = real(F_normalized);

row_norm = sqrt(sum(F_normalized.^2, 2));
row_norm(row_norm <= eps) = 1;
F_normalized = F_normalized ./ repmat(row_norm, 1, size(F_normalized, 2));

max_rep = 20;
res_rep = zeros(max_rep, 8);

for rep = 1:max_rep
    pre = kmeans(F_normalized, num_cluster, ...
        'maxiter', 100, ...
        'replicates', 20, ...
        'emptyaction', 'singleton');

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
    waitbar(frac, wb, sprintf('%s\n%.1f%% (%d/%d)', ...
        msg, 100*frac, doneWork, totalWork));
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