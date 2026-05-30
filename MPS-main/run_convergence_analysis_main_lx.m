%% 收敛性分析

close all; clear; clc; warning off;

%% ===================== 用户配置 =====================
dim_c = 5;
rng_seed = 1;

opts_conv = struct();
% opts_conv.maxIter = 100;
% opts_conv.tol = 1e-8;

save_figures = true;
use_semilogy_rel = true;

%% ===================== 路径 =====================
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end

addpath(genpath(baseDir), '-begin');
addpath(fullfile(baseDir, 'FormulationKernels'));
addpath(fullfile(baseDir, 'function'));
addpath(fullfile(baseDir, 'ClusteringMeasure'));
addpath(fullfile(baseDir, 'MR_MPS'));

dataPath = fullfile(baseDir, 'Datasets', 'Multi_view-Datasets');
kernelCacheDir = fullfile(baseDir, 'MR_MPS', 'KH_cache');
if ~exist(kernelCacheDir, 'dir'), mkdir(kernelCacheDir); end

outDir = fullfile(baseDir, 'MR_MPS', 'convergence_analysis');
if save_figures && ~exist(outDir, 'dir')
    mkdir(outDir);
end

datasetNameList = {'MSRC_v1', 'BBCSport', 'BRCA', '3sources', 'WebKB', 'Prokaryotic'};
n_ds = numel(datasetNameList);

convData = repmat(struct('ok', false, 'obj', [], 'rel', [], 'nIt', 0), n_ds, 1);

%% ===================== 各数据集运行 main_lx 收集 obj =====================
for idx_ds = 1:n_ds

    datasetName = datasetNameList{idx_ds};
    fprintf('\n[%d/%d] 收敛曲线: %s\n', idx_ds, n_ds, datasetName);

    dataFile = fullfile(dataPath, [datasetName, '.mat']);
    if ~isfile(dataFile)
        warning('跳过（无数据）: %s', dataFile);
        continue;
    end

    load(dataFile, 'fea', 'gt');

    Y = gt(:);
    u = unique(Y);
    u(u < 1) = [];

    for ii = 1:length(u)
        Y(Y == u(ii)) = ii;
    end

    num_cluster = length(unique(Y));

    cacheFile = fullfile(kernelCacheDir, sprintf('%s_KH_HP_dim%d.mat', datasetName, dim_c));

    if isfile(cacheFile)
        S = load(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c');

        KH = S.KH;
        HP = S.HP;
        num_kernel = S.num_kernel;

        if isfield(S, 'num_cluster_cache')
            num_cluster = S.num_cluster_cache;
        end
    else
        fprintf('  无 KH 缓存，preprocess 并写入...\n');

        [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c);

        num_cluster_cache = num_cluster;

        save(cacheFile, 'KH', 'HP', 'num_kernel', ...
            'num_cluster_cache', 'dim_c', '-v7.3');
    end

    try
        fixed_full = get_fixed_mr_params(datasetName, 'full');
    catch ME
        warning('跳过 %s: %s', datasetName, ME.message);
        continue;
    end

    alpha0 = fixed_full.alpha;
    eta0   = fixed_full.eta;
    lambda0 = fixed_full.lambda;

    k0 = max(1, min(round(fixed_full.sel_k), num_kernel));

    opts_run = opts_lx_eval_from_fixed(fixed_full);
    opts_run = struct_overlay(opts_run, opts_conv);

    rng(rng_seed);

    [~, obj] = main_lx(KH, HP, num_kernel, dim_c, num_cluster, ...
        alpha0, k0, eta0, opts_run);

    obj = obj(:);
    obj = obj(isfinite(obj));

    nIt = numel(obj);

    if nIt < 1
        warning('跳过 %s: obj 为空', datasetName);
        continue;
    end

    rel = relative_change_like_main_lx(obj);

    convData(idx_ds).ok = true;
    convData(idx_ds).obj = obj;
    convData(idx_ds).rel = rel;
    convData(idx_ds).nIt = nIt;

    fprintf('  alpha=%.3g, sel_k=%d, eta=%.3g, lambda=%.3g\n', ...
        alpha0, k0, eta0, lambda0);

    fprintf('  迭代次数 nIt=%d, 末值 obj=%.6g\n', nIt, obj(end));
end

%% ===================== 图窗1：目标函数迭代值 =====================
figObj = figure('Color', 'w', ...
    'Name', 'main_lx 目标函数收敛', ...
    'Position', [40, 40, 1300, 720]);

for idx_ds = 1:n_ds
    subplot(2, 3, idx_ds);

    if convData(idx_ds).ok
        obj = convData(idx_ds).obj;
        nIt = convData(idx_ds).nIt;

        plot(1:nIt, obj, '-', ...
            'LineWidth', 1.35, ...
            'Color', [0.0, 0.45, 0.74]);

        grid on;
        xlabel('迭代 t');
        ylabel('Q(t)');
        xlim([1, max(nIt, 1)]);
    else
        axis([0, 1, 0, 1]);
        axis off;

        text(0.5, 0.5, sprintf('%s\n(跳过)', datasetNameList{idx_ds}), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end

    title(datasetNameList{idx_ds}, 'Interpreter', 'none');
end

sgtitle(figObj, ...
    '各数据集 main\_lx 代理目标函数 Q(t) 收敛曲线', ...
    'Interpreter', 'none');

%% ===================== 图窗2：相对变化 =====================
figRel = figure('Color', 'w', ...
    'Name', 'main_lx 目标函数相对变化', ...
    'Position', [80, 80, 1300, 720]);

for idx_ds = 1:n_ds
    subplot(2, 3, idx_ds);

    if convData(idx_ds).ok && convData(idx_ds).nIt >= 2

        rel = convData(idx_ds).rel(:);
        nIt = convData(idx_ds).nIt;

        tRel = (2:nIt)';
        yRel = rel(2:end);

        mask = isfinite(yRel) & (yRel > 0);

        if use_semilogy_rel && any(mask)
            semilogy(tRel(mask), yRel(mask), '-', ...
                'LineWidth', 1.35, ...
                'Color', [0.85, 0.33, 0.10]);
        else
            plot(tRel, yRel, '-', ...
                'LineWidth', 1.35, ...
                'Color', [0.85, 0.33, 0.10]);
        end

        grid on;
        xlabel('迭代 t');
        ylabel('|Q_{t-1}-Q_t| / max(|Q_{t-1}|,\epsilon)');
        xlim([2, max(nIt, 2)]);
    else
        axis([0, 1, 0, 1]);
        axis off;

        text(0.5, 0.5, sprintf('%s\n(跳过)', datasetNameList{idx_ds}), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end

    title(datasetNameList{idx_ds}, 'Interpreter', 'none');
end

sgtitle(figRel, ...
    '各数据集代理目标函数相邻迭代相对变化', ...
    'Interpreter', 'none');

%% ===================== 保存图片 =====================
stamp = datestr(now, 'yyyymmdd_HHMMSS');

if save_figures
    p1 = fullfile(outDir, sprintf('convergence_objective_%s.png', stamp));
    p2 = fullfile(outDir, sprintf('convergence_relchange_%s.png', stamp));

    try
        exportgraphics(figObj, p1, 'Resolution', 300);
        exportgraphics(figRel, p2, 'Resolution', 300);
    catch
        saveas(figObj, p1);
        saveas(figRel, p2);
    end

    fprintf('\n已保存:\n  %s\n  %s\n', p1, p2);
end

fprintf('\n完成。图窗1=代理目标函数；图窗2=相对变化。\n');

%% ===================== 本地函数 =====================
function rel = relative_change_like_main_lx(obj)

obj = obj(:);
n = numel(obj);
rel = nan(n, 1);

for t = 2:n
    rel(t) = abs(obj(t - 1) - obj(t)) / max(abs(obj(t - 1)), eps);
end

end

function opts = build_opts_lx()

opts = struct();

opts.kNN_graph = 15;
opts.maxIter = 50;
opts.tol = 1e-6;
opts.irls_eps = 1e-8;

opts.adaptive_graph_on = true;
opts.adaptive_graph_lambda = 1;
opts.adaptive_graph_warmup = 1;

end

function opts = opts_lx_eval_from_fixed(fixed_full)

opts = build_opts_lx();

opts.adaptive_graph_on = logical(fixed_full.adaptive_graph_on);
opts.adaptive_graph_lambda = fixed_full.lambda;

end

function p = get_fixed_mr_params(dataset_name, config_name)

switch dataset_name

    case 'MSRC_v1'
        switch config_name
            case 'full'
                p = struct( ...
                    'alpha', 0.90, ...
                    'sel_k', 15, ...
                    'eta', 10, ...
                    'lambda', 0, ...
                    'adaptive_graph_on', 0);
            otherwise
                error('本脚本仅使用 config full');
        end

    case 'BBCSport'
        switch config_name
            case 'full'
                p = struct( ...
                    'alpha', 0.10, ...
                    'sel_k', 8, ...
                    'eta', 100, ...
                    'lambda', 0, ...
                    'adaptive_graph_on', 0);
            otherwise
                error('本脚本仅使用 config full');
        end

    case 'BRCA'
        switch config_name
            case 'full'
                p = struct( ...
                    'alpha', 0.20, ...
                    'sel_k', 16, ...
                    'eta', 1, ...
                    'lambda', 0, ...
                    'adaptive_graph_on', 0);
            otherwise
                error('本脚本仅使用 config full');
        end

    case 'WebKB'
        switch config_name
            case 'full'
                p = struct( ...
                    'alpha', 0.40, ...
                    'sel_k', 6, ...
                    'eta', 1, ...
                    'lambda', 0, ...
                    'adaptive_graph_on', 0);
            otherwise
                error('本脚本仅使用 config full');
        end

    case '3sources'
        switch config_name
            case 'full'
                p = struct( ...
                    'alpha', 0.20, ...
                    'sel_k', 14, ...
                    'eta', 1, ...
                    'lambda', 0, ...
                    'adaptive_graph_on', 0);
            otherwise
                error('本脚本仅使用 config full');
        end

    case 'Prokaryotic'
        switch config_name
            case 'full'
                p = struct( ...
                    'alpha', 0.40, ...
                    'sel_k', 3, ...
                    'eta', 1, ...
                    'lambda', 0, ...
                    'adaptive_graph_on', 0);
            otherwise
                error('本脚本仅使用 config full');
        end

    otherwise
        error('get_fixed_mr_params: 未配置数据集 %s', dataset_name);
end

end

function out = struct_overlay(base, overlay)

out = base;

if nargin < 2 || isempty(overlay)
    return;
end

fn = fieldnames(overlay);

for t = 1:numel(fn)
    out.(fn{t}) = overlay.(fn{t});
end

end