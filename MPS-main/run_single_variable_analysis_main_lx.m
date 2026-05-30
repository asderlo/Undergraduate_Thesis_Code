%% 单参数敏感性分析
close all; clear; clc; warning off;

%% ===================== 用户配置（与 campare_end MR 段对齐的可调项）=====================
dim_c         = 5;
repeat_times  = 1;         % campare_end: repeat_mr_eval

analysis_list = {'alpha', 'eta', 'sel_kernel', 'gamma'};
pick_idx      = 1;         % 1..numel(analysis_list)
analysis_var  = analysis_list{pick_idx};

value_list    = [];         % []：按 default_value_grid(analysis_var, num_kernel)

opts_fixed = struct();      % 在 campare 的 opts_lx_eval 之上再覆盖字段（非扫描变量）

save_results = false;       % true：逐数据集 mat/png + batch_summary.mat
save_combined_figure = true; % true：保存六合一汇总图（MPS_LX/single_variable_runs）

%% ===================== 路径（与 campare_end 对 MPS-main 的依赖一致）=====================
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
addpath(genpath(baseDir), '-begin');
addpath(fullfile(baseDir, 'FormulationKernels'));
addpath(fullfile(baseDir, 'function'));
addpath(fullfile(baseDir, 'ClusteringMeasure'));
addpath(fullfile(baseDir, 'new'));
addpath(fullfile(baseDir, 'MR_MPS'));

dataPath = fullfile(baseDir, 'Datasets', 'Multi_view-Datasets');
lxRoot = fullfile(baseDir, 'MPS_LX');
kernelCacheDir = fullfile(lxRoot, 'KH_cache');
if ~exist(kernelCacheDir, 'dir'), mkdir(kernelCacheDir); end
kernelCacheLegacy = fullfile(baseDir, 'MR_MPS', 'KH_cache');

outDir = fullfile(lxRoot, 'single_variable_runs');
if (save_results || save_combined_figure) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

% 与 campare_end.m 第 56-57 行一致
datasetNameList = {'MSRC_v1', 'BBCSport', 'BRCA', '3sources', 'WebKB', 'Prokaryotic'};
assert(numel(datasetNameList) == 6, '须为 campare_end 中的 6 个数据集。');

fprintf('\n单变量分析（main\\_lx：扫描 alpha/eta/k；其余定值同 campare_end）\n将处理: %s\n', strjoin(datasetNameList, ', '));

stamp_batch = datestr(now, 'yyyymmdd_HHMMSS');
n_ds = numel(datasetNameList);
plotData = repmat(struct('ok', false, 'metrics_mean', [], 'metrics_std', [], ...
    'value_list_ds', [], 'npt', 0, 'best_val', []), n_ds, 1);
summary = struct('dataset', {}, 'analysis_var', {}, 'value_list', {}, ...
    'metrics_mean', {}, 'metrics_std', {}, 'best_val', {}, ...
    'alpha0', {}, 'eta0', {}, 'k0', {}, 'fixed_full', {}, 'mat_file', {});

%% ===================== 按数据集循环 =====================
for idx_ds = 1:numel(datasetNameList)
    datasetName = datasetNameList{idx_ds};
    fprintf('\n======================================================\n');
    fprintf('数据集 %s (%d/%d)\n', datasetName, idx_ds, numel(datasetNameList));
    fprintf('======================================================\n');

    dataFile = fullfile(dataPath, [datasetName, '.mat']);
    if ~isfile(dataFile)
        warning('run_single_variable_analysis_main_lx:SkipMissing', '跳过（无数据文件）: %s', dataFile);
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
    cacheLegacy = fullfile(kernelCacheLegacy, sprintf('%s_KH_HP_dim%d.mat', datasetName, dim_c));
    if isfile(cacheFile)
        fprintf('从缓存读取 KH / HP: %s\n', cacheFile);
        S = load(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c');
        KH = S.KH;
        HP = S.HP;
        num_kernel = S.num_kernel;
        if isfield(S, 'num_cluster_cache')
            num_cluster = S.num_cluster_cache;
        end
    elseif isfile(cacheLegacy)
        fprintf('从旧缓存读取 KH / HP: %s\n', cacheLegacy);
        S = load(cacheLegacy, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c');
        KH = S.KH;
        HP = S.HP;
        num_kernel = S.num_kernel;
        if isfield(S, 'num_cluster_cache')
            num_cluster = S.num_cluster_cache;
        end
    else
        fprintf('无缓存，preprocess 并写入（与 campare_end 一致）...\n');
        tic;
        [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c);
        t_prep = toc;
        num_cluster_cache = num_cluster;
        save(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c', '-v7.3');
        fprintf('已缓存 KH/HP: %s (耗时 %.2fs)\n', cacheFile, t_prep);
    end

    try
        fixed_full = get_fixed_mr_params(datasetName, 'full');
    catch ME
        warning('run_single_variable_analysis_main_lx:FixedParams', ...
            '跳过 %s：get_fixed_mr_params 失败 — %s', datasetName, ME.message);
        continue;
    end

    alpha0 = fixed_full.alpha;
    eta0   = fixed_full.eta;
    k0     = max(1, min(round(fixed_full.sel_k), num_kernel));

    fprintf('定值: alpha=%.4g, sel_k=%d, eta=%.4g\n', alpha0, k0, eta0);
    fprintf('  static_on=%g\n', fixed_full.static_on);

    value_list_ds = value_list;
    if isempty(value_list_ds)
        try
            value_list_ds = default_value_grid(analysis_var, num_kernel);
        catch ME
            warning('run_single_variable_analysis_main_lx:Grid', '跳过 %s：%s', datasetName, ME.message);
            continue;
        end
    end
    value_list_ds = value_list_ds(:);
    npt = numel(value_list_ds);

    % 四指标与 campare_end 一致：ACC, NMI, ARI, F-score（my_nmi_acc 行向量列 7,4,5,1）
    n_metrics = 4;
    metrics_mean = zeros(npt, n_metrics);
    metrics_std  = zeros(npt, n_metrics);

    for ipt = 1:npt
        alpha = alpha0;
        eta   = eta0;
        sk    = k0;
        opts  = opts_lx_eval_from_campare(fixed_full);
        opts  = struct_overlay(opts, opts_fixed);
        [alpha, eta, sk, opts] = apply_single_override( ...
            analysis_var, value_list_ds(ipt), alpha, eta, sk, opts, num_kernel);

        fprintf('  [%d/%d] %s = %s\n', ipt, npt, analysis_var, mat2str(value_list_ds(ipt)));

        runs_m = zeros(repeat_times, n_metrics);
        for rep = 1:repeat_times
            rng(rep);
            [F, ~] = main_lx(KH, HP, num_kernel, dim_c, num_cluster, ...
                alpha, sk, eta, opts);
            res = my_nmi_acc(real(F), Y, num_cluster);
            b = res(1, :);
            runs_m(rep, :) = [b(7), b(4), b(5), b(1)];
        end
        metrics_mean(ipt, :) = mean(runs_m, 1) * 100;
        metrics_std(ipt, :)  = std(runs_m, 0, 1) * 100;
    end

    [~, imax] = max(metrics_mean(:, 1));
    best_val = value_list_ds(imax);
    fprintf('  >> 按 ACC 取最优格点: %s = %s | ACC=%.2f NMI=%.2f ARI=%.2f F=%.2f (%%)\n', ...
        analysis_var, mat2str(best_val), metrics_mean(imax, 1), metrics_mean(imax, 2), ...
        metrics_mean(imax, 3), metrics_mean(imax, 4));

    plotData(idx_ds).ok = true;
    plotData(idx_ds).metrics_mean = metrics_mean;
    plotData(idx_ds).metrics_std = metrics_std;
    plotData(idx_ds).value_list_ds = value_list_ds;
    plotData(idx_ds).npt = npt;
    plotData(idx_ds).best_val = best_val;

    if save_results
        saveFile = fullfile(outDir, sprintf('%s_sweep_%s_%s.mat', datasetName, analysis_var, stamp_batch));
        save(saveFile, 'datasetName', 'analysis_var', 'value_list_ds', 'alpha0', 'eta0', 'k0', ...
            'fixed_full', 'opts_fixed', 'metrics_mean', 'metrics_std', 'repeat_times', 'dim_c', 'num_kernel');
        fprintf('  已保存: %s\n', saveFile);
        fig = figure('Color', 'w', 'Name', sprintf('%s: %s', datasetName, analysis_var), 'Visible', 'off');
        pd_one = struct('ok', true, 'metrics_mean', metrics_mean, 'metrics_std', metrics_std, ...
            'value_list_ds', value_list_ds, 'npt', npt, 'best_val', best_val);
        plot_four_metrics_on_axes(gca, pd_one, analysis_var, true);
        title(sprintf('%s 单变量（定值同 campare MR-MPS）', datasetName));
        xtl = cell(npt, 1);
        for ii = 1:npt
            xtl{ii} = sprintf('%.4g', value_list_ds(ii));
        end
        set(gca, 'XTick', 1:npt, 'XTickLabel', xtl);
        set(gca, 'TickLabelInterpreter', 'none'); % <-- 添加：使数字刻度标签直立
        % if exist('xtickangle', 'file') == 2
        %     xtickangle(45);
        % end
        pngFile = fullfile(outDir, sprintf('%s_sweep_%s_%s.png', datasetName, analysis_var, stamp_batch));
        try
            exportgraphics(fig, pngFile, 'Resolution', 300);
        catch
            saveas(fig, pngFile);
        end
        close(fig);
        fprintf('  图: %s\n', pngFile);
        s = length(summary) + 1;
        summary(s).dataset = datasetName;
        summary(s).analysis_var = analysis_var;
        summary(s).value_list = value_list_ds;
        summary(s).metrics_mean = metrics_mean;
        summary(s).metrics_std = metrics_std;
        summary(s).best_val = best_val;
        summary(s).alpha0 = alpha0;
        summary(s).eta0 = eta0;
        summary(s).k0 = k0;
        summary(s).fixed_full = fixed_full;
        summary(s).mat_file = saveFile;
    end
end

%% ===================== 六数据集一窗汇总图 =====================
figAll = figure('Color', 'w', 'Name', sprintf('单变量 %s | 六数据集', analysis_var), ...
    'Position', [80, 80, 1280, 720]);
for idx_ds = 1:n_ds
    subplot(2, 3, idx_ds);
    plot_one_dataset_tile(plotData(idx_ds), datasetNameList{idx_ds}, analysis_var);
end
sgtitle(figAll, sprintf('单变量 %s', analysis_var));

if save_combined_figure
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    combinedPng = fullfile(outDir, sprintf('six_datasets_sweep_%s_%s.png', analysis_var, stamp_batch));
    
    % 删除 sgtitle
    delete(findobj(figAll, 'Type', 'Text', 'Tag', 'sgtitle'));
    
    try
        exportgraphics(figAll, combinedPng, 'Resolution', 300);
    catch
        saveas(figAll, combinedPng);
    end
    fprintf('\n六合一汇总图已保存（无标题）: %s\n', combinedPng);
end

function plot_four_metrics_on_axes(ax, pd, analysis_var, showLegend)
    % pd.metrics_mean / metrics_std: npt x 4，列顺序 ACC, NMI, ARI, F-score
    npt = pd.npt;
    metricNames = {'ACC', 'NMI', 'ARI', 'F-score'};
    colors = [ ...
        0.0000, 0.4470, 0.7410;  % 蓝 ACC
        0.8500, 0.3250, 0.0980;  % 橙 NMI
        0.4660, 0.6740, 0.1880;  % 绿 ARI
        0.4940, 0.1840, 0.5560]; % 紫 F
    x = 1:npt;
    hold(ax, 'on');
    for m = 1:4
        y = pd.metrics_mean(:, m);
        plot(ax, x, y, '-o', 'Color', colors(m, :), 'MarkerFaceColor', colors(m, :), ...
            'LineWidth', 1.35, 'MarkerSize', 5, 'DisplayName', metricNames{m});
    end
    hold(ax, 'off');
    grid(ax, 'on');
    ylabel(ax, '指标 (%)');
    
    % 修改 xlabel 为 LaTeX 格式
    switch lower(strtrim(analysis_var))
        case 'alpha'
            xlabel(ax, '$\alpha$', 'Interpreter', 'latex');
        case 'eta'
            xlabel(ax, '$\eta$', 'Interpreter', 'latex');
        case 'sel_kernel'
            xlabel(ax, 'M', 'Interpreter', 'none');
        otherwise
            xlabel(ax, analysis_var, 'Interpreter', 'none');
    end
    
    ylim(ax, 'auto');
    if showLegend
        legend(ax, 'Location', 'best', 'FontSize', 7, 'Box', 'off');
    end
end

function plot_one_dataset_tile(pd, dsName, analysis_var)
    if pd.ok && pd.npt > 0
        npt = pd.npt;
        plot_four_metrics_on_axes(gca, pd, analysis_var, true);
        xtl = cell(npt, 1);
        for ii = 1:npt
            xtl{ii} = sprintf('%.4g', pd.value_list_ds(ii));
        end
        set(gca, 'XTick', 1:npt, 'XTickLabel', xtl);
        set(gca, 'TickLabelInterpreter', 'none'); % <-- 添加：使数字刻度标签直立
        % if exist('xtickangle', 'file') == 2
        %     xtickangle();
        % end
    else
        axis([0, 1, 0, 1]);
        axis off;
        text(0.5, 0.5, sprintf('%s\n(跳过)', dsName), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 10);
    end
    title(dsName, 'Interpreter', 'none');
end

function opts = build_opts_lx()
    opts = struct();
    opts.kNN_graph = 15;
    opts.warmup_T = 0;
    opts.static_term_on = true;
    opts.maxIter = 50;
    % main_lx.m 还会读取这些默认字段（若不提供也会在函数内部补齐）
    opts.irls_eps = 1e-8;
    opts.tol = 1e-6;
    opts.maxBetaIter = 10;
    opts.betaTol = 1e-5;
end

function opts = opts_lx_eval_from_campare(fixed_full)
    % 对应 campare_end 中 opts_lx_eval 的构造（当前 main_lx 仅使用 static_term_on）
    opts = build_opts_lx();
    opts.static_term_on = logical(fixed_full.static_on);
end

function p = get_fixed_mr_params(dataset_name, config_name)
    switch dataset_name
        case 'MSRC_v1'
            switch config_name
                case 'full'
                    p = struct('alpha', 0.90, 'sel_k', 15, 'eta', 10, 'gamma', 0, ...
                        'static_on', 1, 'dual_on', 1, 'dual_entropy_lambda', 0.0);
                otherwise
                    error('本脚本仅使用 config full');
            end
        case 'BBCSport'
            switch config_name
                case 'full'
                    p = struct('alpha', 0.10, 'sel_k', 8, 'eta', 100, 'gamma', 0, ...
                        'static_on', 1, 'dual_on', 1, 'dual_entropy_lambda', 0.0);
                otherwise
                    error('本脚本仅使用 config full');
            end
        case 'BRCA'
            switch config_name
                case 'full'
                    p = struct('alpha', 0.20, 'sel_k', 16, 'eta', 1, 'gamma', 0, ...
                        'static_on', 1, 'dual_on', 1, 'dual_entropy_lambda', 0.0);
                otherwise
                    error('本脚本仅使用 config full');
            end
        case 'WebKB'
            switch config_name
                case 'full'
                    p = struct('alpha', 0.30, 'sel_k', 6, 'eta', 1, 'gamma', 0, ...
                        'static_on', 1, 'dual_on', 1, 'dual_entropy_lambda', 0.0);
                otherwise
                    error('本脚本仅使用 config full');
            end
        case '3sources'
            switch config_name
                case 'full'
                    p = struct('alpha', 0.20, 'sel_k', 14, 'eta', 1, 'gamma', 0, ...
                        'static_on', 1, 'dual_on', 1, 'dual_entropy_lambda', 0.0);
                otherwise
                    error('本脚本仅使用 config full');
            end
        case 'Prokaryotic'
            switch config_name
                case 'full'
                    p = struct('alpha', 0.70, 'sel_k', 3, 'eta', 10, 'gamma', 0, ...
                        'static_on', 1, 'dual_on', 1, 'dual_entropy_lambda', 0.0);
                otherwise
                    error('本脚本仅使用 config full');
            end
        otherwise
            error('get_fixed_mr_params: 未配置数据集 %s（请与 campare_end.m 同步）', dataset_name);
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

function vgrid = default_value_grid(analysis_var, num_kernel)
    switch lower(strtrim(analysis_var))
        case 'alpha'
            vgrid = (0.1:0.1:0.9)';
        case 'eta'
            vgrid = [0.01; 0.1; 1; 10; 100];
        case {'sel_kernel', 'sel_k'}
            vgrid = unique(round(linspace(1, num_kernel, min(8, num_kernel))))';
        case 'knn_graph'
            vgrid = unique(round(linspace(5, 30, 6)))';
        case 'warmup_t'
            vgrid = (0:4)';
        case 'maxiter'
            vgrid = [20; 30; 50; 80];
        otherwise
            error('run_single_variable_analysis_main_lx:NoDefaultGrid', ...
                ['未为变量 "%s" 配置默认网格，请设置非空的 value_list。\n', ...
                '默认可用: alpha, eta, sel_kernel；', ...
                '或 knn_graph, warmup_T, maxIter'], ...
                analysis_var);
    end
end

function [alpha, eta, sk, opts] = apply_single_override( ...
    name, val, alpha, eta, sk, opts, num_kernel)

    nm = lower(strtrim(name));
    switch nm
        case 'alpha'
            alpha = double(val);
        case 'eta'
            eta = double(val);
        case {'sel_kernel', 'sel_k'}
            sk = max(1, min(round(double(val)), num_kernel));
        otherwise
            if ismember(nm, {'static_term_on', 'dual_dynamic_on', 'adaptive_view_graph', ...
                    'adaptive_sparse', 'adaptive_norm_rows', 'adaptive_normalized_l'})
                opts.(name) = logical(val ~= 0);
                return;
            end
            if ismember(nm, {'knn_graph', 'warmup_t', 'maxiter', 'adaptive_k', 'adaptive_every'})
                opts.(name) = max(1, round(double(val)));
                return;
            end
            opts.(name) = val;
    end
end