%% plot_convergence_curve_best_params.m

close all; clear; clc; warning off;

%% ===================== 1. 环境与路径 =====================
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir), baseDir = pwd; end
addpath(baseDir);
addpath(fullfile(baseDir, 'ClusteringMeasure'), fullfile(baseDir, 'FormulationKernels'), fullfile(baseDir, 'function'));

% get_MR_MPS_best_params 在仓库根目录
if isempty(which('get_MR_MPS_best_params'))
    repoRoot = fileparts(baseDir);
    if ~isempty(repoRoot)
        addpath(repoRoot);
    end
end

dataPath = fullfile(baseDir, 'Datasets', 'Multi_view-Datasets');
outDir   = fullfile(baseDir, 'MR_MPS', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% ===================== 2. 数据集与参数 =====================
datasetName = {'MSRC_v1','BBCSport','BRCA',...
    '3sources','WebKB','Prokaryotic'};
num_datasets = numel(datasetName);
obj_curves = cell(1, num_datasets);
param_used = cell(1, num_datasets);
ablDir = fullfile(baseDir, 'MR_MPS', 'ablation_2stage_results');

for idx = 1:num_datasets
    dname = datasetName{idx};
    dataFile = fullfile(dataPath, [dname '.mat']);
    if ~isfile(dataFile)
        fprintf('[跳过] 未找到数据文件: %s\n', dataFile);
        continue;
    end
    load(dataFile, 'fea', 'gt');
    gt = gt(:);
    num_cluster = length(unique(gt));

    % 读取 MR_ablation_2stage 的 full 最优参数
    p = [];
    ablCsv = fullfile(ablDir, sprintf('%s_ablation_2stage.csv', dname));
    if isfile(ablCsv)
        try
            T = readtable(ablCsv, 'TextType', 'string');
            row = T(T.config == "full", :);
            if ~isempty(row)
                p = struct();
                p.alpha_lx = row.alpha(1);
                p.sel_kernel_lx = row.sel_k(1);
                p.eta_lx = row.eta(1);
                if any(strcmp('dual_gamma', T.Properties.VariableNames))
                    p.gamma_lx = row.dual_gamma(1);
                elseif any(strcmp('gamma', T.Properties.VariableNames))
                    p.gamma_lx = row.gamma(1);
                else
                    p.gamma_lx = 0.5;
                end
            end
        catch
            p = [];
        end
    end
    if isempty(p)
        p = get_MR_MPS_best_params(dname);
    end
    if ~isfield(p, 'gamma_lx') || isempty(p.gamma_lx)
        p.gamma_lx = 0.5;
    end

    % KH/HP 缓存（与 campare_end / plot_convergence_curve 一致）
    cacheDir  = fullfile(baseDir, 'MR_MPS', 'KH_cache');
    if ~exist(cacheDir, 'dir'), mkdir(cacheDir); end
    dim_c_local = 5;
    cacheFile = fullfile(cacheDir, sprintf('%s_KH_HP_dim%d.mat', dname, dim_c_local));
    if isfile(cacheFile)
        S = load(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache');
        KH = S.KH; HP = S.HP; num_kernel = S.num_kernel;
        if isfield(S, 'num_cluster_cache'), num_cluster = S.num_cluster_cache; end
    else
        [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c_local);
        num_cluster_cache = num_cluster;
        dim_c = dim_c_local;
        save(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c', '-v7.3');
    end

    % opts：严格对齐 main_lx 当前参数命名
    opts = struct();
    opts.kNN_graph = 15;
    opts.warmup_T = 0;
    opts.static_term_on = true;
    opts.dual_dynamic_on = true;
    opts.dual_dynamic_gamma = p.gamma_lx;
    opts.dual_dynamic_sigma = 0;
    opts.maxIter = 80;

    fprintf('正在计算 %s 收敛曲线(full最优): alpha=%.3g, sel_k=%d, eta=%.3g, gamma=%.3g\n', ...
        dname, p.alpha_lx, p.sel_kernel_lx, p.eta_lx, p.gamma_lx);
    [~, obj] = main_lx(KH, HP, num_kernel, dim_c_local, num_cluster, ...
        p.alpha_lx, p.sel_kernel_lx, p.eta_lx, opts);
    obj_curves{idx} = obj(:);
    param_used{idx} = p;
end

%% ===================== 3. 绘图 =====================
fontNameCN = 'SimSun';
fontNameEN = 'Times New Roman';
fontSizeAxis = 10;
fontSizeTitle = 10;
lineWidthPlot = 1.5;
colors = lines(num_datasets);

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 8.5, 5.5]);

for idx = 1:num_datasets
    if isempty(obj_curves{idx}), continue; end
    subplot(2, 3, idx);

    obj_all = obj_curves{idx};
    nIter = numel(obj_all);
    targetIter = min(30, nIter);
    x = 1:targetIter;
    obj_plot = obj_all(1:targetIter);

    plot(x, obj_plot, '-o', ...
        'Color', colors(idx,:), ...
        'LineWidth', lineWidthPlot, ...
        'MarkerSize', 4, ...
        'MarkerFaceColor', colors(idx,:));
    grid on; box on;
    set(gca, 'FontName', fontNameEN, 'FontSize', fontSizeAxis, 'LineWidth', 0.8);
    if targetIter <= 10
        set(gca, 'XTick', 1:targetIter);
    else
        set(gca, 'XTick', 1:max(1, floor(targetIter/10)):targetIter);
    end
    xlabel('迭代次数 (Iterations)', 'FontName', fontNameCN, 'FontSize', fontSizeAxis);
    ylabel('目标函数值', 'FontName', fontNameCN, 'FontSize', fontSizeAxis);

    p = param_used{idx};
    if isempty(p)
        ttl = datasetName{idx};
    else
        ttl = sprintf('%s (a=%.2g,k=%d,\\eta=%.2g,\\gamma=%.2g)', ...
            datasetName{idx}, p.alpha_lx, p.sel_kernel_lx, p.eta_lx, p.gamma_lx);
    end
    title(ttl, 'FontName', fontNameEN, 'FontSize', fontSizeTitle, 'FontWeight', 'bold', 'Interpreter', 'tex');

    xlim([1 targetIter]);
    margin = 0.1 * (max(obj_plot) - min(obj_plot) + eps);
    ylim([min(obj_plot)-margin, max(obj_plot)+margin]);
end

%% ===================== 4. 保存 =====================
figPath = fullfile(outDir, 'Convergence_best_params');
exportgraphics(fig, [figPath '.pdf'], 'ContentType', 'vector');
exportgraphics(fig, [figPath '.png'], 'Resolution', 300);
fprintf('--> 绘图完成！已保存至: %s\n', outDir);

