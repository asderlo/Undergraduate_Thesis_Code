% campare_end.m
% 对比 MKKM、MKKM-SR、SimpleMKKM、LSWMKC、MPS、MPS-LX（主模型为 MPS-main/main_lx.m）
% 输出 ACC / NMI / ARI / F-score（百分比形式）及其标准差

% MPS-LX：二阶段网格搜索 alpha、sel_k、eta（与当前 main_lx 一致；main_lx 未使用 dual 动态图，cfg 中 dual_on 固定为 false）
% 消融：full（完整）、no_manifold（无流形正则 η·L）、no_multiscale（单尺度 dim_c=1）、no_kernel_sel（全部核）
% 若设置 use_fixed_mr_params=true 则改读本文件内 get_fixed_mr_params（快速调试）。

% 开关：run_all_datasets=true 时依次跑全部数据集；
%       false 时用 index 手动选一个
%       run_ablation=false 时跳过消融段及对应结果打印

clear
clc
warning off;

%% 路径
path_sm = fileparts(mfilename('fullpath'));
if isempty(path_sm), path_sm = pwd; end

% 仅添加当前实验目录（避免把上层副本/历史版本提前到路径前面）
addpath(path_sm);
addpath(genpath(path_sm));

% 统一定位 MPS-main：优先当前目录下的版本，避免调用到其它副本
mps_candidates = { ...
    fullfile(path_sm, 'MPS-main'), ...
    fullfile(path_sm, '..', 'MPS-main'), ...
    'd:\matlab\end\MPS-main' ...
    };
mpsDir = '';
for ci = 1:numel(mps_candidates)
    if isfolder(mps_candidates{ci})
        mpsDir = mps_candidates{ci};
        break;
    end
end
if isempty(mpsDir)
    error('未找到 MPS-main 目录，请检查工程路径。');
end

% 将核心目录置于路径最前，确保 main_lx/main/preprocess 调用一致
addpath(mpsDir, '-begin');
addpath(fullfile(mpsDir, 'FormulationKernels'), '-begin');
addpath(fullfile(mpsDir, 'function'), '-begin');
addpath(fullfile(mpsDir, 'ClusteringMeasure'), '-begin');
addpath(fullfile(mpsDir, 'new'), '-begin');

mkkm_srDir = fullfile(path_sm, '..', 'MKKM-SR-master');
if ~isfolder(mkkm_srDir)
    mkkm_srDir = 'd:\matlab\end\MKKM-SR-master';
end
if isfolder(mkkm_srDir)
    run_mkkm_sr = true;
else
    run_mkkm_sr = false;
    warning('未找到 MKKM-SR 目录，跳过 MKKM-SR。');
end

lswmkcDir = fullfile(path_sm, '..', 'LSWMKC-main');
if ~isfolder(lswmkcDir)
    lswmkcDir = 'd:\matlab\end\LSWMKC-main';
end
if isfolder(lswmkcDir)
    run_lswmkc = true;
else
    run_lswmkc = false;
    warning('未找到 LSWMKC 目录，跳过 LSWMKC。');
end

%% 数据与可调参数（与 MPS 一致）
dataPath = fullfile(mpsDir, 'Datasets', 'Multi_view-Datasets');
datasetName = {'MSRC_v1','BBCSport','BRCA',...
    '3sources','WebKB','Prokaryotic'};

% 开关：全部数据集
run_all_datasets = true;
index = 3;  

% 开关：消融实验
run_ablation = true;

if run_all_datasets
    data_indices = 1:numel(datasetName);
else
    if index < 1 || index > numel(datasetName)
        error('index 须在 1~%d 之间', numel(datasetName));
    end
    data_indices = index;
end

dim_c = 5;
neibour_lsw = 5;
sm_maxiter = 500;
sm_threshold = 1e-4;
numTrials = 10;
repeat_mr_eval = 10;
repeat_times_search = 3;
alpha_list_mr = 0.1:0.1:0.9;
sel_ratio_list_mr = 0.1:0.1:0.9;
eta_list_mr = [0.01, 0.1, 1, 10, 100];
gamma_list_mr = 0.1:0.1:1;
eta_stage1_mr = 1;

%% ===================== MPS-LX（main_lx.m）opts 基准 =====================
% 二阶段搜索里传给 main_lx 的基准 opts；static 由 opts_for_mr_config 按 cfg 覆盖。
% 当前仓库 main_lx.m 仅使用 static_term_on / kNN 等字段，dual_* 不参与迭代，dual_on 搜索关闭。
opts_lx = struct();
opts_lx.kNN_graph = 15;
opts_lx.warmup_T = 0;
opts_lx.static_term_on = true;
opts_lx.dual_dynamic_on = false;
opts_lx.dual_dynamic_gamma = 0;
opts_lx.dual_dynamic_sigma = 0;
opts_lx.dual_entropy_lambda = 0.0;
opts_lx.dual_entropy_eps = 1e-12;
opts_lx.maxIter = 50;

lxRoot = fullfile(mpsDir, 'MPS_LX');
compareOutDir = fullfile(lxRoot, 'compare_end_results');
if ~exist(compareOutDir, 'dir'), mkdir(compareOutDir); end

for ds_k = 1:numel(data_indices)
    idx = data_indices(ds_k);
    dataname = datasetName(idx);
    dataFile = fullfile(dataPath, [dataname{1}, '.mat']);
    if numel(data_indices) > 1
        fprintf('\n========== [%d/%d] 数据集 %s ==========\n', ds_k, numel(data_indices), dataname{1});
    end

    if ~isfile(dataFile)
        warning('跳过：未找到 %s', dataFile);
        continue;
    end
    load(dataFile, 'fea', 'gt');
    num_cluster = length(unique(gt));

%% MPS 最优参数来自 get_MR_MPS_best_params；
% MPS-LX 使用两阶段搜索（Stage1: eta=1 搜 alpha+sel_k；Stage2: 搜 eta）
p_best = get_MR_MPS_best_params(dataname{1});
alpha_mps = p_best.alpha_mps;       
sel_kernel_mps = p_best.sel_kernel_mps;

%% KH：优先从缓存读取核集合；若无缓存，则 preprocess 一次并写入缓存（MPS_LX 目录，可与旧 MR_MPS/KH_cache 手动互拷 .mat）
kernelCacheDir = fullfile(lxRoot, 'KH_cache');
if ~exist(kernelCacheDir, 'dir'), mkdir(kernelCacheDir); end
cacheFile = fullfile(kernelCacheDir, sprintf('%s_KH_HP_dim%d.mat', dataname{1}, dim_c));

if isfile(cacheFile)
    fprintf('从缓存读取 KH / HP: %s\n', cacheFile);
    S = load(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c');
    KH = S.KH;
    HP = S.HP;
    num_kernel = S.num_kernel;
    if isfield(S, 'num_cluster_cache')
        num_cluster = S.num_cluster_cache;
    end
    t_prep = NaN;
else
    fprintf('用 MPS preprocess 生成 KH（与 MPS 构造方式完全一致）...\n');
    tic;
    [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c);
    t_prep = toc;
    num_cluster_cache = num_cluster; 
    save(cacheFile, 'KH', 'HP', 'num_kernel', 'num_cluster_cache', 'dim_c', '-v7.3');
    fprintf('已将 KH / HP 缓存到: %s\n', cacheFile);
end

n = size(KH, 1);
Y = gt(:);
u = unique(Y);
u(u < 1) = [];
for i = 1:length(u), Y(Y == u(i)) = i; end
numclass = num_cluster;
fprintf('  n=%d, numclass=%d, numker=%d, 预处理时间=%.2fs（NaN 表示来自缓存）\n', n, numclass, num_kernel, t_prep);

%% ---------- MPS-LX（main_lx）参数：二阶段搜索 ----------
% use_fixed_mr_params=true 时跳过网格、直接读 get_fixed_mr_params（仅调试用）
use_fixed_mr_params = false;
addpath(fullfile(mpsDir, 'MR_MPS')); % save_best_params_3stage.m 所在目录
sel_list_mr = ceil(sel_ratio_list_mr * num_kernel);
sel_list_mr = max(1, min(sel_list_mr, num_kernel));

cfg_full = struct('name','full', 'dim_c',dim_c, 'force_all_kernels',false, ...
    'static_on',true, 'dual_on',false, 'dual_entropy_lambda',opts_lx.dual_entropy_lambda);

if use_fixed_mr_params
    fixed_full = get_fixed_mr_params(dataname{1}, 'full');
    alpha_lx = fixed_full.alpha;
    sel_kernel_lx = fixed_full.sel_k;
    eta_lx = fixed_full.eta;
    gamma_lx = fixed_full.gamma;
    best_lx = struct('alpha',alpha_lx, 'sel_k',sel_kernel_lx, 'eta',eta_lx, ...
        'gamma',gamma_lx, 'ACC',NaN, 'stage1_ACC', NaN);
    cfg_eval_main = struct('static_on', fixed_full.static_on, 'dual_on', fixed_full.dual_on, ...
        'dual_entropy_lambda', fixed_full.dual_entropy_lambda);
else
    fprintf('MPS-LX 参数：二阶段搜索（main_lx，Stage1 eta=%g；Stage2 eta，repeat=%d）...\n', ...
        eta_stage1_mr, repeat_times_search);
    best_lx = search_best_mr_config(KH, HP, num_kernel, dim_c, num_cluster, gt, ...
        alpha_list_mr, sel_list_mr, eta_stage1_mr, eta_list_mr, gamma_list_mr, opts_lx, cfg_full, repeat_times_search);
    alpha_lx = best_lx.alpha;
    sel_kernel_lx = best_lx.sel_k;
    eta_lx = best_lx.eta;
    gamma_lx = best_lx.gamma;
    fprintf('  2-stage 最优: alpha=%.2f, sel_k=%d, eta=%.3g, gamma=%.1f, search_ACC=%.4f (stage1_ACC=%.4f)\n', ...
        alpha_lx, sel_kernel_lx, eta_lx, gamma_lx, best_lx.ACC, best_lx.stage1_ACC);
    try
        b1s = struct('alpha', alpha_lx, 'sel_k', sel_kernel_lx, 'ACC', NaN);
        b3s = struct('eta', eta_lx, 'gamma', gamma_lx, 'ACC', best_lx.ACC);
        save_best_params_3stage(dataname{1}, b1s, struct('eta', eta_lx), b3s);
    catch
        warning('写入 best_params_3stage 失败');
    end
    cfg_eval_main = struct('static_on', cfg_full.static_on, 'dual_on', cfg_full.dual_on, ...
        'dual_entropy_lambda', cfg_full.dual_entropy_lambda);
end

%% ---------- 1) MKKM ----------
fprintf('跑 MKKM...\n');
tic;
[H_mkkm, Sigma0, ~] = mkkmeans_train(KH, numclass);
t_mkkm = toc;
runs_mkkm = zeros(numTrials, 4);
for t = 1:numTrials
    rng(t);
    res = my_nmi_acc(H_mkkm, Y, numclass);
    runs_mkkm(t, :) = [res(1, 7), res(1, 4), res(1, 5), res(1, 1)]; 
end
acc_mkkm = runs_mkkm(:, 1); nmi_mkkm = runs_mkkm(:, 2);
ari_mkkm = runs_mkkm(:, 3); fscore_mkkm = runs_mkkm(:, 4);

%% ---------- 2) SimpleMKKM ----------
fprintf('跑 SimpleMKKM...\n');
options.seuildiffsigma = 1e-5; options.goldensearch_deltmax = 1e-3;
options.goldensearchmax = 1e-8; options.numericalprecision = 1e-16;
options.firstbasevariable = 'first'; options.nbitermax = sm_maxiter;
options.seuil = 0; options.seuilitermax = 10;
options.miniter = 0; options.threshold = sm_threshold; options.Sigma0 = Sigma0;
tic;
[H_sm, ~, ~] = simpleMKKM(KH, numclass, options);
t_sm = toc;
runs_sm = zeros(numTrials, 4);
for t = 1:numTrials
    rng(t);
    res = my_nmi_acc(H_sm, Y, numclass);
    runs_sm(t, :) = [res(1, 7), res(1, 4), res(1, 5), res(1, 1)];
end
acc_sm = runs_sm(:, 1); nmi_sm = runs_sm(:, 2);
ari_sm = runs_sm(:, 3); fscore_sm = runs_sm(:, 4);

%% ---------- 3) MKKM-SR ----------
if run_mkkm_sr
    fprintf('跑 MKKM-SR...\n');
    lambda_sr_list = 2.^(-10:2:12);
    acc_sr_all = zeros(numTrials, numel(lambda_sr_list));
    nmi_sr_all = zeros(numTrials, numel(lambda_sr_list));
    ari_sr_all = zeros(numTrials, numel(lambda_sr_list));
    fscore_sr_all = zeros(numTrials, numel(lambda_sr_list)); 
    t_sr_total = 0;
    for a = 1:numel(lambda_sr_list)
        lam = lambda_sr_list(a);
        for t = 1:numTrials
            rng(t);
            Y_init = Y_Initialize(n, numclass);
            tic;
            [y_pred, ~, ~] = MKKM_SR(KH, Y_init, lam);
            t_sr_total = t_sr_total + toc;
            m = Clustering8Measure(Y, y_pred(:));
            fscore_sr_all(t, a) = m(1); nmi_sr_all(t, a) = m(4);
            ari_sr_all(t, a) = m(5);  acc_sr_all(t, a)  = m(7);
        end
    end
    [~, idx_best_sr] = max(mean(acc_sr_all, 1));
    acc_sr = acc_sr_all(:, idx_best_sr); nmi_sr = nmi_sr_all(:, idx_best_sr);
    ari_sr = ari_sr_all(:, idx_best_sr); fscore_sr = fscore_sr_all(:, idx_best_sr);
    t_sr = t_sr_total / numel(lambda_sr_list);
else
    acc_sr = []; nmi_sr = []; ari_sr = []; fscore_sr = []; t_sr = nan;
end

%% ---------- 4) LSWMKC ----------
if run_lswmkc
    fprintf('跑 LSWMKC...\n');
    addpath(genpath(lswmkcDir), '-begin'); 
    global neibour; neibour = neibour_lsw;
    alpha_lsw = 2.^(0:1:10);
    H_lsw_cell = cell(1, numel(alpha_lsw));
    t_lsw = 0;
    for a = 1:numel(alpha_lsw)
        tic;
        [Kstar, ~, ~, ~, ~] = Graph_main(KH, alpha_lsw(a));
        t_lsw = t_lsw + toc;
        Kstar = kcenter(Kstar); Kstar = knorm(Kstar);
        opt.disp = 0; [H_lsw, ~] = eigs(Kstar, numclass, 'la', opt);
        H_lsw_cell{a} = H_lsw;
    end
    acc_lsw = zeros(numTrials, numel(alpha_lsw)); nmi_lsw = zeros(numTrials, numel(alpha_lsw));
    ari_lsw = zeros(numTrials, numel(alpha_lsw)); fscore_lsw = zeros(numTrials, numel(alpha_lsw));
    for a = 1:numel(alpha_lsw)
        for t = 1:numTrials
            rng(t);
            res = my_nmi_acc(H_lsw_cell{a}, Y, numclass);
            acc_lsw(t, a) = res(1, 7); nmi_lsw(t, a) = res(1, 4);
            ari_lsw(t, a) = res(1, 5); fscore_lsw(t, a) = res(1, 1);
        end
    end
    [~, idx_best] = max(mean(acc_lsw, 1));
    acc_lsw = acc_lsw(:, idx_best); nmi_lsw = nmi_lsw(:, idx_best);
    ari_lsw = ari_lsw(:, idx_best); fscore_lsw = fscore_lsw(:, idx_best);
    t_lsw = t_lsw / numel(alpha_lsw);
    rmpath(genpath(lswmkcDir)); 
else
    acc_lsw = []; nmi_lsw = []; ari_lsw = []; fscore_lsw = []; t_lsw = nan;
end

%% ---------- 5) MPS (直接使用最优参数) ----------
fprintf('跑 MPS（直接使用最优参数 a=%.2f, k=%d，rep=10）...\n', alpha_mps, sel_kernel_mps);
addpath(genpath(mpsDir), '-begin'); 
rng('default');
repeat_mps = 10;                              
runs_mps = zeros(repeat_mps, 4);      
t_mps = 0;
for rep = 1:repeat_mps
    tic;
    [F_mps, ~] = main(KH, HP, num_kernel, dim_c, num_cluster, alpha_mps, sel_kernel_mps);
    t_mps = t_mps + toc;
    res = my_nmi_acc(F_mps, gt, num_cluster);
    b = res(1, :);
    runs_mps(rep, :) = [b(7), b(4), b(5), b(1)]; 
end
mps_mean = mean(runs_mps, 1);
mps_std  = std(runs_mps, 0, 1);

acc_mps = mps_mean(1); nmi_mps = mps_mean(2); ari_mps = mps_mean(3); fscore_mps = mps_mean(4);
std_mps_acc = mps_std(1); std_mps_nmi = mps_std(2); std_mps_ari = mps_std(3); std_mps_fscore = mps_std(4);

%% ---------- 6) MPS-LX / main_lx（two-stage 最优参数）----------
fprintf('跑 MPS-LX / main_lx（a=%.2f, k=%d, eta=%g, gamma=%.1f，rep=%d）...\n', ...
    alpha_lx, sel_kernel_lx, eta_lx, gamma_lx, repeat_mr_eval);
rng('default');
repeat_lx = repeat_mr_eval;
runs_lx = zeros(repeat_lx, 4);
t_lx = 0;
opts_lx_eval = opts_lx;
opts_lx_eval.static_term_on = cfg_eval_main.static_on;
opts_lx_eval.dual_dynamic_on = cfg_eval_main.dual_on;
opts_lx_eval.dual_entropy_lambda = cfg_eval_main.dual_entropy_lambda;
opts_lx_eval.dual_dynamic_gamma = gamma_lx;
for rep = 1:repeat_lx
    rng(rep);
    tic;
    [F_lx, ~] = main_lx(KH, HP, num_kernel, dim_c, num_cluster, alpha_lx, sel_kernel_lx, eta_lx, opts_lx_eval);
    t_lx = t_lx + toc;
    res = my_nmi_acc(F_lx, gt, num_cluster);
    b = res(1, :);
    runs_lx(rep, :) = [b(7), b(4), b(5), b(1)]; 
end
lx_mean = mean(runs_lx, 1);
lx_std  = std(runs_lx, 0, 1);

acc_lx = lx_mean(1); nmi_lx = lx_mean(2); ari_lx = lx_mean(3); fscore_lx = lx_mean(4);
std_lx_acc = lx_std(1); std_lx_nmi = lx_std(2); std_lx_ari = lx_std(3); std_lx_fscore = lx_std(4);

if run_ablation
%% ---------- 7) MPS-LX 消融：各配置独立二阶段搜索（full 复用主实验 best_lx）----------
ablation_configs = {};
ablation_configs{end+1} = struct('name','full',           'dim_c',dim_c, 'force_all_kernels',false, 'static_on',true,  'dual_on',false, 'dual_entropy_lambda',0);
ablation_configs{end+1} = struct('name','no_manifold',    'dim_c',dim_c, 'force_all_kernels',false, 'static_on',false, 'dual_on',false, 'dual_entropy_lambda',0);
ablation_configs{end+1} = struct('name','no_multiscale',  'dim_c',1,     'force_all_kernels',false, 'static_on',true,  'dual_on',false, 'dual_entropy_lambda',0);
ablation_configs{end+1} = struct('name','no_kernel_sel',  'dim_c',dim_c, 'force_all_kernels',true,  'static_on',true,  'dual_on',false, 'dual_entropy_lambda',0);

ablation_results = struct();
for aci = 1:numel(ablation_configs)
    cfg_ab = ablation_configs{aci};
    fprintf('MPS-LX 消融 %s：二阶段搜索...\n', cfg_ab.name);
    if cfg_ab.dim_c == dim_c
        HP_ab = HP;
    else
        HP_ab = keep_first_scale_HP(HP);
    end
    if cfg_ab.force_all_kernels
        sel_candidates_ab = num_kernel;
    else
        sel_candidates_ab = sel_list_mr;
    end

    if use_fixed_mr_params
        fixed_ab = get_fixed_mr_params(dataname{1}, cfg_ab.name);
        sk = fixed_ab.sel_k;
        if strcmp(cfg_ab.name, 'no_kernel_sel') && (isnan(sk) || isempty(sk))
            sk = num_kernel;
        end
        best_ab = struct('alpha', fixed_ab.alpha, 'sel_k', sk, ...
            'eta', fixed_ab.eta, 'gamma', fixed_ab.gamma, 'ACC', NaN, 'stage1_ACC', NaN);
    elseif strcmp(cfg_ab.name, 'full')
        best_ab = best_lx;
        fprintf('  full 复用主实验最优: alpha=%.2f, sel_k=%d, eta=%.3g, gamma=%.1f, search_ACC=%.4f\n', ...
            best_ab.alpha, best_ab.sel_k, best_ab.eta, best_ab.gamma, best_ab.ACC);
    else
        best_ab = search_best_mr_config(KH, HP_ab, num_kernel, cfg_ab.dim_c, num_cluster, gt, ...
            alpha_list_mr, sel_candidates_ab, eta_stage1_mr, eta_list_mr, gamma_list_mr, opts_lx, cfg_ab, repeat_times_search);
        fprintf('  %s 搜索最优: alpha=%.2f, sel_k=%d, eta=%.3g, gamma=%.1f, search_ACC=%.4f\n', ...
            cfg_ab.name, best_ab.alpha, best_ab.sel_k, best_ab.eta, best_ab.gamma, best_ab.ACC);
    end
    opts_ab = opts_for_mr_config(opts_lx, cfg_ab, best_ab.gamma);
    [metrics_ab, std_ab, t_ab] = eval_mr_mps_runs(KH, HP_ab, num_kernel, cfg_ab.dim_c, num_cluster, ...
        best_ab.alpha, best_ab.sel_k, best_ab.eta, opts_ab, repeat_lx, gt);

    ablation_results.(cfg_ab.name).cfg = cfg_ab;
    ablation_results.(cfg_ab.name).best = best_ab;
    ablation_results.(cfg_ab.name).mean = metrics_ab;
    ablation_results.(cfg_ab.name).std = std_ab;
    ablation_results.(cfg_ab.name).time = t_ab;
end
end  % run_ablation

%% ---------- 结果（百分比形式，含 ACC / NMI / ARI / F-score）----------
dataname = char(dataname);
fprintf('数据集 %s\n', dataname);
fprintf('===== MKKM/MKKM-SR/SimpleMKKM/LSWMKC/MPS/MPS-LX(main_lx)（统一使用my_nmi_acc，%d trials mean±std，百分比）=====\n', numTrials);

fprintf('MKKM:       ACC=%.2f±%.2f%%,  NMI=%.2f±%.2f%%,  ARI=%.2f±%.2f%%,  F-score=%.2f±%.2f%%,  时间=%.2fs\n', ...
    mean(acc_mkkm)*100, std(acc_mkkm)*100, mean(nmi_mkkm)*100, std(nmi_mkkm)*100, mean(ari_mkkm)*100, std(ari_mkkm)*100, mean(fscore_mkkm)*100, std(fscore_mkkm)*100, t_mkkm);

if run_mkkm_sr
    fprintf('MKKM-SR:    ACC=%.2f±%.2f%%,  NMI=%.2f±%.2f%%,  ARI=%.2f±%.2f%%,  F-score=%.2f±%.2f%%,  时间=%.2fs\n', ...
        mean(acc_sr)*100, std(acc_sr)*100, mean(nmi_sr)*100, std(nmi_sr)*100, mean(ari_sr)*100, std(ari_sr)*100, mean(fscore_sr)*100, std(fscore_sr)*100, t_sr);
end

fprintf('SimpleMKKM: ACC=%.2f±%.2f%%,  NMI=%.2f±%.2f%%,  ARI=%.2f±%.2f%%,  F-score=%.2f±%.2f%%,  时间=%.2fs\n', ...
    mean(acc_sm)*100, std(acc_sm)*100, mean(nmi_sm)*100, std(nmi_sm)*100, mean(ari_sm)*100, std(ari_sm)*100, mean(fscore_sm)*100, std(fscore_sm)*100, t_sm);

if run_lswmkc
    fprintf('LSWMKC:     ACC=%.2f±%.2f%%,  NMI=%.2f±%.2f%%,  ARI=%.2f±%.2f%%,  F-score=%.2f±%.2f%%,  时间=%.2fs\n', ...
        mean(acc_lsw)*100, std(acc_lsw)*100, mean(nmi_lsw)*100, std(nmi_lsw)*100, mean(ari_lsw)*100, std(ari_lsw)*100, mean(fscore_lsw)*100, std(fscore_lsw)*100, t_lsw);
end

fprintf('MPS (a=%.2f, k=%d): ACC=%.2f±%.2f%%,  NMI=%.2f±%.2f%%,  ARI=%.2f±%.2f%%,  F-score=%.2f±%.2f%%,  时间=%.2fs\n', ...
    alpha_mps, sel_kernel_mps, acc_mps*100, std_mps_acc*100, nmi_mps*100, std_mps_nmi*100, ari_mps*100, std_mps_ari*100, fscore_mps*100, std_mps_fscore*100, t_mps);

fprintf('MPS-LX (a=%.2f, k=%d, eta=%.2g, gamma=%.1f): ACC=%.2f±%.2f%%,  NMI=%.2f±%.2f%%,  ARI=%.2f±%.2f%%,  F-score=%.2f±%.2f%%,  时间=%.2fs\n', ...
    alpha_lx, sel_kernel_lx, eta_lx, gamma_lx, acc_lx*100, std_lx_acc*100, nmi_lx*100, std_lx_nmi*100, ari_lx*100, std_lx_ari*100, fscore_lx*100, std_lx_fscore*100, t_lx);

if run_ablation
    ab_names = fieldnames(ablation_results);
    for ai = 1:numel(ab_names)
        nm = ab_names{ai};
        rr = ablation_results.(nm);
        fprintf('MPS-LX %s (a=%.2f, k=%d, eta=%.2g, gamma=%.1f): ACC=%.2f±%.2f%%,  NMI=%.2f±%.2f%%,  ARI=%.2f±%.2f%%,  F-score=%.2f±%.2f%%,  时间=%.2fs\n', ...
            nm, rr.best.alpha, rr.best.sel_k, rr.best.eta, rr.best.gamma, ...
            rr.mean(1)*100, rr.std(1)*100, rr.mean(2)*100, rr.std(2)*100, ...
            rr.mean(3)*100, rr.std(3)*100, rr.mean(4)*100, rr.std(4)*100, rr.time);
    end
end

compare_summary = struct();
compare_summary.MKKM = pack_method_result([mean(acc_mkkm), mean(nmi_mkkm), mean(ari_mkkm), mean(fscore_mkkm)], ...
    [std(acc_mkkm), std(nmi_mkkm), std(ari_mkkm), std(fscore_mkkm)], t_mkkm);
compare_summary.SimpleMKKM = pack_method_result([mean(acc_sm), mean(nmi_sm), mean(ari_sm), mean(fscore_sm)], ...
    [std(acc_sm), std(nmi_sm), std(ari_sm), std(fscore_sm)], t_sm);
compare_summary.MPS = pack_method_result(mps_mean, mps_std, t_mps);
compare_summary.MPS_LX = pack_method_result(lx_mean, lx_std, t_lx);
if run_mkkm_sr
    compare_summary.MKKM_SR = pack_method_result([mean(acc_sr), mean(nmi_sr), mean(ari_sr), mean(fscore_sr)], ...
        [std(acc_sr), std(nmi_sr), std(ari_sr), std(fscore_sr)], t_sr);
end
if run_lswmkc
    compare_summary.LSWMKC = pack_method_result([mean(acc_lsw), mean(nmi_lsw), mean(ari_lsw), mean(fscore_lsw)], ...
        [std(acc_lsw), std(nmi_lsw), std(ari_lsw), std(fscore_lsw)], t_lsw);
end
if ~run_ablation
    ablation_results = struct();
end
search_grid_mr = struct('alpha_list',alpha_list_mr, 'sel_ratio_list',sel_ratio_list_mr, ...
    'eta_list',eta_list_mr, 'gamma_list',gamma_list_mr);
outMat = fullfile(compareOutDir, sprintf('%s_compare_end.mat', dataname));
save(outMat, 'dataname', 'compare_summary', 'ablation_results', 'best_lx', ...
    'alpha_mps', 'sel_kernel_mps', 'search_grid_mr', 'repeat_times_search', ...
    'repeat_mr_eval', 'run_ablation', 'opts_lx', 'use_fixed_mr_params', '-v7.3');
fprintf('比较实验结果已保存: %s\n', outMat);

outCsv = fullfile(compareOutDir, sprintf('%s_compare_end.csv', dataname));
fid = fopen(outCsv, 'w');
fprintf(fid, 'category,name,alpha,sel_k,eta,gamma,ACC_mean,ACC_std,NMI_mean,NMI_std,ARI_mean,ARI_std,F_mean,F_std,time\n');
fprintf(fid, 'method,MKKM,NaN,NaN,NaN,NaN,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
    mean(acc_mkkm), std(acc_mkkm), mean(nmi_mkkm), std(nmi_mkkm), mean(ari_mkkm), std(ari_mkkm), mean(fscore_mkkm), std(fscore_mkkm), t_mkkm);
if run_mkkm_sr
    fprintf(fid, 'method,MKKM_SR,NaN,NaN,NaN,NaN,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
        mean(acc_sr), std(acc_sr), mean(nmi_sr), std(nmi_sr), mean(ari_sr), std(ari_sr), mean(fscore_sr), std(fscore_sr), t_sr);
end
fprintf(fid, 'method,SimpleMKKM,NaN,NaN,NaN,NaN,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
    mean(acc_sm), std(acc_sm), mean(nmi_sm), std(nmi_sm), mean(ari_sm), std(ari_sm), mean(fscore_sm), std(fscore_sm), t_sm);
if run_lswmkc
    fprintf(fid, 'method,LSWMKC,NaN,NaN,NaN,NaN,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
        mean(acc_lsw), std(acc_lsw), mean(nmi_lsw), std(nmi_lsw), mean(ari_lsw), std(ari_lsw), mean(fscore_lsw), std(fscore_lsw), t_lsw);
end
fprintf(fid, 'method,MPS,%.8g,%d,NaN,NaN,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
    alpha_mps, sel_kernel_mps, mps_mean(1), mps_std(1), mps_mean(2), mps_std(2), mps_mean(3), mps_std(3), mps_mean(4), mps_std(4), t_mps);
fprintf(fid, 'method,MPS_LX,%.8g,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
    alpha_lx, sel_kernel_lx, eta_lx, gamma_lx, lx_mean(1), lx_std(1), lx_mean(2), lx_std(2), lx_mean(3), lx_std(3), lx_mean(4), lx_std(4), t_lx);
if run_ablation
    ab_names = fieldnames(ablation_results);
    for ai = 1:numel(ab_names)
        nm = ab_names{ai};
        rr = ablation_results.(nm);
        fprintf(fid, 'ablation,%s,%.8g,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n', ...
            nm, rr.best.alpha, rr.best.sel_k, rr.best.eta, rr.best.gamma, ...
            rr.mean(1), rr.std(1), rr.mean(2), rr.std(2), rr.mean(3), rr.std(3), rr.mean(4), rr.std(4), rr.time);
    end
end
fclose(fid);
fprintf('比较实验 CSV 已保存: %s\n', outCsv);

end  % for ds_k：数据集循环

%% ===================== MPS-LX 参数搜索与评估辅助函数 =====================
function best = search_best_mr_config(KH, HP, num_kernel, dim_c, num_cluster, gt, ...
    alpha_list, sel_list, eta_stage1, eta_list, gamma_list, opts_base, cfg, repeat_times)

stage1_eta = eta_stage1 * double(cfg.static_on);
stage1_gamma = opts_base.dual_dynamic_gamma * double(cfg.dual_on);

best1.ACC = -inf;
best1.alpha = NaN;
best1.sel_k = NaN;

for ia = 1:numel(alpha_list)
    alpha = alpha_list(ia);
    for ik = 1:numel(sel_list)
        sel_k = sel_list(ik);
        opts = opts_for_mr_config(opts_base, cfg, stage1_gamma);
        acc_runs = zeros(repeat_times, 1);
        for r = 1:repeat_times
            rng(r);
            [F, ~] = main_lx(KH, HP, num_kernel, dim_c, num_cluster, alpha, sel_k, stage1_eta, opts);
            res = my_nmi_acc(real(F), gt, num_cluster);
            acc_runs(r) = res(1, 7);
        end
        acc = mean(acc_runs);
        if acc > best1.ACC
            best1.ACC = acc;
            best1.alpha = alpha;
            best1.sel_k = sel_k;
        end
    end
end

if cfg.static_on
    eta_candidates = eta_list;
else
    eta_candidates = 0;
end
if cfg.dual_on
    gamma_candidates = gamma_list;
else
    gamma_candidates = 0;
end

best.ACC = -inf;
best.alpha = best1.alpha;
best.sel_k = best1.sel_k;
best.eta = NaN;
best.gamma = NaN;
best.stage1_ACC = best1.ACC;

for ie = 1:numel(eta_candidates)
    eta = eta_candidates(ie);
    for ig = 1:numel(gamma_candidates)
        gamma = gamma_candidates(ig);
        opts = opts_for_mr_config(opts_base, cfg, gamma);
        acc_runs = zeros(repeat_times, 1);
        for r = 1:repeat_times
            rng(r);
            [F, ~] = main_lx(KH, HP, num_kernel, dim_c, num_cluster, best1.alpha, best1.sel_k, eta, opts);
            res = my_nmi_acc(real(F), gt, num_cluster);
            acc_runs(r) = res(1, 7);
        end
        acc = mean(acc_runs);
        if acc > best.ACC
            best.ACC = acc;
            best.eta = eta;
            best.gamma = gamma;
        end
    end
end
end

function opts = opts_for_mr_config(opts_base, cfg, gamma)
opts = opts_base;
opts.static_term_on = cfg.static_on;
opts.dual_dynamic_on = cfg.dual_on;
opts.dual_dynamic_gamma = gamma;
opts.dual_entropy_lambda = cfg.dual_entropy_lambda;
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

function [metrics_mean, metrics_std, t_mean] = ...
    eval_mr_mps_runs(KH, HP_use, num_kernel, dim_c_use, num_cluster, alpha, sel_k, eta, opts, repeat_times, gt)
rng('default');
runs = zeros(repeat_times, 4);
t_total = 0;
for rep = 1:repeat_times
    rng(rep);
    tic;
    [F_tmp, ~] = main_lx(KH, HP_use, num_kernel, dim_c_use, num_cluster, alpha, sel_k, eta, opts);
    t_total = t_total + toc;
    res = my_nmi_acc(F_tmp, gt, num_cluster);
    b = res(1,:);
    runs(rep,:) = [b(7), b(4), b(5), b(1)];
end
metrics_mean = mean(runs,1);
metrics_std = std(runs,0,1);
t_mean = t_total / repeat_times;
end

function method_result = pack_method_result(metrics_mean, metrics_std, runtime)
method_result = struct('mean', metrics_mean, 'std', metrics_std, 'time', runtime);
end

function p = get_fixed_mr_params(dataset_name, config_name)
% 与 use_fixed_mr_params 配套；main_lx 未用 dual，gamma 仅占位。
switch dataset_name
    case 'MSRC_v1'
        switch config_name
            case 'full',            p = struct('alpha',0.30,'sel_k',20,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_manifold',     p = struct('alpha',0.50,'sel_k',15,'eta',0,'gamma',0,'static_on',0,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_multiscale',   p = struct('alpha',0.30,'sel_k',20,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_kernel_sel',   p = struct('alpha',0.30,'sel_k',NaN,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            otherwise, error('未知配置: %s', config_name);
        end
    case 'BBCSport'
        switch config_name
            case 'full',            p = struct('alpha',0.90,'sel_k',4,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_manifold',     p = struct('alpha',0.90,'sel_k',1,'eta',0,'gamma',0,'static_on',0,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_multiscale',   p = struct('alpha',0.90,'sel_k',4,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_kernel_sel',   p = struct('alpha',0.90,'sel_k',NaN,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            otherwise, error('未知配置: %s', config_name);
        end
    case 'BRCA'
        switch config_name
            case 'full',            p = struct('alpha',0.40,'sel_k',8,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_manifold',     p = struct('alpha',0.10,'sel_k',16,'eta',0,'gamma',0,'static_on',0,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_multiscale',   p = struct('alpha',0.40,'sel_k',8,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_kernel_sel',   p = struct('alpha',0.40,'sel_k',NaN,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            otherwise, error('未知配置: %s', config_name);
        end
    case '3sources'
        switch config_name
            case 'full',            p = struct('alpha',0.70,'sel_k',5,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_manifold',     p = struct('alpha',0.90,'sel_k',14,'eta',0,'gamma',0,'static_on',0,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_multiscale',   p = struct('alpha',0.70,'sel_k',5,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_kernel_sel',   p = struct('alpha',0.70,'sel_k',NaN,'eta',1,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            otherwise, error('未知配置: %s', config_name);
        end
    case 'WebKB'
        switch config_name
            case 'full',            p = struct('alpha',0.50,'sel_k',12,'eta',10,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_manifold',     p = struct('alpha',0.40,'sel_k',14,'eta',0,'gamma',0,'static_on',0,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_multiscale',   p = struct('alpha',0.50,'sel_k',12,'eta',10,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_kernel_sel',   p = struct('alpha',0.50,'sel_k',NaN,'eta',10,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            otherwise, error('未知配置: %s', config_name);
        end
    case 'Prokaryotic'
        switch config_name
            case 'full',            p = struct('alpha',0.20,'sel_k',2,'eta',10,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_manifold',     p = struct('alpha',0.30,'sel_k',2,'eta',0,'gamma',0,'static_on',0,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_multiscale',   p = struct('alpha',0.20,'sel_k',2,'eta',10,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            case 'no_kernel_sel',   p = struct('alpha',0.20,'sel_k',NaN,'eta',10,'gamma',0,'static_on',1,'dual_on',0,'dual_entropy_lambda',0);
            otherwise, error('未知配置: %s', config_name);
        end
    otherwise
        error('get_fixed_mr_params: 未配置数据集 %s', dataset_name);
end
end

%% ===================== 核心防御机制：统一基础函数 =====================
function [res] = my_nmi_acc(F_normalized, gt, num_cluster)
    F_normalized = real(F_normalized); 
    max_rep = 20;
    res_rep = zeros(max_rep,8);
    F_normalized = F_normalized ./ repmat(sqrt(sum(F_normalized.^2,2)),1,size(F_normalized,2));
    
    for rep = 1:max_rep
        pre = kmeans(F_normalized,num_cluster,...
            'maxiter',100,'replicates',20,'emptyaction','singleton');
        res_rep(rep,:) = Clustering8Measure(gt,pre);
    end
    res = [mean(res_rep); std(res_rep)];
end