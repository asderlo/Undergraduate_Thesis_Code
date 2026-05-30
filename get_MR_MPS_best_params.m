function [s] = get_MR_MPS_best_params(datasetName)
% get_MR_MPS_best_params  MR-MPS / MPS 最佳参数（优先使用三阶段搜索结果）
% 用法: s = get_MR_MPS_best_params('BBCSport');
% 返回: s.alpha_mps, s.sel_kernel_mps (MPS 最优)
%       s.alpha_lx,  s.sel_kernel_lx, s.eta_lx, s.gamma_lx (MR-MPS / main_lx 最优)
%
% 【三阶段最优】若存在 MPS-main/MR_MPS/best_params_3stage.mat（由 MR_main_3stage_eta_gamma
% 调用 save_best_params_3stage 生成），则 alpha_lx / sel_kernel_lx / eta_lx / gamma_lx
% 自动覆盖下方 switch 中的手写默认值。

switch datasetName
    case 'MSRC_v1'
        alpha_mps = 0.50; sel_kernel_mps = 3;
        alpha_lx = 0.20;  sel_kernel_lx = 15; eta_lx = 1; gamma_lx = 0;
    case 'BBCSport'
        alpha_mps = 0.90; sel_kernel_mps = 2;
        alpha_lx = 0.90;  sel_kernel_lx = 9;  eta_lx = 10; gamma_lx = 0;
    case 'BRCA'
        alpha_mps = 0.40; sel_kernel_mps = 16;
        alpha_lx = 0.20;  sel_kernel_lx = 16; eta_lx = 0.1; gamma_lx = 0;
    case 'WikipediaArticles'
        alpha_mps = 0.90; sel_kernel_mps = 2;
        alpha_lx = 0.90;  sel_kernel_lx = 1;  eta_lx = 1; gamma_lx = 0;
    case '3sources'
        alpha_mps = 0.90; sel_kernel_mps = 6;
        alpha_lx = 0.90;  sel_kernel_lx = 5;  eta_lx = 1; gamma_lx = 0;
    case 'ORL'
        alpha_mps = 0.90; sel_kernel_mps = 9;
        alpha_lx = 0.90;  sel_kernel_lx = 6;  eta_lx = 0.1; gamma_lx = 0;
    case 'Prokaryotic'
        alpha_mps = 0.60; sel_kernel_mps = 2;
        % alpha_lx = 0.90;  sel_kernel_lx = 6;  eta_lx = 0.1; gamma_lx = 0;
    case 'WebKB'
        alpha_mps = 0.10; sel_kernel_mps = 14;
        % alpha_lx = 0.90;  sel_kernel_lx = 6;  eta_lx = 0.1; gamma_lx = 0;
    case 'BBC'
        alpha_mps = 0.90; sel_kernel_mps = 12;
    otherwise
        error('get_MR_MPS_best_params: 未知数据集 "%s"，请与 campare_end.m 同步添加。', datasetName);
end

% ---------- 优先加载 MR_main_3stage_eta_gamma 保存的三阶段最优 ----------
root = fileparts(mfilename('fullpath'));
matFile = fullfile(root, 'MPS-main', 'MR_MPS', 'best_params_3stage.mat');
if isfile(matFile)
    L = load(matFile, 'entries');
    if isfield(L, 'entries') && ~isempty(L.entries)
        names = {L.entries.dataset};
        j = find(strcmp(names, datasetName), 1);
        if ~isempty(j)
            e = L.entries(j);
            alpha_lx = e.alpha_lx;
            sel_kernel_lx = e.sel_kernel_lx;
            eta_lx = e.eta_lx;
            gamma_lx = e.gamma_lx;
        end
    end
end

s = struct('alpha_mps', alpha_mps, 'sel_kernel_mps', sel_kernel_mps, ...
           'alpha_lx', alpha_lx, 'sel_kernel_lx', sel_kernel_lx, 'eta_lx', eta_lx, ...
           'gamma_lx', gamma_lx);
end
