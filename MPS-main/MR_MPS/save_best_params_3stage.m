function entries = save_best_params_3stage(datasetName, best1, best2, best3)
%SAVE_BEST_PARAMS_3STAGE 将 MR_main_3stage_eta_gamma 的最优参数写入 MR_MPS/best_params_3stage.mat
% 供 get_MR_MPS_best_params / plot_convergence_curve 等自动读取“三阶段最优”。

thisDir = fileparts(mfilename('fullpath'));
matFile = fullfile(thisDir, 'best_params_3stage.mat');

if isfile(matFile)
    L = load(matFile, 'entries');
    entries = L.entries;
else
    entries = struct( ...
        'dataset', {}, ...
        'alpha_lx', {}, ...
        'sel_kernel_lx', {}, ...
        'eta_lx', {}, ...
        'gamma_lx', {}, ...
        'acc_3stage', {});
end

rec = struct( ...
    'dataset',        datasetName, ...
    'alpha_lx',       best1.alpha, ...
    'sel_kernel_lx',  best1.sel_k, ...
    'eta_lx',         best2.eta, ...
    'gamma_lx',       best3.gamma, ...
    'acc_3stage',     best3.ACC );

if isempty(entries)
    entries = rec;
else
    names = {entries.dataset};
    j = find(strcmp(names, datasetName), 1);
    if isempty(j)
        entries(end+1, 1) = rec;
    else
        entries(j) = rec;
    end
end

save(matFile, 'entries', '-v7.3');
fprintf('已写入三阶段最优参数: %s\n', matFile);

csvFile = fullfile(thisDir, 'best_params_3stage.csv');
fid = fopen(csvFile, 'w');
if fid < 0
    warning('无法写入 CSV: %s', csvFile);
else
    fprintf(fid, 'dataset,alpha_lx,sel_kernel_lx,eta_lx,gamma_lx,acc_3stage\n');
    for ii = 1:numel(entries)
        e = entries(ii);
        fprintf(fid, '%s,%.8g,%d,%.8g,%.8g,%.8g\n', ...
            e.dataset, e.alpha_lx, e.sel_kernel_lx, e.eta_lx, e.gamma_lx, e.acc_3stage);
    end
    fclose(fid);
end
end
