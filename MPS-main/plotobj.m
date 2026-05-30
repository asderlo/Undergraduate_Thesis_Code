close all;
clear;
clc;
warning off;

addpath("ClusteringMeasure\");
addpath("FormulationKernels\");
addpath("function\");

dataPath = 'Datasets/Multi_view-Datasets/';
datasetName = {'MSRC_v1', 'BBCSport', 'BRCA','Caltech101-20', 'EMNIST',...
    '3sources', 'ORL', 'scene-15_3v'};

ResSavePath = 'MPS/Res3/';
MaxResSavePath = 'MPS/maxRes/';

if(~exist(ResSavePath,'file'))
    mkdir(ResSavePath);
    addpath(genpath(ResSavePath));
end

if(~exist(MaxResSavePath,'file'))
    mkdir(MaxResSavePath);
    addpath(genpath(MaxResSavePath));
end

for dataIndex = 7
    dataName = [dataPath datasetName{dataIndex} '.mat'];
    load(dataName, 'fea', 'gt');

    ResBest = zeros(1, 8);
    ResStd  = zeros(1, 8);

    % ===== 新增：记录最优参数与对应 obj 曲线 =====
    best_r1 = NaN;
    best_r2 = NaN;
    best_obj = [];

    % Data Preparation
    tic;
    num_cluster = length(unique(gt));
    dim_c = 5;
    [KH, HP, num_kernel] = preprocess(fea, num_cluster, dim_c);
    time1 = toc;

    % parameters setting
    r1 = 0.8;
    r2 = 8;

    acc    = zeros(length(r1), length(r2));
    nmi    = zeros(length(r1), length(r2));
    ari    = zeros(length(r1), length(r2));
    Fscore = zeros(length(r1), length(r2));

    Runtime = zeros(1, length(r1)*length(r2));
    idx = 1;

    for r1Index = 1:length(r1)
        r1Temp = r1(r1Index);
        for r2Index = 1:length(r2)
            r2Temp = ceil(r2(r2Index));

            tic;
            fprintf('Please wait a few minutes\n');
            disp(['Dataset: ', datasetName{dataIndex}, ...
                ', --r1--: ', num2str(r1Temp), ', --r2--: ', num2str(r2Temp)]);

            % Main algorithm
            [F, obj] = main(KH, HP, num_kernel, dim_c, num_cluster, r1Temp, r2Temp);
            time2 = toc;

            tic;
            res = my_nmi_acc(real(F), gt, num_cluster);
            time3 = toc;

            Runtime(idx) = time1 + time2 + time3/20;
            disp(['runtime: ', num2str(Runtime(idx))]);
            idx = idx + 1;

            tempResBest(1, :) = res(1, :);
            tempResStd(1, :)  = res(2, :);

            acc(r1Index, r2Index)    = tempResBest(1, 7);
            nmi(r1Index, r2Index)    = tempResBest(1, 4);
            ari(r1Index, r2Index)    = tempResBest(1, 5);
            Fscore(r1Index, r2Index) = tempResBest(1, 1);

            resFile = [ResSavePath datasetName{dataIndex}, ...
                '-ACC=', num2str(tempResBest(1, 7)), ...
                '-r1=', num2str(r1Temp), ...
                '-r2=', num2str(r2Temp), '.mat'];
            save(resFile, 'tempResBest', 'tempResStd');

            % ===== 更新全局最优（按 ACC）并保存对应 obj 曲线 =====
            if tempResBest(1, 7) > ResBest(1, 7)
                ResBest(1, :) = tempResBest(1, :);
                ResStd(1, :)  = tempResStd(1, :);

                best_r1 = r1Temp;
                best_r2 = r2Temp;
                best_obj = obj; % 保存这组参数下的收敛曲线
            end
        end
    end

       % ===== 画最优参数对应的目标函数曲线（和截图一致风格）=====
    if ~isempty(best_obj)
        obj_plot = best_obj(:);                 % 用原始 objective（不归一化）
        iters = 1:length(obj_plot);

        figure('Color','w');
        plot(iters, obj_plot, '-o', 'LineWidth', 2, 'MarkerSize', 6);
        grid on; box on;

        xlabel('Number of iterations', 'FontSize', 12);
        ylabel('Objective Value', 'FontSize', 12);

        set(gca, 'FontSize', 11, 'LineWidth', 1);
        xlim([1, max(iters)]);

        % 让 y 轴范围更像论文图：稍微留白
        yr = max(obj_plot) - min(obj_plot);
        if yr < 1e-12
            ylim([obj_plot(1)-1, obj_plot(1)+1]);
        else
            ylim([min(obj_plot) - 0.05*yr, max(obj_plot) + 0.05*yr]);
        end

        % 底部居中添加 “(b) MSRC_v1”
        panel_tag = '(b)';  % 你想改成 (a)/(c) 就改这里
        txt = [panel_tag, ' ', datasetName{dataIndex}];
        annotation('textbox', [0 0.01 1 0.08], ...
            'String', txt, 'EdgeColor', 'none', ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 12);

        % 可选：保存高分辨率图片
        exportgraphics(gcf, ['obj_' datasetName{dataIndex} '.png'], 'Resolution', 300);
    end
end