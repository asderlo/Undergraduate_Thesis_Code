function [KH, Y] = fea2KH_MPS(fea, gt)
% fea2KH_MPS - 按 MPS (FormulationKernels/preprocess) 的方式从 fea、gt 构造多核矩阵 KH
% 每视图先 L2 行归一化，再建 5 种核：Gaussian(t=1), Polynomial(d=3), Linear, Sigmoid(c=0,d=0.1), InvPloyPlus(c=0.01,d=1)
% 最后 kcenter + knorm，与 SimpleMKKM 的输入一致。
%
% 输入: fea — cell（每元素 n x d_v）或 单矩阵 n x d（视为单视图）
%       gt  — n x 1 标签
% 输出: KH — n x n x numker，numker = num_view * 5

if ~iscell(fea)
    fea = {fea};
end
fea = fea(:);
num_view = length(fea);
n = size(fea{1}, 1);

% 标签整理为连续正整数
if nargin < 2 || isempty(gt)
    Y = ones(n, 1);
else
    if iscell(gt), gt = cell2mat(gt); end
    if istable(gt), gt = table2array(gt); end
    Y = double(gt(:));
    u = unique(Y);
    u(u < 1) = [];
    if isempty(u), u = 1; end
    for i = 1:length(u)
        Y(Y == u(i)) = i;
    end
end
if length(Y) ~= n
    error('fea2KH_MPS:lengthMismatch', 'fea 样本数 n=%d 与 gt 长度 %d 不一致。', n, length(Y));
end

% 每视图 L2 行归一化（与 MPS normalize_fea(fea, 1) 一致）
fea_normalized = cell(num_view, 1);
for v = 1:num_view
    X = double(fea{v});
    nrm = sqrt(sum(X.^2, 2));
    nrm(nrm < 1e-14) = 1;
    fea_normalized{v} = X ./ nrm;
end

% 每视图 5 个核（与 MPS preprocess 完全一致）
num_kernel = num_view * 5;
KH = zeros(n, n, num_kernel);
for v = 1:num_view
    X = fea_normalized{v};
    % 1) Gaussian, t=1
    D2 = euDist2_sq(X, []);
    KH(:, :, 1 + (v-1)*5) = exp(-D2 / (2*1^2));
    % 2) Polynomial, d=3
    G = X * X';
    KH(:, :, 2 + (v-1)*5) = (G).^3;
    % 3) Linear
    KH(:, :, 3 + (v-1)*5) = full(G);
    % 4) Sigmoid, c=0, d=0.1
    KH(:, :, 4 + (v-1)*5) = tanh(0.1*G + 0);
    % 5) InvPloyPlus, c=0.01, d=1
    KH(:, :, 5 + (v-1)*5) = (D2 + 0.01).^(-1);
end

% 对称化
for k = 1:num_kernel
    KH(:,:,k) = (KH(:,:,k) + KH(:,:,k)') / 2;
end
% MPS: KH = knorm(kcenter(KH))
KH = kcenter(KH);
KH = knorm(KH);

end

function D2 = euDist2_sq(fea_a, fea_b)
% 平方欧氏距离矩阵（与 MPS EuDist2(., [], 0) 一致）
if isempty(fea_b)
    aa = sum(fea_a.*fea_a, 2);
    ab = fea_a * fea_a';
    nSmp = size(fea_a, 1);
    D2 = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
    D2 = max(D2, D2');
    D2 = abs(D2);
else
    aa = sum(fea_a.*fea_a, 2);
    bb = sum(fea_b.*fea_b, 2);
    ab = fea_a * fea_b';
    D2 = repmat(aa, 1, size(fea_b,1)) + repmat(bb', size(fea_a,1), 1) - 2*ab;
    D2 = abs(D2);
end
end
