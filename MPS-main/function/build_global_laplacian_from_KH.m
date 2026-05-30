function [L_global, S_cell, L_cell] = build_global_laplacian_from_KH(KH, kNN, w)

if nargin < 2 || isempty(kNN), kNN = 15; end

[n,~,V] = size(KH);

if nargin < 3 || isempty(w)
    w = ones(V,1) / V;
else
    w = w(:);
    if numel(w) ~= V
        error('w 的长度必须为 V (= size(KH,3)).');
    end
    w = w / max(sum(w), eps);
end

S_cell = cell(V,1);
L_cell = cell(V,1);
L_global = zeros(n);

for v = 1:V
    Kv = KH(:,:,v);
    % 与 main_lx 保持一致：用 -K 作为“距离”输入 update_graph
    [Sv,~] = update_graph(-Kv, kNN);
    Sv = (Sv + Sv')/2;
    S_cell{v} = Sv;

    Dv = diag(sum(Sv, 2));
    Lv = Dv - Sv;
    L_cell{v} = Lv;

    L_global = L_global + w(v) * Lv;
end

L_global = (L_global + L_global')/2;
end

