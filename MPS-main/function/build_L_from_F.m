function Lc = build_L_from_F(F, k)
%BUILD_L_FROM_F 根据嵌入 F 构建 kNN 图拉普拉斯 Lc
% 说明：供外部脚本/函数调用（main_lx 内部也有同名子函数，不冲突）。

[n, ~] = size(F);
if nargin < 2 || isempty(k), k = 15; end

F = real(F);

% 两两距离（优先使用 pdist2_fast）
if exist('pdist2_fast','file')
    D = pdist2_fast(F, F, 'sqeuclidean');
else
    D = pdist2(F, F, 'squaredeuclidean');
end

% 构建对称 kNN 邻接矩阵
W = sparse(n,n);
for i = 1:n
    [~, idx_sorted] = sort(D(i,:), 'ascend');
    for t = 2:min(k+1, n)
        j = idx_sorted(t);
        W(i,j) = 1;
    end
end
W = max(W, W');

deg = diag(sum(W,2));
Lc = deg - W;
end

