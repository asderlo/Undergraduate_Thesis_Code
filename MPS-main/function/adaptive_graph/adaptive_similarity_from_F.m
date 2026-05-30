function [S, info] = adaptive_similarity_from_F(F, k, opts)
%ADAPTIVE_SIMILARITY_FROM_F Build adaptive kNN graph from embedding F.
%
% This implements "adaptive manifold graph learning" where the similarity
% matrix S is recomputed from the current clustering representation F.
%
% Inputs
%   F    : n x c embedding (e.g., spectral/cluster indicator)
%   k    : number of neighbors
%   opts : struct (optional)
%       .symmetric      (default true)  : force S=(S+S')/2
%       .use_sparse     (default false) : return sparse S (top-k only)
%       .normalize_rows (default false) : row-normalize S to sum=1
%
% Outputs
%   S    : n x n similarity matrix
%   info : struct with fields .Q (from update_graph), .k_eff

if nargin < 2 || isempty(k), k = 15; end
if nargin < 3 || isempty(opts), opts = struct(); end
if ~isfield(opts,'symmetric'),      opts.symmetric = true; end
if ~isfield(opts,'use_sparse'),     opts.use_sparse = false; end
if ~isfield(opts,'normalize_rows'), opts.normalize_rows = false; end

F = real(F);
[n, ~] = size(F);

% update_graph internally uses neighbors 2:(k+2), so we need k <= n-2.
k_eff = max(1, min(k, n-2));

% Pairwise squared Euclidean distance in F-space
if exist('pdist2_fast','file')
    D = pdist2_fast(F, F, 'sqeuclidean');
else
    D = pdist2(F, F, 'squaredeuclidean');
end

% Build adaptive similarity using existing LLE-style weighting
[S, Q] = update_graph(D, k_eff);

if opts.symmetric
    S = (S + S') / 2;
end

if opts.normalize_rows
    rs = sum(S, 2);
    rs(rs < eps) = 1;
    S = S ./ rs;
end

if opts.use_sparse
    S = sparse(S);
end

info = struct('Q', Q, 'k_eff', k_eff);
end

