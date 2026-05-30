function L = laplacian_from_similarity(S, opts)
%LAPLACIAN_FROM_SIMILARITY Construct graph Laplacian from similarity S.
%
% Inputs
%   S    : n x n similarity (assumed nonnegative; will be symmetrized by default)
%   opts : struct (optional)
%       .symmetric  (default true): use S_sym=(S+S')/2
%       .normalized (default false): return normalized Laplacian
%
% Output
%   L : n x n Laplacian (unnormalized or normalized)

if nargin < 2 || isempty(opts), opts = struct(); end
if ~isfield(opts,'symmetric'),  opts.symmetric = true; end
if ~isfield(opts,'normalized'), opts.normalized = false; end

if opts.symmetric
    S = (S + S')/2;
end

deg = sum(S, 2);

if ~opts.normalized
    L = diag(deg) - S;
    L = (L + L')/2;
    return;
end

% Normalized Laplacian: L = I - D^{-1/2} S D^{-1/2}
dinv = deg;
dinv(dinv < eps) = 1;
dinv_sqrt = 1 ./ sqrt(dinv);
Dn = spdiags(dinv_sqrt, 0, size(S,1), size(S,2));
L = speye(size(S,1)) - Dn * S * Dn;
L = (L + L')/2;
end

