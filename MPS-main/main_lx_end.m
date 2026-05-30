function [F, obj] = main_lx_end(KH, HP, num_kernel, dim_c, num_cluster, alpha, sel_kernel, eta, opts)
%% Initialization
beta = ones(dim_c, 1) / sqrt(dim_c);
mu = ones(num_kernel, 1);
omega = ones(num_kernel, 1);

[num_sample, ~, num_kernel] = size(KH);

if nargin < 9 || isempty(opts), opts = struct(); end

thisDir = fileparts(mfilename('fullpath'));
if ~isempty(thisDir)
    agDir = fullfile(thisDir, 'function', 'adaptive_graph');
    if exist(agDir, 'dir')
        addpath(agDir);
    end
end

if ~isfield(opts,'kNN_graph'),      opts.kNN_graph      = 15; end
if ~isfield(opts,'maxIter'),        opts.maxIter        = 50; end
if ~isfield(opts,'static_term_on'), opts.static_term_on = true; end
if ~isfield(opts,'irls_eps'),       opts.irls_eps       = 1e-8; end
if ~isfield(opts,'tol'),            opts.tol            = 1e-6; end
if ~isfield(opts,'betaTol'),        opts.betaTol        = 1e-10; end
if ~isfield(opts,'maxBetaIter'),    opts.maxBetaIter    = 10; end

maxIter = max(1, round(opts.maxIter));
tol = max(opts.tol, eps);
irls_eps = max(opts.irls_eps, eps);

sel_kernel = max(1, min(round(sel_kernel), num_kernel));
eta_static = eta * double(logical(opts.static_term_on));

%% Precompute static graph S and Laplacian L
S = cell(1, num_kernel);
L = cell(1, num_kernel);

for v = 1:num_kernel
    Kv = KH(:, :, v);

    [S_tmp, ~] = update_graph(-Kv, opts.kNN_graph);

    S_tmp = max(real(S_tmp), 0);
    S_tmp = (S_tmp + S_tmp') / 2;
    S_tmp(1:num_sample+1:end) = 0;

    S{v} = S_tmp;

    D = diag(sum(S_tmp, 2));
    L{v} = D - S_tmp;
end

%% Initialize gamma
gamma = zeros(num_kernel, 1);
gamma(1:sel_kernel) = 1;

%% Initialize F
F = update_F(HP, S, L, gamma, beta, mu, omega, ...
             alpha, eta_static, dim_c, num_cluster, num_sample);

obj = zeros(maxIter, 1);

%% Main loop
for iter = 1:maxIter

    FFt = F * F';

    %% 1. Update gamma 
    view_loss = inf(num_kernel, 1);

    for v = 1:num_kernel
        Hp = HP{v};
        part_res_sq = 0;

        for d = 1:dim_c
            A = Hp{d} * Hp{d}';
            R_part = FFt - beta(d) * A;
            part_res_sq = part_res_sq + norm(R_part, 'fro')^2;
        end

        graph_res_sq = norm(FFt - S{v}, 'fro')^2;
        manifold_loss = trace(F' * L{v} * F);

        view_loss(v) = alpha * sqrt(part_res_sq) ...
                     + (1 - alpha) * sqrt(graph_res_sq) ...
                     + eta_static * manifold_loss;
    end

    [~, view_ord] = sort(real(view_loss), 'ascend');

    gamma = zeros(num_kernel, 1);
    gamma(view_ord(1:sel_kernel)) = 1;

    %% 2. Update mu and omega
    for v = 1:num_kernel
        if gamma(v) == 0
            mu(v) = 0;
            omega(v) = 0;
            continue;
        end

        Hp = HP{v};
        part_res_sq = 0;

        for d = 1:dim_c
            A = Hp{d} * Hp{d}';
            R_part = FFt - beta(d) * A;
            part_res_sq = part_res_sq + norm(R_part, 'fro')^2;
        end

        graph_res_sq = norm(FFt - S{v}, 'fro')^2;

        mu(v) = 1 / (2 * sqrt(part_res_sq + irls_eps));
        omega(v) = 1 / (2 * sqrt(graph_res_sq + irls_eps));
    end

    %% 3. Update F
    F = update_F(HP, S, L, gamma, beta, mu, omega, ...
                 alpha, eta_static, dim_c, num_cluster, num_sample);

    F = real(F);
    [F, ~] = qr(F, 0);
    FFt = F * F';

    %% 4. Update beta 
    Theta = zeros(dim_c, 1);
    total_mu = 0;
    
    for v = 1:num_kernel
        if gamma(v) == 0
            continue;
        end
        total_mu = total_mu + mu(v);
    end
    
    for d = 1:dim_c
        theta_d = 0;
    
        for v = 1:num_kernel
            if gamma(v) == 0
                continue;
            end
    
            Hp = HP{v};
            A = Hp{d} * Hp{d}';
    
            theta_d = theta_d + mu(v) * trace(F' * A * F);
        end
    
        Theta(d) = max(real(theta_d), 0);
    end
    
    Cj = (1:dim_c)' * num_cluster * total_mu;
    Cj = max(real(Cj), eps);

    beta = update_beta_exact(Theta, Cj, dim_c, opts.maxBetaIter, opts.betaTol);

    %% 5. Calculate objective
    obj1 = 0;
    obj2 = 0;
    reg_term_static = 0;

    for v = 1:num_kernel
        if gamma(v) == 0
            continue;
        end

        Hp = HP{v};
        part_res_sq = 0;

        for d = 1:dim_c
            A = Hp{d} * Hp{d}';
            R_part = FFt - beta(d) * A;
            part_res_sq = part_res_sq + norm(R_part, 'fro')^2;
        end

        obj1 = obj1 + sqrt(part_res_sq);
        obj2 = obj2 + sqrt(norm(FFt - S{v}, 'fro')^2);
        reg_term_static = reg_term_static + trace(F' * L{v} * F);
    end

    obj(iter) = alpha * obj1 ...
              + (1 - alpha) * obj2 ...
              + eta_static * reg_term_static;

    if ~isfinite(obj(iter))
        error('main_lx:NonFiniteObjective', ...
              'obj 在第 %d 轮出现 NaN/Inf。', iter);
    end

    %% 6. Check convergence
    if iter > 2
        rel_change = abs(obj(iter - 1) - obj(iter)) / ...
                     max(abs(obj(iter - 1)), eps);

        if rel_change < tol
            obj = obj(1:iter);
            break;
        end
    end

    if iter == maxIter
        obj = obj(1:iter);
    end
end

obj = obj(:);

end


function F = update_F(HP, S, L, gamma, beta, mu, omega, ...
                      alpha, eta_static, dim_c, num_cluster, num_sample)

tmp = zeros(num_sample);
L_total = zeros(num_sample);

num_kernel = length(S);

for v = 1:num_kernel
    if gamma(v) == 0
        continue;
    end

    Hp = HP{v};

    for d = 1:dim_c
        A = Hp{d} * Hp{d}';
        tmp = tmp + alpha * mu(v) * beta(d) * A;
    end

    tmp = tmp + (1 - alpha) * omega(v) * S{v};
    L_total = L_total + L{v};
end

M_final = 2* tmp - eta_static * L_total;
M_final = (M_final + M_final') / 2;

if any(~isfinite(M_final(:)))
    error('main_lx:NonFiniteMatrix', ...
          'M_final 出现 NaN/Inf，无法继续迭代。');
end

try
    eig_opts.disp = 0;
    [F, ~] = eigs(M_final, num_cluster, 'la', eig_opts);
catch
    [V_all, D_all] = eig(full(M_final));
    [~, ord] = sort(real(diag(D_all)), 'descend');
    F = V_all(:, ord(1:num_cluster));
end

F = real(F);
[F, ~] = qr(F, 0);

end


function beta = update_beta_exact(Theta, Cj, dim_c, maxBetaIter, betaTol)

Theta = max(real(Theta(:)), 0);
Cj = max(real(Cj(:)), eps);

valid = Theta > eps;

if ~any(valid)
    beta = ones(dim_c, 1) / sqrt(dim_c);
    return;
end

lambda_low = -min(Cj(valid)) + 1e-12;
lambda_high = 1;

while true
    denom = Cj + lambda_high;
    beta_tmp = zeros(dim_c, 1);
    active = valid & denom > eps;
    beta_tmp(active) = Theta(active) ./ denom(active);

    if sum(beta_tmp.^2) < 1
        break;
    end

    lambda_high = lambda_high * 2;

    if lambda_high > 1e12
        break;
    end
end

for it = 1:maxBetaIter
    lambda_mid = (lambda_low + lambda_high) / 2;

    denom = Cj + lambda_mid;
    beta_tmp = zeros(dim_c, 1);
    active = valid & denom > eps;
    beta_tmp(active) = Theta(active) ./ denom(active);

    norm_sq = sum(beta_tmp.^2);

    if norm_sq > 1
        lambda_low = lambda_mid;
    else
        lambda_high = lambda_mid;
    end

    if abs(norm_sq - 1) < betaTol
        break;
    end
end

denom = Cj + lambda_high;
beta = zeros(dim_c, 1);
active = valid & denom > eps;
beta(active) = Theta(active) ./ denom(active);

beta = max(beta, 0);

if norm(beta) > eps
    beta = beta / norm(beta);
else
    beta = ones(dim_c, 1) / sqrt(dim_c);
end

end