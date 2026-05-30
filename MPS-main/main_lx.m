function [F, obj] = main_lx(KH, HP, num_kernel, dim_c, num_cluster, alpha, sel_kernel, eta, opts)

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
if ~isfield(opts,'maxBetaIter'),    opts.maxBetaIter    = 10; end
if ~isfield(opts,'betaTol'),        opts.betaTol        = 1e-5; end

%% 预计算静态图 S 和 Laplacian L
S = cell(1, num_kernel);
L = cell(1, num_kernel);

for v = 1 : num_kernel
    sum_KH = KH(:, :, v);
    [S_tmp, ~] = update_graph(-sum_KH, opts.kNN_graph);

    S_tmp = max(real(S_tmp), 0);
    S_tmp = (S_tmp + S_tmp') / 2;
    S_tmp(1:num_sample+1:end) = 0;

    S{v} = S_tmp;

    D = diag(sum(S{v}, 2));
    L{v} = D - S{v};
end

maxIter = max(1, round(opts.maxIter));
tol = max(opts.tol, eps);
irls_eps = max(opts.irls_eps, eps);

flag = 1;
iter = 0;

sel_kernel = max(1, min(round(sel_kernel), num_kernel));
gamma = zeros(num_kernel, 1);
gamma(1:sel_kernel) = 1;

eta_static = eta * double(logical(opts.static_term_on));

while flag
    iter = iter + 1;

    %% 1. Update F
    tmp = zeros(num_sample);
    L_total = zeros(num_sample);

    for v = 1 : num_kernel
        if gamma(v) == 0
            continue;
        end

        Hp = HP{v};

        for d = 1 : dim_c
            A = Hp{d} * Hp{d}';
            tmp = tmp + alpha * mu(v) * beta(d) * A;
        end

        tmp = tmp + (1 - alpha) * omega(v) * S{v};
        L_total = L_total + L{v};
    end

    M_final = tmp - 0.5 * eta_static * L_total;
    M_final = (M_final + M_final') / 2;

    if any(~isfinite(M_final(:)))
        error('main_lx_jt:NonFiniteMatrix', ...
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

    %% 2. Update IRLS weights (mu and omega)
    FFt = F * F';

    for v = 1 : num_kernel
        if gamma(v) == 0
            mu(v) = 0;
            omega(v) = 0;
            continue;
        end

        Hp = HP{v};
        part_res_sq = 0;

        for d = 1 : dim_c
            A = Hp{d} * Hp{d}';
            R_part = FFt - beta(d) * A;
            part_res_sq = part_res_sq + norm(R_part, 'fro')^2;
        end

        mu(v) = 1 / (2 * sqrt(part_res_sq + irls_eps));
        omega(v) = 1 / (2 * sqrt(norm(FFt - S{v}, 'fro')^2 + irls_eps));
    end

    %% 3. Update beta (EXACT update for the full subproblem)
    % Solve: min_{beta >= 0, ||beta||_2=1} sum_j (C_j * beta_j^2 - 2*Theta_j * beta_j)
    % where: Theta_j = sum_i mu_i * Tr(F' * H_j^{(i)} * H_j^{(i)T} * F)
    %        C_j = sum_i mu_i * ||H_j^{(i)} * H_j^{(i)T}||_F^2 = (j * num_cluster) * sum_i mu_i
    
    % Compute Theta_j and total_mu
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
        Theta(d) = max(theta_d, 0);
    end
    
    % Compute C_j = (j * num_cluster) * total_mu
    Cj = (1:dim_c)' * num_cluster * total_mu;
    
    % Solve for lambda using bisection method
    % Equation: sum_j (Theta_j / (C_j - lambda))^2 = 1, lambda < min(C_j)
    
    if total_mu <= eps || all(Theta <= eps)
        % Degenerate case: use uniform beta
        beta = ones(dim_c, 1) / sqrt(dim_c);
    else
        % Find valid lambda range
        lambda_low = -1e10;  % f(lambda) -> 0 as lambda -> -inf
        lambda_high = min(Cj) - 1e-10;  % f(lambda) -> +inf as lambda -> min(Cj) from below
        
        % If Theta_j is zero for the smallest C_j, we may need to adjust
        % Find the smallest C_j for which Theta_j > 0
        valid_indices = find(Theta > 0);
        if isempty(valid_indices)
            beta = ones(dim_c, 1) / sqrt(dim_c);
        else
            % The effective upper bound is the smallest C_j among those with Theta_j > 0
            min_valid_Cj = min(Cj(valid_indices));
            lambda_high = min(lambda_high, min_valid_Cj - 1e-10);
            
            if lambda_high <= lambda_low
                % Fallback: use approximate solution
                beta = Theta / norm(Theta);
                if any(~isfinite(beta)) || norm(beta) <= eps
                    beta = ones(dim_c, 1) / sqrt(dim_c);
                else
                    beta = beta / norm(beta);
                end
            else
                % Bisection search
                max_lambda_iter = 100;
                lambda_tol = 1e-10;
                
                for lambda_iter = 1:max_lambda_iter
                    lambda_mid = (lambda_low + lambda_high) / 2;
                    
                    beta_candidate = zeros(dim_c, 1);
                    for d = 1:dim_c
                        denom = Cj(d) - lambda_mid;
                        if denom <= 0 || Theta(d) <= 0
                            beta_candidate(d) = 0;
                        else
                            beta_candidate(d) = Theta(d) / denom;
                        end
                    end
                    
                    norm_sq = sum(beta_candidate.^2);
                    
                    if norm_sq < 1
                        lambda_low = lambda_mid;
                    else
                        lambda_high = lambda_mid;
                    end
                    
                    if abs(norm_sq - 1) < lambda_tol
                        break;
                    end
                end
                
                beta = beta_candidate;
                beta = max(beta, 0);
                if norm(beta) > eps
                    beta = beta / norm(beta);
                else
                    beta = ones(dim_c, 1) / sqrt(dim_c);
                end
            end
        end
    end

    %% 4. Update gamma (kernel selection)
    view_loss = inf(num_kernel, 1);

    for v = 1 : num_kernel
        Hp = HP{v};
        part_res_sq = 0;

        for d = 1 : dim_c
            A = Hp{d} * Hp{d}';
            R_part = FFt - beta(d) * A;
            part_res_sq = part_res_sq + norm(R_part, 'fro')^2;
        end

        graph_res_sq = norm(FFt - S{v}, 'fro')^2;
        manifold_loss = eta_static * trace(F' * L{v} * F);

        view_loss(v) = alpha * sqrt(part_res_sq + irls_eps) ...
                     + (1 - alpha) * sqrt(graph_res_sq + irls_eps) ...
                     + manifold_loss;
    end

    [~, view_ord] = sort(real(view_loss), 'ascend');

    gamma = zeros(num_kernel, 1);
    gamma(view_ord(1:sel_kernel)) = 1;

    %% 5. Calculate objective for convergence monitoring
    obj1 = 0;
    obj2 = 0;
    reg_term_static = 0;

    for v = 1 : num_kernel
        if gamma(v) == 0
            continue;
        end

        Hp = HP{v};
        part_res_sq = 0;

        for d = 1 : dim_c
            A = Hp{d} * Hp{d}';
            R_part = FFt - beta(d) * A;
            part_res_sq = part_res_sq + norm(R_part, 'fro')^2;
        end

        obj1 = obj1 + sqrt(part_res_sq + irls_eps);
        obj2 = obj2 + sqrt(norm(FFt - S{v}, 'fro')^2 + irls_eps);
        reg_term_static = reg_term_static + trace(F' * L{v} * F);
    end

    obj(iter) = alpha * obj1 ...
              + (1 - alpha) * obj2 ...
              + eta_static * reg_term_static;

    if ~isfinite(obj(iter))
        error('main_lx_jt:NonFiniteObjective', ...
              'obj 在第 %d 轮出现 NaN/Inf。', iter);
    end

    %% 6. Check convergence
    if iter > 2
        rel_change = abs(obj(iter - 1) - obj(iter)) / ...
                     max(abs(obj(iter - 1)), eps);
    else
        rel_change = inf;
    end

    if iter >= maxIter || (iter > 2 && rel_change < tol)
        flag = 0;
    end
end

obj = obj(:);

end