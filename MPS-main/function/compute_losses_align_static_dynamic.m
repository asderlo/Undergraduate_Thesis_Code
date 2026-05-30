function stat = compute_losses_align_static_dynamic(F, KH, L_cell, mu, k_consensus)

[n,~,V] = size(KH);
if nargin < 4 || isempty(mu)
    mu = ones(V,1) / V;
else
    mu = mu(:);
    mu = mu / max(sum(mu), eps);
end
if nargin < 5 || isempty(k_consensus), k_consensus = 15; end

F = real(F);
FFt = F*F';

% ---- E_align: 相对对齐误差（对核尺度不敏感）----
FFv = FFt(:);                 % n^2 x 1
KHv = reshape(KH, [], V);     % n^2 x V
diff2 = sum((KHv - FFv).^2, 1)';   % Vx1
kn2   = sum(KHv.^2, 1)';           % Vx1
rel2  = diff2 ./ max(kn2, 1e-12);
E_align = mean(rel2);

% ---- E_static: 归一化 trace ----
t_static = zeros(V,1);
for v = 1:V
    Lv = (L_cell{v} + L_cell{v}')/2;
    t = trace(F' * Lv * F);
    trL = trace(Lv);
    t_static(v) = (t / max(trL, 1e-12)) * n;
end
E_static = mu' * t_static;

% ---- E_dynamic: 基于 F 的共识图 ----
L_star = build_L_from_F(F, k_consensus);
L_star = (L_star + L_star')/2;
td = trace(F' * L_star * F);
trLs = trace(L_star);
E_dynamic = (td / max(trLs, 1e-12)) * n;

stat = struct();
stat.E_align = E_align;
stat.E_static = E_static;
stat.E_dynamic = E_dynamic;
stat.L_star = L_star;
end

