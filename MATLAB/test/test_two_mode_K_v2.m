function test_two_mode_K_v2(saveDir)
% Test K-constant with proper radiation coefficients a_n.
%
% S_far = 0  <=>  sum_{n sym} a_n * W_n(x_M) / D_n(EI) = 0
%
% where a_n = integral of W_n(x) * cos(kx) dx over raft [-L/2, L/2]
% (Fourier projection of mode shape onto traveling wave)

if nargin < 1, saveDir = 'data'; end
addpath('../src');

Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L_raft = St(1).args.L_raft;
omega  = St(1).args.omega;
rho_R  = St(1).args.rho_raft;
rho_w  = St(1).args.rho;
d      = St(1).args.d;
g      = St(1).args.g;
H      = St(1).args.domainDepth;
k      = real(St(1).args.k);

fprintf('Parameters: L=%.4f, omega=%.1f, d=%.4f, k=%.2f\n', L_raft, omega, d, k);
fprintf('kL = %.2f (%.1f wavelengths per raft length)\n', k*L_raft, k*L_raft/(2*pi));

% Added mass (real, mode-independent approximation)
m_a = d * rho_w / (k * tanh(k * H));

% ---- Build symmetric mode bank ----
n_elastic = 6;
betaL_roots = freefree_betaL_roots(n_elastic);

% Collect symmetric modes: rigid (n=0) + elastic 1,3,5
sym_beta = [0; betaL_roots(1)/L_raft; betaL_roots(3)/L_raft; betaL_roots(5)/L_raft];
sym_labels = {'rigid', 'elastic 1', 'elastic 3', 'elastic 5'};
n_sym = numel(sym_beta);

% Evaluate mode shapes on fine grid for normalization and integration
xi_fine = linspace(0, L_raft, 5000)';
x_fine  = xi_fine - L_raft/2;  % centered coordinates [-L/2, L/2]

% Build normalized mode shapes on centered coordinates
Psi = zeros(numel(xi_fine), n_sym);

% Rigid mode
Psi(:,1) = 1 / sqrt(L_raft);

% Elastic symmetric modes
for j = 2:n_sym
    idx_elastic = [1, 3, 5];
    bL = betaL_roots(idx_elastic(j-1));
    psi_raw = freefree_mode_shape(xi_fine, L_raft, bL);
    L2_norm = sqrt(trapz(xi_fine, psi_raw.^2));
    Psi(:,j) = psi_raw / L2_norm;
end

% ---- Compute radiation coefficients a_n ----
% a_n = integral of W_n(x) * cos(kx) dx over [-L/2, L/2]
a = zeros(n_sym, 1);
for j = 1:n_sym
    integrand = Psi(:,j) .* cos(k * x_fine);
    a(j) = trapz(x_fine, integrand);
end

fprintf('\n=== Radiation coefficients a_n ===\n');
for j = 1:n_sym
    fprintf('  %-12s: beta=%.2f, a_n = %+.6e\n', sym_labels{j}, sym_beta(j), a(j));
end

% ---- Function to evaluate W_n at arbitrary x_M ----
% x_M is measured from center
W_at_xM = @(xM) interp1(x_fine, Psi, xM, 'linear');  % returns row of n_sym values

% ---- Extract lowest S~0 curve (same as v1) ----
mp_list = unique([St.motor_position]);
EI_list = unique([St.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);
MP_norm_list = mp_list / L_raft;

eta_1   = reshape([St.eta_1],   n_mp, n_EI);
eta_end = reshape([St.eta_end], n_mp, n_EI);
eta_1_sq   = abs(eta_1).^2;
eta_end_sq = abs(eta_end).^2;
asymmetry  = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);
S_grid = (eta_end + eta_1) / 2;
A_grid = (eta_end - eta_1) / 2;
SA_ratio = log10(abs(S_grid) ./ (abs(A_grid) + eps));

curve_EI = [];
curve_mp = [];

for ie = 1:n_EI
    col = asymmetry(:, ie);
    crossings_mp = [];
    crossings_SA = [];
    for im = 1:(n_mp-1)
        if col(im) * col(im+1) < 0
            t = col(im) / (col(im) - col(im+1));
            mp_zero = MP_norm_list(im) + t * (MP_norm_list(im+1) - MP_norm_list(im));
            sa_col = SA_ratio(:, ie);
            sa_zero = sa_col(im) + t * (sa_col(im+1) - sa_col(im));
            crossings_mp(end+1) = mp_zero; %#ok<AGROW>
            crossings_SA(end+1) = sa_zero; %#ok<AGROW>
        end
    end
    if ~isempty(crossings_mp)
        s0_mask = crossings_SA < 0;
        if any(s0_mask)
            candidates_mp = crossings_mp(s0_mask);
            [mp_pick, ~] = min(candidates_mp);
            curve_EI(end+1) = EI_list(ie); %#ok<AGROW>
            curve_mp(end+1) = mp_pick; %#ok<AGROW>
        end
    end
end

fprintf('\nExtracted %d points on lowest S~0 curve\n', numel(curve_EI));

% ---- Compute S_pred = sum a_n * W_n(x_M) / D_n(EI) at each point ----
% And test if the N-mode prediction gives S ~ 0 on the curve

fprintf('\n=== Modal contributions along the curve ===\n');
fprintf('%-11s | %-7s', 'EI', 'x_M/L');
for j = 1:n_sym
    fprintf(' | %-11s', sym_labels{j});
end
fprintf(' | %-11s | %-11s\n', 'S_pred', 'S_pred/max');
fprintf('%s\n', repmat('-', 1, 12 + 10 + n_sym*14 + 28));

S_pred_all = zeros(size(curve_EI));
max_contrib = zeros(size(curve_EI));

for ip = 1:numel(curve_EI)
    EI_val = curve_EI(ip);
    xM_val = curve_mp(ip) * L_raft;  % meters from center

    % D_n for each symmetric mode
    D = EI_val * sym_beta.^4 + d * rho_w * g - omega^2 * (rho_R + m_a);

    % W_n(x_M) for each symmetric mode
    W = W_at_xM(xM_val);  % 1 x n_sym

    % Individual contributions: a_n * W_n(x_M) / D_n
    contribs = (a .* W(:)) ./ D;

    S_pred = sum(contribs);
    S_pred_all(ip) = S_pred;
    max_contrib(ip) = max(abs(contribs));

    fprintf('%10.3e | %6.4f', EI_val, curve_mp(ip));
    for j = 1:n_sym
        fprintf(' | %+10.3e', contribs(j));
    end
    fprintf(' | %+10.3e | %+10.3e\n', S_pred, S_pred/max_contrib(ip));
end

fprintf('\n=== S_pred statistics ===\n');
fprintf('  If the model is correct, S_pred should be ~0 on the curve.\n');
fprintf('  |S_pred|/max_contrib: mean=%.3e, max=%.3e\n', ...
    mean(abs(S_pred_all)./max_contrib), max(abs(S_pred_all)./max_contrib));

% ---- Also: predict the curve by finding x_M where S_pred=0 for each EI ----
EI_pred = logspace(log10(min(EI_list)), log10(max(EI_list)), 200);
xM_pred = NaN(size(EI_pred));
xM_scan = linspace(0, L_raft/2*0.96, 500);  % scan x_M from center to near-edge

for ie = 1:numel(EI_pred)
    EI_val = EI_pred(ie);
    D = EI_val * sym_beta.^4 + d * rho_w * g - omega^2 * (rho_R + m_a);

    % Skip if any D_n is very close to zero (resonance)
    if any(abs(D) < 1e-2 * max(abs(D))), continue; end

    % Evaluate S_pred(x_M) on scan grid
    S_scan = zeros(size(xM_scan));
    for ix = 1:numel(xM_scan)
        W = W_at_xM(xM_scan(ix));
        S_scan(ix) = sum((a .* W(:)) ./ D);
    end

    % Find first zero crossing (lowest x_M)
    for ix = 1:(numel(xM_scan)-1)
        if S_scan(ix) * S_scan(ix+1) < 0
            t = S_scan(ix) / (S_scan(ix) - S_scan(ix+1));
            xM_pred(ie) = xM_scan(ix) + t * (xM_scan(ix+1) - xM_scan(ix));
            break;
        end
    end
end

% ---- Plot comparison ----
fig = figure('Color','w','Units','centimeters','Position',[2 2 24 16]);
ax = gca; hold(ax, 'on');

% Background: asymmetry
contourf(ax, reshape([St.EI],n_mp,n_EI), ...
    reshape([St.motor_position],n_mp,n_EI)/L_raft, ...
    asymmetry, 50, 'LineStyle','none');
contour(ax, reshape([St.EI],n_mp,n_EI), ...
    reshape([St.motor_position],n_mp,n_EI)/L_raft, ...
    asymmetry, [0 0], 'LineColor','k','LineWidth',1.5);
caxis([-1 1]); colormap(ax, bwr_colormap()); colorbar;

% Data curve (lowest S~0)
plot(ax, curve_EI, curve_mp, 'ko', 'MarkerSize', 6, 'LineWidth', 1.5, ...
    'DisplayName', 'Data (\alpha=0, lowest S\approx0)');

% Predicted curve
valid = ~isnan(xM_pred);
plot(ax, EI_pred(valid), xM_pred(valid)/L_raft, 'r-', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Modal prediction (%d sym modes)', n_sym));

set(ax,'XScale','log');
xlabel('EI (N m^4)'); ylabel('x_M / L');
title('S_{far}=0 prediction vs data');
legend('show','Location','northeast','FontSize',11);
set(ax,'FontSize',12);

outfile = fullfile(saveDir, 'two_mode_K_v2.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[24 16],'PaperPosition',[0 0 24 16]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);

end

function betaL = freefree_betaL_roots(n)
    betaL = zeros(n, 1);
    f = @(y) cosh(y) .* cos(y) - 1;
    for jj = 1:n
        a = jj * pi;
        b = (jj + 1) * pi;
        betaL(jj) = fzero(f, [a, b]);
    end
end

function psi = freefree_mode_shape(xi, L, betaL)
    beta = betaL / L;
    bx = beta * xi;
    alpha = (sin(betaL) - sinh(betaL)) / (cosh(betaL) - cos(betaL));
    psi = (sin(bx) + sinh(bx)) + alpha * (cos(bx) + cosh(bx));
    scale = max(abs(psi));
    if isfinite(scale) && scale > 0
        psi = psi / scale;
    end
end

function cmap = bwr_colormap(n_colors, gamma)
    if nargin < 1, n_colors = 256; end
    if nargin < 2, gamma = 1.3; end
    rgb_anchors = [0.99 0.35 0.00; 1 1 1; 0 0.35 0.80];
    lab_anchors = rgb2lab(rgb_anchors);
    t_lin = linspace(-1,1,n_colors).';
    t_nl = sign(t_lin) .* (abs(t_lin).^gamma);
    lab_interp = interp1([-1 0 1].', lab_anchors, t_nl, 'linear');
    cmap = max(min(lab2rgb(lab_interp),1),0);
end
