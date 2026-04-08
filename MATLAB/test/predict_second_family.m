function predict_second_family(saveDir)
% Build an operational prediction f(x_M, EI) = 0 for the lowest S~0 curve.
%
% Strategy:
% 1. Run solver at ~15 points along the curve, decompose into modes -> q_n
% 2. Also record S_far = (eta_end + eta_1)/2 from each run
% 3. Fit radiation coefficients: S_far = sum a_n * q_n (symmetric modes only)
% 4. Predict: f(x_M, EI) = sum a_n * W_n(x_M) / D_n(EI) = 0

if nargin < 1, saveDir = 'data'; end
addpath('../src');

Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L_raft = St(1).args.L_raft;
omega  = St(1).args.omega;
rho_R  = St(1).args.rho_raft;
rho_w  = St(1).args.rho;
d_val  = St(1).args.d;
g_val  = St(1).args.g;
H      = St(1).args.domainDepth;
k_val  = real(St(1).args.k);
m_a    = d_val * rho_w / (k_val * tanh(k_val * H));
base   = St(1).args;

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

% ---- Extract lowest S~0 curve ----
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
            crossings_mp(end+1) = mp_zero;
            crossings_SA(end+1) = sa_zero;
        end
    end
    if ~isempty(crossings_mp)
        s0_mask = crossings_SA < 0;
        if any(s0_mask)
            curve_EI(end+1) = EI_list(ie);
            curve_mp(end+1) = min(crossings_mp(s0_mask));
        end
    end
end

% Remove the outlier at EI~2.7e-5 (jumped to wrong curve)
bad = abs(curve_mp - 0.3634) < 0.01;
curve_EI(bad) = [];
curve_mp(bad) = [];

% Sample 15 points
n_sample = 15;
sample_idx = round(linspace(1, numel(curve_EI), n_sample));
sample_EI = curve_EI(sample_idx);
sample_mp = curve_mp(sample_idx);

fprintf('Running solver at %d points...\n', n_sample);

n_modes = 8;
all_q    = zeros(n_sample, n_modes);
all_Sfar = zeros(n_sample, 1);  % complex

for ip = 1:n_sample
    EI_val = sample_EI(ip);
    mp_val = sample_mp(ip) * L_raft;

    [U, x, phi, phi_z, eta, args] = flexible_surferbot_v2( ...
        'sigma', base.sigma, 'rho', base.rho, 'nu', base.nu, ...
        'g', base.g, 'L_raft', base.L_raft, ...
        'motor_position', mp_val, 'd', base.d, ...
        'EI', EI_val, 'rho_raft', base.rho_raft, ...
        'domainDepth', base.domainDepth, 'L_domain', base.L_domain, ...
        'motor_inertia', base.motor_inertia, 'BC', base.BC, ...
        'omega', base.omega);

    Srun = struct('x', x, 'eta', eta, 'args', args);
    modal = decompose_raft_freefree_modes(Srun, 'num_modes', n_modes, 'verbose', false);

    all_q(ip, :) = modal.q.';
    all_Sfar(ip) = (eta(end) + eta(1)) / 2;

    if ip == 1
        mode_types = modal.mode_type;
        % Identify symmetric modes
        Psi = modal.Psi;
        Psi_left  = Psi(1, :);
        Psi_right = Psi(end, :);
        is_sym = abs(Psi_left + Psi_right) > abs(Psi_left - Psi_right);
        fprintf('Symmetric modes: ');
        for j = 1:n_modes
            if is_sym(j), fprintf('%d(%s) ', j-1, mode_types{j}); end
        end
        fprintf('\n');
    end
end

% ---- Fit radiation coefficients from symmetric modes only ----
sym_idx = find(is_sym);
n_sym = numel(sym_idx);
fprintf('\nFitting %d radiation coefficients from %d data points\n', n_sym, n_sample);

% S_far = sum_j a_j * q_j  for symmetric j
% In matrix form: S_far = Q_sym * a
Q_sym = all_q(:, sym_idx);  % n_sample x n_sym, complex

% Least-squares fit (complex)
a_fit = Q_sym \ all_Sfar;

% Check fit quality
S_pred_fit = Q_sym * a_fit;
residual = abs(all_Sfar - S_pred_fit) ./ abs(all_Sfar);

fprintf('\nFitted radiation coefficients a_n:\n');
for j = 1:n_sym
    fprintf('  mode %d (%s): a = %+.4e %+.4ei  (|a| = %.4e)\n', ...
        sym_idx(j)-1, mode_types{sym_idx(j)}, real(a_fit(j)), imag(a_fit(j)), abs(a_fit(j)));
end

fprintf('\nFit quality (relative residual |S_pred - S_actual|/|S_actual|):\n');
fprintf('  Mean: %.3e\n', mean(residual));
fprintf('  Max:  %.3e\n', max(residual));

% ---- Build prediction: f(x_M, EI) = sum a_n * W_n(x_M) / D_n(EI) ----
% Need mode shapes W_n and eigenvalues beta_n from the decomposition
% Re-run one case to get the mode shape basis
[~, x0, ~, ~, eta0, args0] = flexible_surferbot_v2( ...
    'sigma', base.sigma, 'rho', base.rho, 'nu', base.nu, ...
    'g', base.g, 'L_raft', base.L_raft, ...
    'motor_position', 0.2*base.L_raft, 'd', base.d, ...
    'EI', 1e-3, 'rho_raft', base.rho_raft, ...
    'domainDepth', base.domainDepth, 'L_domain', base.L_domain, ...
    'motor_inertia', base.motor_inertia, 'BC', base.BC, ...
    'omega', base.omega);
Sref = struct('x', x0, 'eta', eta0, 'args', args0);
modal_ref = decompose_raft_freefree_modes(Sref, 'num_modes', n_modes, 'verbose', false);

x_raft = modal_ref.x_raft;
Psi_basis = modal_ref.Psi;   % n_x x n_modes
beta_vals = modal_ref.beta;   % n_modes x 1

% Evaluate mode shapes on a fine x_M grid
% x_raft is in nondimensional coords (x/L_raft), range ~ [-0.5, 0.5]
% Motor position is always positive (one side of center)
xM_scan = linspace(0.01, max(x_raft)*0.95, 300);

% Interpolate each mode shape to the scan grid
W_scan = zeros(n_sym, numel(xM_scan));
for j = 1:n_sym
    W_scan(j, :) = interp1(x_raft, Psi_basis(:, sym_idx(j)), xM_scan, 'linear', 0);
end

% For each EI, compute f(x_M) = sum a_n * W_n(x_M) / D_n(EI) and find zeros
EI_pred = logspace(log10(min(EI_list)), log10(max(EI_list)), 500);
xM_pred = NaN(size(EI_pred));

for ie = 1:numel(EI_pred)
    EI_val = EI_pred(ie);

    % D_n for symmetric modes
    D_sym = zeros(n_sym, 1);
    for j = 1:n_sym
        beta_n = beta_vals(sym_idx(j));
        D_sym(j) = EI_val * beta_n^4 + d_val * rho_w * g_val - omega^2 * (rho_R + m_a);
    end

    % Skip near resonances
    if any(abs(D_sym) < 1e-3 * max(abs(D_sym))), continue; end

    % f(x_M) = sum a_n * W_n(x_M) / D_n  (complex)
    coeffs = a_fit ./ D_sym;  % n_sym x 1
    f_scan = coeffs.' * W_scan;  % 1 x n_xM, complex

    % All a_n share same phase, so factor it out and find zero of the real part
    % after phase rotation
    common_phase = angle(sum(a_fit));
    f_rotated = real(f_scan * exp(-1i * common_phase));

    % Find first zero crossing (lowest x_M)
    for ix = 1:(numel(xM_scan)-1)
        if f_rotated(ix) * f_rotated(ix+1) < 0
            t = f_rotated(ix) / (f_rotated(ix) - f_rotated(ix+1));
            xM_pred(ie) = xM_scan(ix) + t * (xM_scan(ix+1) - xM_scan(ix));
            break;
        end
    end
end

% x_M is already in nondimensional coords (= x_M / L_raft)
xM_pred_norm = xM_pred;

% ---- Plot ----
fig = figure('Color','w','Units','centimeters','Position',[2 2 24 16]);
ax = gca; hold(ax,'on');

contourf(ax, reshape([St.EI],n_mp,n_EI), ...
    reshape([St.motor_position],n_mp,n_EI)/L_raft, ...
    asymmetry, 50, 'LineStyle','none');
contour(ax, reshape([St.EI],n_mp,n_EI), ...
    reshape([St.motor_position],n_mp,n_EI)/L_raft, ...
    asymmetry, [0 0], 'LineColor','k','LineWidth',1.5);
caxis([-1 1]); colormap(ax, bwr_colormap()); colorbar;

plot(ax, curve_EI, curve_mp, 'ko', 'MarkerSize', 5, 'LineWidth', 1, ...
    'DisplayName', 'Data (lowest S\approx0)');

valid = ~isnan(xM_pred_norm);
plot(ax, EI_pred(valid), xM_pred_norm(valid), 'r-', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Prediction (%d sym modes, fitted a_n)', n_sym));

set(ax,'XScale','log');
xlabel('EI (N m^4)'); ylabel('x_M / L');
title('f(x_M, EI) = \Sigma a_n W_n(x_M) / D_n(EI) = 0');
legend('show','Location','northeast','FontSize',11);
set(ax,'FontSize',12);

outfile = fullfile(saveDir, 'predict_second_family.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[24 16],'PaperPosition',[0 0 24 16]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);

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
