function analyze_extract_H_and_predict_second_family_v2(saveDir)
% Improved H extraction using structured off-curve sampling.
%
% KEY CHANGE vs v1: all 25 extraction runs are at a 5-EI x 5-xM grid,
% NONE on the second-family curve. This eliminates the near-degeneracy
% in the direction a^T*q_sym that plagued v1 (where 20/25 runs were
% on-curve, all satisfying a^T*q_sym ≈ 0 by construction).
%
% H extraction: H * q_mat = Q_mat   (Q = modal pressure projection)
% Radiation fit: Sfar = a^T * q_sym  (from same 25 runs)
% Prediction: a^T * (K(EI)-H)^{-1} * F(xM) = 0

if nargin < 1, saveDir = 'data'; end
addpath('../src');

%% ---- Load sweep data (ground truth curve) ----
Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L_raft = St(1).args.L_raft;
base   = St(1).args;

mp_list = unique([St.motor_position]);
EI_list = unique([St.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);
MP_norm_list = mp_list / L_raft;

eta_1   = reshape([St.eta_1],   n_mp, n_EI);
eta_end = reshape([St.eta_end], n_mp, n_EI);
asymmetry = -(abs(eta_1).^2 - abs(eta_end).^2) ./ (abs(eta_1).^2 + abs(eta_end).^2);
SA_ratio  = log10(abs((eta_end+eta_1)/2) ./ (abs((eta_end-eta_1)/2) + eps));

% Extract lowest S~0 curve (ground truth for comparison)
curve_EI = []; curve_mp = [];
for ie = 1:n_EI
    col = asymmetry(:, ie); cr_mp = []; cr_SA = [];
    for im = 1:(n_mp-1)
        if col(im) * col(im+1) < 0
            t = col(im) / (col(im) - col(im+1));
            cr_mp(end+1) = MP_norm_list(im) + t*(MP_norm_list(im+1)-MP_norm_list(im)); %#ok<AGROW>
            sa_col = SA_ratio(:,ie);
            cr_SA(end+1) = sa_col(im) + t*(sa_col(im+1)-sa_col(im)); %#ok<AGROW>
        end
    end
    if ~isempty(cr_mp)
        s0 = cr_SA < 0;
        if any(s0)
            curve_EI(end+1) = EI_list(ie); %#ok<AGROW>
            curve_mp(end+1) = min(cr_mp(s0)); %#ok<AGROW>
        end
    end
end
bad = abs(curve_mp - 0.3634) < 0.01;
curve_EI(bad) = []; curve_mp(bad) = [];

%% ---- Structured 5x5 off-curve sampling ----
% 5 EI values logspaced across the sweep range
% 5 xM/L values uniformly spaced, none coinciding with the curve
EI_grid  = logspace(log10(min(EI_list)), log10(max(EI_list)), 5);
xM_grid  = [0.06, 0.16, 0.27, 0.38, 0.46];  % xM/L, diverse and off-curve

[EI_mesh, xM_mesh] = meshgrid(EI_grid, xM_grid);
sEI = EI_mesh(:).';   % 1 x 25
sMP = xM_mesh(:).';   % 1 x 25 (normalized x_M/L)

n_total = numel(sEI);
fprintf('Sampling: %d EI values x %d xM values = %d runs (structured off-curve)\n', ...
    numel(EI_grid), numel(xM_grid), n_total);

%% ---- Run solver ----
n_modes = 8;
all_q    = zeros(n_modes, n_total);
all_Q    = zeros(n_modes, n_total);
all_Sfar = zeros(n_total, 1);

fprintf('Running %d solver cases...\n', n_total);
for ip = 1:n_total
    [~, x, ~, ~, eta, args] = flexible_surferbot_v2( ...
        'sigma',    base.sigma,         'rho',    base.rho, ...
        'nu',       base.nu,            'g',      base.g, ...
        'L_raft',   base.L_raft,        'motor_position', sMP(ip)*L_raft, ...
        'd',        base.d,             'EI',     sEI(ip), ...
        'rho_raft', base.rho_raft,      'domainDepth', base.domainDepth, ...
        'L_domain', base.L_domain,      'motor_inertia', base.motor_inertia, ...
        'BC',       base.BC,            'omega',  base.omega);

    Srun  = struct('x', x, 'eta', eta, 'args', args);
    modal = decompose_raft_freefree_modes(Srun, 'num_modes', n_modes, 'verbose', false);

    all_q(:,ip)  = modal.q;
    all_Q(:,ip)  = modal.Q;
    all_Sfar(ip) = (eta(end) + eta(1)) / 2;

    if ip == 1
        beta       = modal.beta;
        mode_types = modal.mode_type;
        Psi_left   = modal.Psi(1,:);
        Psi_right  = modal.Psi(end,:);
        is_sym     = abs(Psi_left+Psi_right) > abs(Psi_left-Psi_right);
        x_raft_ref = modal.x_raft;
        Psi_ref    = modal.Psi;
    end
end

sym_idx = find(is_sym);
ant_idx = find(~is_sym);
fprintf('Symmetric modes:');
for j = sym_idx, fprintf(' %d(%s)', j-1, mode_types{j}); end
fprintf('\n');

%% ---- Extract H: H * all_q = all_Q ----
H = all_Q / all_q;   % n_modes x n_modes, least-squares

resid_mat = all_Q - H * all_q;
rel_resid  = norm(resid_mat,'fro') / norm(all_Q,'fro');
fprintf('\n=== H extraction ===\n');
fprintf('Relative residual = %.3e\n', rel_resid);

% Parity check
H_ss = H(sym_idx, sym_idx);
H_aa = H(ant_idx, ant_idx);
H_sa = H(sym_idx, ant_idx);
H_as = H(ant_idx, sym_idx);
fprintf('Parity: |H_ss|=%.3e  |H_aa|=%.3e  |H_sa|=%.3e  |H_as|=%.3e\n', ...
    norm(H_ss,'fro'), norm(H_aa,'fro'), norm(H_sa,'fro'), norm(H_as,'fro'));

% EI-independence: compare first 12 vs last 13 runs
H_A = all_Q(:,1:12) / all_q(:,1:12);
H_B = all_Q(:,13:end) / all_q(:,13:end);
fprintf('EI-independence: ||H_A-H_B||/||H_A|| = %.3e\n', norm(H_A-H_B,'fro')/norm(H_A,'fro'));

% Condition of q_mat (measures degeneracy)
sv = svd(all_q);
fprintf('q_mat condition number = %.3e  (singular values: %.2e ... %.2e)\n', ...
    sv(1)/sv(end), sv(1), sv(end));

%% ---- Fit radiation coefficients from same 25 runs ----
Q_sym = all_q(sym_idx, :).';   % n_total x n_sym
a_fit = Q_sym \ all_Sfar;

Sfar_pred = Q_sym * a_fit;
resid_a = mean(abs(all_Sfar - Sfar_pred) ./ (abs(all_Sfar) + eps));
fprintf('\n=== Radiation coefficients (from all 25 runs) ===\n');
fprintf('Fit residual: %.2e\n', resid_a);
for jj = 1:numel(sym_idx)
    j = sym_idx(jj);
    fprintf('  a_%d (%s): %+.4e%+.4ei  (|a|=%.4e)\n', ...
        j-1, mode_types{j}, real(a_fit(jj)), imag(a_fit(jj)), abs(a_fit(jj)));
end

%% ---- Gaussian forcing projection on raft grid ----
sigma_f = 0.05 * L_raft;
F_motor = base.motor_inertia * base.omega^2;

x_raft_col = x_raft_ref(:);
n_raft = numel(x_raft_col);
dx_r = diff(x_raft_col);
w_trap = zeros(n_raft, 1);
w_trap(1) = dx_r(1)/2; w_trap(end) = dx_r(end)/2;
if n_raft > 2
    w_trap(2:end-1) = (x_raft_col(3:end) - x_raft_col(1:end-2)) / 2;
end

xM_scan_m = linspace(0.005*L_raft, 0.499*L_raft, 800);
n_scan = numel(xM_scan_m);
F_scan = zeros(n_modes, n_scan);
for ix = 1:n_scan
    xM = xM_scan_m(ix);
    g_vals = exp(-(x_raft_col - xM).^2 / (2*sigma_f^2)) / (sigma_f * sqrt(2*pi));
    F_scan(:, ix) = F_motor * (Psi_ref.' * (g_vals .* w_trap));
end

%% ---- Predict curve ----
beta4   = beta.^4;
K_const = -base.rho_raft * base.omega^2;

EI_pred = logspace(log10(min(EI_list)), log10(max(EI_list)), 600);
xM_pred = NaN(size(EI_pred));

for ie = 1:numel(EI_pred)
    EI_val = EI_pred(ie);
    K_diag = EI_val * beta4 + K_const;
    A_mat  = diag(K_diag) - H;

    if rcond(A_mat) < 1e-12, continue; end
    try
        q_scan = -A_mat \ F_scan;
    catch
        continue;
    end

    Sfar_scan = a_fit.' * q_scan(sym_idx, :);   % 1 x n_scan
    Sfar_abs  = abs(Sfar_scan);
    [Sfar_min, imin] = min(Sfar_abs);

    % Accept if minimum is deep (< 15% of max)
    if Sfar_min < 0.15 * max(Sfar_abs)
        win = max(1,imin-30):min(n_scan,imin+30);
        f_re = real(Sfar_scan(win));
        found = false;
        for ix = 1:(numel(win)-1)
            if f_re(ix)*f_re(ix+1) < 0
                t = f_re(ix)/(f_re(ix)-f_re(ix+1));
                ig = win(ix);
                xM_pred(ie) = (xM_scan_m(ig) + t*(xM_scan_m(ig+1)-xM_scan_m(ig))) / L_raft;
                found = true; break;
            end
        end
        if ~found
            xM_pred(ie) = xM_scan_m(imin) / L_raft;
        end
    end
end

%% ---- Save CSVs ----
T_data = table(curve_EI(:), curve_mp(:), 'VariableNames', {'EI', 'xM_L'});
writetable(T_data, fullfile(saveDir, 'Hv2_pred_data.csv'));

valid = ~isnan(xM_pred);
T_pred = table(EI_pred(valid).', xM_pred(valid).', 'VariableNames', {'EI', 'xM_L'});
writetable(T_pred, fullfile(saveDir, 'Hv2_pred_curve.csv'));

fprintf('\n=== Prediction summary ===\n');
fprintf('Data points: %d\n', numel(curve_EI));
fprintf('Predicted: %d / %d EI values\n', sum(valid), numel(EI_pred));
if sum(valid) > 0
    fprintf('xM/L range predicted: [%.3f, %.3f]\n', min(xM_pred(valid)), max(xM_pred(valid)));
end

% Comparison at curve EI values
fprintf('\n=== Data vs Prediction ===\n');
fprintf('  EI          | xM/L data | xM/L pred |  diff\n');
for ii = 1:numel(curve_EI)
    [~,ic] = min(abs(EI_pred - curve_EI(ii)));
    xMp = xM_pred(ic);
    if isnan(xMp)
        fprintf('  %10.3e | %9.4f |       NaN |   NaN\n', curve_EI(ii), curve_mp(ii));
    else
        fprintf('  %10.3e | %9.4f | %9.4f | %+.4f\n', curve_EI(ii), curve_mp(ii), xMp, xMp-curve_mp(ii));
    end
end

% Sfar diagnostic at 5 EI values
EI_diag = EI_pred(round(linspace(1, numel(EI_pred), 5)));
diag_rows = [];
for ied = 1:numel(EI_diag)
    EI_val = EI_diag(ied);
    K_diag = EI_val * beta4 + K_const;
    A_mat  = diag(K_diag) - H;
    if rcond(A_mat) < 1e-12, continue; end
    q_s = -A_mat \ F_scan;
    Sf  = a_fit.' * q_s(sym_idx,:);
    for ix = 1:n_scan
        diag_rows(end+1,:) = [EI_val, xM_scan_m(ix)/L_raft, abs(Sf(ix)), real(Sf(ix)), imag(Sf(ix))]; %#ok<AGROW>
    end
end
T_diag = array2table(diag_rows, 'VariableNames', {'EI','xM_L','Sfar_abs','Sfar_re','Sfar_im'});
writetable(T_diag, fullfile(saveDir, 'Hv2_Sfar_scan.csv'));
fprintf('\nCSVs saved to %s/Hv2_pred_{data,curve}.csv, Hv2_Sfar_scan.csv\n', saveDir);

end
