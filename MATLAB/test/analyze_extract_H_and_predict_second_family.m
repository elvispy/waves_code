function analyze_extract_H_and_predict_second_family(saveDir)
% Extract hydrodynamic coupling matrix H by least squares, then predict
% the second-family curve via S_far(x_M,EI) = a^T*(K(EI)-H)^{-1}*F(x_M) = 0.
%
% THEORY:
%   Modal balance:   K_n * q_n = Q_n - F_n  (from decompose_raft_freefree_modes)
%   => Q_n = K_n * q_n + F_n  where K_n = EI*beta_n^4 - rho_raft*omega^2
%   Since Q_n = sum_m H_nm * q_m  =>  H * q = Q  (per solver run)
%
% EXTRACTION:
%   Stacking N solver runs:   H * [q^1 ... q^N] = [Q^1 ... Q^N]
%   Least-squares:            H = Q_mat / q_mat  (25 runs, 8 modes)
%
% PREDICTION:
%   (K(EI) - H) * q = -F(x_M)  =>  S_far = a^T * q_sym = 0

if nargin < 1, saveDir = 'data'; end
addpath('../src');

%% ---- Load sweep data ----
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

%% ---- Extract lowest S~0 curve ----
curve_EI = [];
curve_mp = [];
for ie = 1:n_EI
    col = asymmetry(:, ie);
    cr_mp = []; cr_SA = [];
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
% Remove known outlier
bad = abs(curve_mp - 0.3634) < 0.01;
curve_EI(bad) = []; curve_mp(bad) = [];
fprintf('Curve: %d points, EI in [%.2e, %.2e], x_M/L in [%.3f, %.3f]\n', ...
    numel(curve_EI), min(curve_EI), max(curve_EI), min(curve_mp), max(curve_mp));

%% ---- Choose solver sample points: 20 on-curve + 5 off-curve ----
n_on  = 20;
n_off = 5;
n_total = n_on + n_off;

idx_on = round(linspace(1, numel(curve_EI), n_on));
on_EI  = curve_EI(idx_on);
on_MP  = curve_mp(idx_on);

% Off-curve: same EI, x_M shifted +0.10 above the curve (for conditioning)
idx_off = round(linspace(3, numel(curve_EI)-2, n_off));
off_EI  = curve_EI(idx_off);
off_MP  = min(curve_mp(idx_off) + 0.10, 0.48);

sEI = [on_EI, off_EI];
sMP = [on_MP, off_MP];

%% ---- Run solver at all sample points ----
n_modes = 8;
all_q    = zeros(n_modes, n_total);   % displacement modal amplitudes
all_Q    = zeros(n_modes, n_total);   % fluid pressure projections: Q = H*q
all_Sfar = zeros(n_total, 1);

fprintf('Running %d solver cases (%d on-curve, %d off-curve)...\n', n_total, n_on, n_off);
for ip = 1:n_total
    [~, x, ~, ~, eta, args] = flexible_surferbot_v2( ...
        'sigma',   base.sigma,        'rho',    base.rho, ...
        'nu',      base.nu,           'g',      base.g, ...
        'L_raft',  base.L_raft,       'motor_position', sMP(ip)*L_raft, ...
        'd',       base.d,            'EI',     sEI(ip), ...
        'rho_raft',base.rho_raft,     'domainDepth', base.domainDepth, ...
        'L_domain',base.L_domain,     'motor_inertia', base.motor_inertia, ...
        'BC',      base.BC,           'omega',  base.omega);

    Srun  = struct('x', x, 'eta', eta, 'args', args);
    modal = decompose_raft_freefree_modes(Srun, 'num_modes', n_modes, 'verbose', false);

    all_q(:,ip)  = modal.q;
    all_Q(:,ip)  = modal.Q;    % Q_n = sum_m H_nm*q_m  (from pressure projection)
    all_Sfar(ip) = (eta(end) + eta(1)) / 2;

    if ip == 1
        beta       = modal.beta;        % dimensional wavenumber [1/m]
        mode_types = modal.mode_type;
        Psi_left   = modal.Psi(1,:);
        Psi_right  = modal.Psi(end,:);
        is_sym     = abs(Psi_left+Psi_right) > abs(Psi_left-Psi_right);
        x_raft_ref = modal.x_raft;      % physical coords [m], range [-L/2, L/2]
        Psi_ref    = modal.Psi;         % weighted-orthonormal mode shapes on raft
    end
end

sym_idx = find(is_sym);   % indices of symmetric modes (0,2,4,6)
ant_idx = find(~is_sym);  % indices of antisymmetric modes (1,3,5,7)
fprintf('Symmetric modes:');
for j = sym_idx, fprintf(' %d(%s)', j-1, mode_types{j}); end
fprintf('\n');

%% ---- Extract H by least squares: H * all_q = all_Q ----
% MATLAB B/A solves X*A = B in least-squares sense, i.e., X = B * pinv(A)
H = all_Q / all_q;   % n_modes × n_modes

resid_mat = all_Q - H * all_q;
rel_resid  = norm(resid_mat, 'fro') / norm(all_Q, 'fro');
fprintf('\n=== H extraction ===\n');
fprintf('Relative residual ||Q - H*q||_F / ||Q||_F = %.3e\n', rel_resid);
if rel_resid > 0.05
    warning('H residual > 5%%. H may depend on (EI,xM): model assumption violated.');
end

%% ---- Parity structure check ----
% If fluid respects beam symmetry, H should block-diagonalize: H_sa = H_as ~ 0
H_ss = H(sym_idx, sym_idx);
H_aa = H(ant_idx, ant_idx);
H_sa = H(sym_idx, ant_idx);
H_as = H(ant_idx, sym_idx);
fprintf('Parity check (Frobenius norms):\n');
fprintf('  |H_ss|=%.3e  |H_aa|=%.3e  |H_sa|=%.3e  |H_as|=%.3e\n', ...
    norm(H_ss,'fro'), norm(H_aa,'fro'), norm(H_sa,'fro'), norm(H_as,'fro'));

%% ---- EI-independence check: compare H from first vs second half ----
H_A = all_Q(:,1:12) / all_q(:,1:12);
H_B = all_Q(:,13:end) / all_q(:,13:end);
fprintf('EI-independence check: ||H_A - H_B||_F / ||H_A||_F = %.3e\n', ...
    norm(H_A-H_B,'fro') / norm(H_A,'fro'));

%% ---- Fit radiation coefficients from on-curve runs ----
% S_far = sum_{n sym} a_n * q_n   =>  Sfar_on = Q_sym_on * a
Q_sym_on = all_q(sym_idx, 1:n_on).';   % n_on × n_sym
Sfar_on  = all_Sfar(1:n_on);
a_fit = Q_sym_on \ Sfar_on;            % n_sym × 1, complex

Sfar_pred = Q_sym_on * a_fit;
resid_a = mean(abs(Sfar_on - Sfar_pred) ./ (abs(Sfar_on) + eps));
fprintf('\n=== Radiation coefficients ===\n');
fprintf('Fit residual (mean |err|/|S|): %.2e\n', resid_a);
for jj = 1:numel(sym_idx)
    j = sym_idx(jj);
    fprintf('  a_%d (%s): %+.4e%+.4ei  (|a|=%.4e)\n', ...
        j-1, mode_types{j}, real(a_fit(jj)), imag(a_fit(jj)), abs(a_fit(jj)));
end

%% ---- Predict the curve ----
% For each EI value, scan xM and find zero of S_far(xM, EI):
%   q = -(K(EI)-H)^{-1} * F_motor * W(xM)
%   S_far = a^T * q(sym_idx)
%   => S_far = -F_motor * a^T * [(K-H)^{-1}](sym_idx,:) * W(xM) = 0

beta4   = beta.^4;    % n_modes × 1
K_const = -base.rho_raft * base.omega^2;  % scalar EI-independent term

% Trapz weights on raft grid (needed for Gaussian projection)
x_raft_col = x_raft_ref(:);
n_raft = numel(x_raft_col);
dx_r = diff(x_raft_col);
w_trap = zeros(n_raft, 1);
w_trap(1) = dx_r(1)/2;
w_trap(end) = dx_r(end)/2;
if n_raft > 2
    w_trap(2:end-1) = (x_raft_col(3:end) - x_raft_col(1:end-2)) / 2;
end

% Gaussian forcing projection: F_n(xM) = I_motor*omega^2 * Psi_n projected onto gauss
sigma_f = 0.05 * L_raft;
F_motor = base.motor_inertia * base.omega^2;
xM_scan_m = linspace(0.005*L_raft, 0.499*L_raft, 800);
n_scan = numel(xM_scan_m);

F_scan = zeros(n_modes, n_scan);
for ix = 1:n_scan
    xM = xM_scan_m(ix);
    gauss_vals = exp(-(x_raft_col - xM).^2 / (2*sigma_f^2)) / (sigma_f * sqrt(2*pi));
    F_scan(:, ix) = F_motor * (Psi_ref.' * (gauss_vals .* w_trap));
end

EI_pred = logspace(log10(min(EI_list)), log10(max(EI_list)), 600);
xM_pred = NaN(size(EI_pred));

for ie = 1:numel(EI_pred)
    EI_val = EI_pred(ie);
    K_diag = EI_val * beta4 + K_const;
    A_mat  = diag(K_diag) - H;

    if rcond(A_mat) < 1e-12, continue; end

    try
        q_scan = -A_mat \ F_scan;   % n_modes × n_scan
    catch
        continue;
    end

    Sfar_scan = a_fit.' * q_scan(sym_idx, :);   % 1 × n_scan, complex

    % Find minimum of |S_far|; also look for sign change of real and imag parts
    Sfar_abs = abs(Sfar_scan);
    [~, imin] = min(Sfar_abs);

    % Accept minimum only if it's a genuine local minimum below 20% of range
    Sfar_range = max(Sfar_abs) - min(Sfar_abs);
    if Sfar_abs(imin) < 0.20 * max(Sfar_abs) && Sfar_range > 0
        % Refine with real-part sign change near the minimum
        window = max(1, imin-20):min(n_scan, imin+20);
        f_re = real(Sfar_scan(window));
        found = false;
        for ix = 1:(numel(window)-1)
            if f_re(ix) * f_re(ix+1) < 0
                t = f_re(ix) / (f_re(ix) - f_re(ix+1));
                ix_global = window(ix);
                xM_pred(ie) = (xM_scan_m(ix_global) + t*(xM_scan_m(ix_global+1)-xM_scan_m(ix_global))) / L_raft;
                found = true;
                break;
            end
        end
        if ~found
            % Fall back to minimum location
            xM_pred(ie) = xM_scan_m(imin) / L_raft;
        end
    end
end

%% ---- Save CSVs ----
T_data = table(curve_EI(:), curve_mp(:), 'VariableNames', {'EI', 'xM_L'});
writetable(T_data, fullfile(saveDir, 'H_pred_data.csv'));

valid = ~isnan(xM_pred);
T_pred = table(EI_pred(valid).', xM_pred(valid).', 'VariableNames', {'EI', 'xM_L'});
writetable(T_pred, fullfile(saveDir, 'H_pred_curve.csv'));

fprintf('\n=== Prediction summary ===\n');
fprintf('Data points:      %d\n', numel(curve_EI));
fprintf('EI values scanned: %d\n', numel(EI_pred));
fprintf('Predicted points: %d\n', sum(valid));
if sum(valid) > 0
    fprintf('Predicted xM/L range: [%.3f, %.3f]\n', min(xM_pred(valid)), max(xM_pred(valid)));
end
fprintf('CSVs: %s/H_pred_data.csv  (data)\n', saveDir);
fprintf('       %s/H_pred_curve.csv (prediction)\n', saveDir);

%% ---- Diagnostic: |S_far| vs xM at 5 representative EI values ----
EI_diag = EI_pred(round(linspace(1, numel(EI_pred), 5)));
diag_rows = [];
for ie_d = 1:numel(EI_diag)
    EI_val = EI_diag(ie_d);
    K_diag = EI_val * beta4 + K_const;
    A_mat  = diag(K_diag) - H;
    if rcond(A_mat) < 1e-12, continue; end
    q_s = -A_mat \ F_scan;
    Sfar_s = a_fit.' * q_s(sym_idx,:);
    for ix = 1:n_scan
        diag_rows(end+1,:) = [EI_val, xM_scan_m(ix)/L_raft, abs(Sfar_s(ix)), real(Sfar_s(ix)), imag(Sfar_s(ix))]; %#ok<AGROW>
    end
end
T_diag = array2table(diag_rows, 'VariableNames', {'EI','xM_L','Sfar_abs','Sfar_re','Sfar_im'});
writetable(T_diag, fullfile(saveDir, 'H_Sfar_scan.csv'));
fprintf('Diagnostic S_far scan saved to %s/H_Sfar_scan.csv\n', saveDir);

%% ---- Quick numeric comparison: data vs prediction at sampled EI values ----
fprintf('\n=== Data vs Prediction at on-curve EI values ===\n');
fprintf('  EI          | xM/L data | xM/L pred | diff\n');
for ii = 1:n_on
    EI_i = on_EI(ii);
    xM_i = on_MP(ii);
    [~, closest] = min(abs(EI_pred - EI_i));
    xM_p = xM_pred(closest);
    fprintf('  %10.3e | %9.4f | %9.4f | %+.4f\n', EI_i, xM_i, xM_p, xM_p - xM_i);
end

end
