function analyze_extract_H_and_predict_second_family_v3(saveDir)
% Correct prediction of second family: condition is Re(S_far*conj(A_far))=0
%
% CORRECTION vs v1/v2: the asymmetry zero |eta_1|=|eta_end| is equivalent to
%   Re(S_far * conj(A_far)) = 0,   where S=(eta_end+eta_1)/2, A=(eta_end-eta_1)/2
% NOT to S_far=0 as previously assumed.
%
% So we need radiation coefficients for BOTH:
%   S_far = a_S^T * q_sym    (symmetric modal amplitudes)
%   A_far = a_A^T * q_ant    (antisymmetric modal amplitudes)
% And predict where Re(a_S^T*q_sym * conj(a_A^T*q_ant)) = 0.

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

% Extract lowest S~0 curve (ground truth)
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
EI_grid  = logspace(log10(min(EI_list)), log10(max(EI_list)), 5);
xM_grid  = [0.06, 0.16, 0.27, 0.38, 0.46];  % xM/L, off-curve

[EI_mesh, xM_mesh] = meshgrid(EI_grid, xM_grid);
sEI = EI_mesh(:).';   % 1 x 25
sMP = xM_mesh(:).';   % 1 x 25

n_total = numel(sEI);
fprintf('Sampling: 5 EI x 5 xM = %d runs\n', n_total);

%% ---- Run solver ----
n_modes = 8;
all_q    = zeros(n_modes, n_total);
all_Q    = zeros(n_modes, n_total);
all_Sfar = zeros(n_total, 1);
all_Afar = zeros(n_total, 1);   % NEW: antisymmetric far-field amplitude

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
    all_Afar(ip) = (eta(end) - eta(1)) / 2;   % NEW

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
fprintf('\nAntisymmetric modes:');
for j = ant_idx, fprintf(' %d(%s)', j-1, mode_types{j}); end
fprintf('\n');

%% ---- Extract H ----
H = all_Q / all_q;
resid_mat = all_Q - H * all_q;
rel_resid  = norm(resid_mat,'fro') / norm(all_Q,'fro');
fprintf('\n=== H extraction ===\n');
fprintf('Relative residual = %.3e\n', rel_resid);

H_ss = H(sym_idx, sym_idx);
H_aa = H(ant_idx, ant_idx);
H_sa = H(sym_idx, ant_idx);
H_as = H(ant_idx, sym_idx);
fprintf('Parity: |H_ss|=%.3e  |H_aa|=%.3e  |H_sa|=%.3e  |H_as|=%.3e\n', ...
    norm(H_ss,'fro'), norm(H_aa,'fro'), norm(H_sa,'fro'), norm(H_as,'fro'));

%% ---- Fit radiation coefficients ----
% S_far = a_S^T * q_sym
Q_sym = all_q(sym_idx, :).';   % n_total x n_sym
a_S = Q_sym \ all_Sfar;

Sfar_pred = Q_sym * a_S;
resid_aS = mean(abs(all_Sfar - Sfar_pred) ./ (abs(all_Sfar) + eps));
fprintf('\n=== Symmetric radiation coefficients (S_far = a_S^T * q_sym) ===\n');
fprintf('Fit residual: %.2e\n', resid_aS);
for jj = 1:numel(sym_idx)
    j = sym_idx(jj);
    fprintf('  a_S_%d (%s): %+.4e%+.4ei  (|a|=%.4e)\n', ...
        j-1, mode_types{j}, real(a_S(jj)), imag(a_S(jj)), abs(a_S(jj)));
end

% A_far = a_A^T * q_ant
Q_ant = all_q(ant_idx, :).';   % n_total x n_ant
a_A = Q_ant \ all_Afar;

Afar_pred = Q_ant * a_A;
resid_aA = mean(abs(all_Afar - Afar_pred) ./ (abs(all_Afar) + eps));
fprintf('\n=== Antisymmetric radiation coefficients (A_far = a_A^T * q_ant) ===\n');
fprintf('Fit residual: %.2e\n', resid_aA);
for jj = 1:numel(ant_idx)
    j = ant_idx(jj);
    fprintf('  a_A_%d (%s): %+.4e%+.4ei  (|a|=%.4e)\n', ...
        j-1, mode_types{j}, real(a_A(jj)), imag(a_A(jj)), abs(a_A(jj)));
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

%% ---- Predict curve: zeros of Re(S_far * conj(A_far)) ----
beta4   = beta.^4;
K_const = -base.rho_raft * base.omega^2;

EI_pred = logspace(log10(min(EI_list)), log10(max(EI_list)), 600);
xM_pred = NaN(size(EI_pred));

% Diagnostic: also save S_far and A_far scan
diag_EI_vals = EI_pred(round(linspace(1, numel(EI_pred), 7)));
diag_rows = [];

for ie = 1:numel(EI_pred)
    EI_val = EI_pred(ie);
    K_diag = EI_val * beta4 + K_const;
    A_mat  = diag(K_diag) - H;

    if rcond(A_mat) < 1e-12, continue; end
    try
        q_scan = -A_mat \ F_scan;   % n_modes x n_scan
    catch
        continue;
    end

    Sfar_scan = a_S.' * q_scan(sym_idx, :);   % 1 x n_scan
    Afar_scan = a_A.' * q_scan(ant_idx, :);   % 1 x n_scan

    % Condition: Re(S_far * conj(A_far)) = 0
    cross_re = real(Sfar_scan .* conj(Afar_scan));   % 1 x n_scan

    % Also check |S| < |A| (second family qualifier)
    is_second_fam = abs(Sfar_scan) < abs(Afar_scan);

    % Find sign changes in cross_re where second family qualifier holds
    found = false;
    for ix = 1:(n_scan-1)
        if cross_re(ix) * cross_re(ix+1) < 0
            % Check second-family qualifier near the crossing
            if is_second_fam(ix) || is_second_fam(ix+1)
                t = cross_re(ix) / (cross_re(ix) - cross_re(ix+1));
                xM_pred(ie) = (xM_scan_m(ix) + t*(xM_scan_m(ix+1)-xM_scan_m(ix))) / L_raft;
                found = true;
                break;
            end
        end
    end

    % Diagnostic data
    if any(EI_val == diag_EI_vals)
        for ix = 1:n_scan
            diag_rows(end+1,:) = [EI_val, xM_scan_m(ix)/L_raft, ...
                abs(Sfar_scan(ix)), abs(Afar_scan(ix)), cross_re(ix), ...
                real(Sfar_scan(ix)), imag(Sfar_scan(ix)), ...
                real(Afar_scan(ix)), imag(Afar_scan(ix))]; %#ok<AGROW>
        end
    end
end

%% ---- Save CSVs ----
T_data = table(curve_EI(:), curve_mp(:), 'VariableNames', {'EI', 'xM_L'});
writetable(T_data, fullfile(saveDir, 'Hv3_pred_data.csv'));

valid = ~isnan(xM_pred);
T_pred = table(EI_pred(valid).', xM_pred(valid).', 'VariableNames', {'EI', 'xM_L'});
writetable(T_pred, fullfile(saveDir, 'Hv3_pred_curve.csv'));

T_diag = array2table(diag_rows, 'VariableNames', ...
    {'EI','xM_L','Sfar_abs','Afar_abs','cross_re','Sfar_re','Sfar_im','Afar_re','Afar_im'});
writetable(T_diag, fullfile(saveDir, 'Hv3_SA_scan.csv'));

fprintf('\n=== Prediction summary ===\n');
fprintf('Data points: %d\n', numel(curve_EI));
fprintf('Predicted: %d / %d EI values\n', sum(valid), numel(EI_pred));
if sum(valid) > 0
    fprintf('xM/L range predicted: [%.3f, %.3f]\n', min(xM_pred(valid)), max(xM_pred(valid)));
end

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

fprintf('\nCSVs saved to %s/Hv3_*\n', saveDir);

end
