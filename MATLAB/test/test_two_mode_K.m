function test_two_mode_K(saveDir)
% Test: does K = W_2(x_M) * D_0 / (W_0 * D_2(EI)) remain constant
% along the lowest second-family curve (S~0)?
%
% From the two-mode balance:
%   c_0 * W_0 / D_0  +  c_2 * W_2(x_M) / D_2(EI) = 0
%   => W_2(x_M) / D_2(EI) = -(c_0/c_2) * W_0 / D_0 = const
%   => K := W_2(x_M) / D_2(EI) should be constant along the curve.

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
sigma  = St(1).args.sigma;
nu     = St(1).args.nu;

fprintf('Parameters: L=%.4f, omega=%.1f, d=%.4f, rho_R=%.4f\n', L_raft, omega, d, rho_R);

% Wavenumber and added mass (fixed omega)
% Use actual domain depth from data (solver auto-selects it)
H = St(1).args.domainDepth;
k = real(St(1).args.k);
m_a = d * rho_w / (k * tanh(k * H));
fprintf('Using H=%.4f from data, k=%.4f from data\n', H, k);

fprintf('k = %.4f, H = %.4f, m_a = %.6f\n', k, H, m_a);

% Beam eigenvalues
betaL_roots = freefree_betaL_roots(6);
% First elastic symmetric mode is betaL_roots(1)
bL2 = betaL_roots(1);
beta2 = bL2 / L_raft;

fprintf('First elastic symmetric mode: betaL = %.4f, beta = %.2f\n', bL2, beta2);

% D_0 (rigid, independent of EI)
D_0 = d * rho_w * g - omega^2 * (rho_R + m_a);
fprintf('D_0 = %.6e\n', D_0);

% W_0 = 1/sqrt(L) (constant, normalized rigid mode)
W_0 = 1 / sqrt(L_raft);

% W_2(x_M): first elastic symmetric mode shape, L2-normalized on [0, L]
% Evaluated at x_M (measured from center, so xi = x_M + L/2)
xi_fine = linspace(0, L_raft, 5000)';
psi_raw = freefree_mode_shape(xi_fine, L_raft, bL2);
L2_norm = sqrt(trapz(xi_fine, psi_raw.^2));

% Function to evaluate W_2 at any x_M (from center)
W2_of_xM = @(xM) interp1(xi_fine, psi_raw/L2_norm, xM + L_raft/2, 'linear');

% ---- Extract the lowest S~0 curve from v3 data ----
% Re-extract zero crossings column by column
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

% For each EI column, find zero crossings and pick the lowest x_M one
curve_EI = [];
curve_mp = [];
curve_SA = [];

for ie = 1:n_EI
    col = asymmetry(:, ie);
    crossings_mp = [];
    crossings_SA = [];
    for im = 1:(n_mp-1)
        if col(im) * col(im+1) < 0
            t = col(im) / (col(im) - col(im+1));
            mp_zero = MP_norm_list(im) + t * (MP_norm_list(im+1) - MP_norm_list(im));
            sa_zero = SA_ratio(im, ie) + t * (SA_ratio(im+1, ie) - SA_ratio(im, ie));
            crossings_mp(end+1) = mp_zero; %#ok<AGROW>
            crossings_SA(end+1) = sa_zero; %#ok<AGROW>
        end
    end

    % Pick the lowest-x_M crossing that is S~0 (SA < 0)
    if ~isempty(crossings_mp)
        s0_mask = crossings_SA < 0;
        if any(s0_mask)
            candidates_mp = crossings_mp(s0_mask);
            candidates_SA = crossings_SA(s0_mask);
            [mp_pick, idx] = min(candidates_mp);
            curve_EI(end+1) = EI_list(ie); %#ok<AGROW>
            curve_mp(end+1) = mp_pick; %#ok<AGROW>
            curve_SA(end+1) = candidates_SA(idx); %#ok<AGROW>
        end
    end
end

fprintf('\nExtracted %d points on the lowest S~0 curve\n', numel(curve_EI));

% ---- Compute K = W_2(x_M) / D_2(EI) at each point ----
K_vals = zeros(size(curve_EI));
D2_vals = zeros(size(curve_EI));
W2_vals = zeros(size(curve_EI));

for ip = 1:numel(curve_EI)
    EI_val = curve_EI(ip);
    xM_val = curve_mp(ip) * L_raft;  % back to meters

    D2 = EI_val * beta2^4 + d * rho_w * g - omega^2 * (rho_R + m_a);
    W2 = W2_of_xM(xM_val);

    D2_vals(ip) = D2;
    W2_vals(ip) = W2;
    K_vals(ip) = W2 / D2;
end

% ---- Print results ----
fprintf('\n%-12s | %-8s | %-12s | %-12s | %-12s\n', ...
    'EI', 'x_M/L', 'W_2(x_M)', 'D_2(EI)', 'K=W_2/D_2');
fprintf('%s\n', repmat('-', 1, 65));
for ip = 1:numel(curve_EI)
    fprintf('%11.3e | %7.4f  | %+11.4e | %+11.4e | %+11.4e\n', ...
        curve_EI(ip), curve_mp(ip), W2_vals(ip), D2_vals(ip), K_vals(ip));
end

fprintf('\n=== K statistics ===\n');
fprintf('  Mean:  %+.4e\n', mean(K_vals));
fprintf('  Std:   %.4e\n', std(K_vals));
fprintf('  CoV:   %.1f%%\n', 100*std(K_vals)/abs(mean(K_vals)));
fprintf('  Min:   %+.4e\n', min(K_vals));
fprintf('  Max:   %+.4e\n', max(K_vals));
fprintf('  Range: %.1fx\n', max(abs(K_vals))/min(abs(K_vals)));

end

function betaL = freefree_betaL_roots(n)
    betaL = zeros(n, 1);
    f = @(y) cosh(y) .* cos(y) - 1;
    for k = 1:n
        a = k * pi;
        b = (k + 1) * pi;
        betaL(k) = fzero(f, [a, b]);
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
