function test_second_family(saveDir)
% Test the S*A=0 prediction for asymmetry-factor zeros.
%
% Theory: alpha=0  <=>  S*A = 0, where
%   S = sum over symmetric modes of  W_n(x_M) * W_n(edge) / D_n(omega)
%   A = sum over antisymmetric modes of  W_n(x_M) * W_n(edge) / D_n(omega)
%
% First family (vertical lines): D_n -> 0 (resonance, independent of x_M).
% Second family: S=0 or A=0 away from resonance.
%
% This version evaluates the EXACT mode shapes (no Taylor expansion)
% and finds zeros of S and A as functions of x_M for each omega.

if nargin < 1, saveDir = 'data'; end
addpath('../src');

% ---- Load sweep data ----
Dat = load(fullfile(saveDir, 'sweepOmegaMotorPosition.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

omega_list = unique([St.omega]);
mp_list    = unique([St.motor_position]);
n_omega    = numel(omega_list);
n_mp       = numel(mp_list);

eta_1_sq   = abs(reshape([St.eta_1],   n_omega, n_mp)).^2;
eta_end_sq = abs(reshape([St.eta_end], n_omega, n_mp)).^2;
asymmetry  = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

O_grid_hz = reshape([St.omega], n_omega, n_mp) / (2*pi);
MP_grid   = reshape([St.motor_position], n_omega, n_mp);

if isfield(St(1), 'args') && isfield(St(1).args, 'L_raft')
    L_raft = St(1).args.L_raft;
else
    L_raft = 0.05;
end
MP_grid_norm = MP_grid / L_raft;

% ---- Physical parameters (from sweep_omega_motorPosition.m) ----
EI       = 3.0e9 * 3e-2 * (9.9e-4)^3 / 12;
rho_R    = 0.052;       % kg/m
rho_w    = 1000;         % kg/m^3
d        = 0.03;         % m
g        = 9.81;         % m/s^2
sigma    = 72.2e-3;      % N/m
nu       = 0;            % inviscid

% ---- Build mode shape bank ----
% Free-free beam: mode 0 = rigid translation (symmetric)
%                 mode 1 = rigid rotation (antisymmetric)
%                 elastic mode j: symmetric if j odd, antisymmetric if j even
%                 (in the freefree_betaL_roots indexing)
n_elastic = 6;
betaL_el = freefree_betaL_roots(n_elastic);

% Evaluate all mode shapes on a fine x_M grid (in [-L/2, L/2] coordinates)
% and at the right edge x = +L/2.
n_xM = 500;
xM_grid = linspace(0, L_raft/2 * 0.96, n_xM);  % motor positions from center to near-edge

% xi coordinates: xi = x + L/2, so xi in [0, L]
% center: xi = L/2.  right edge: xi = L.  x_M in [-L/2, L/2] -> xi_M = x_M + L/2

xi_edge = L_raft;  % right edge in xi coordinates

% Total modes: 2 rigid + n_elastic
n_total = 2 + n_elastic;
W_at_xM   = zeros(n_total, n_xM);   % W_n(x_M) for each mode and x_M
W_at_edge  = zeros(n_total, 1);      % W_n(+L/2) for each mode
beta_all   = zeros(n_total, 1);
is_symmetric = false(n_total, 1);

% Mode 0: rigid translation (symmetric)
% Normalized: W_0 = 1/sqrt(L)
W_at_xM(1, :)  = 1 / sqrt(L_raft);
W_at_edge(1)    = 1 / sqrt(L_raft);
beta_all(1)     = 0;
is_symmetric(1) = true;

% Mode 1: rigid rotation (antisymmetric)
% W_1 = x / norm, where x is measured from center
% norm = sqrt(integral of x^2 from -L/2 to L/2) = L^(3/2) / (2*sqrt(3))
rot_norm = L_raft^(3/2) / (2*sqrt(3));
W_at_xM(2, :)  = xM_grid / rot_norm;
W_at_edge(2)    = (L_raft/2) / rot_norm;
beta_all(2)     = 0;
is_symmetric(2) = false;

% Elastic modes
for j = 1:n_elastic
    bL = betaL_el(j);
    beta_n = bL / L_raft;
    beta_all(2+j) = beta_n;

    % Symmetric if j is odd (1st, 3rd, 5th elastic modes)
    is_symmetric(2+j) = (mod(j, 2) == 1);

    % Evaluate on fine grid for normalization
    xi_norm_grid = linspace(0, L_raft, 2000)';
    psi_norm = freefree_mode_shape(xi_norm_grid, L_raft, bL);
    L2_norm = sqrt(trapz(xi_norm_grid, psi_norm.^2));

    % At x_M positions (xi_M = x_M + L/2)
    xi_M = xM_grid + L_raft/2;
    W_at_xM(2+j, :) = freefree_mode_shape(xi_M(:), L_raft, bL)' / L2_norm;

    % At right edge (xi = L)
    W_at_edge(2+j) = freefree_mode_shape(xi_edge, L_raft, bL) / L2_norm;
end

% ---- For each omega, compute S(x_M) and A(x_M), find zeros ----
omega_fine = 2*pi * linspace(4, 100, 400);
freq_fine  = omega_fine / (2*pi);

% Storage for zero contours
S_zeros = NaN(numel(omega_fine), 10);  % up to 10 zeros per frequency
A_zeros = NaN(numel(omega_fine), 10);

for iw = 1:numel(omega_fine)
    omega = omega_fine(iw);

    % Domain depth and wavenumber (same logic as solver)
    H = 2.5 * g / omega^2;
    k = real(dispersion_k(omega, g, H, nu, sigma, rho_w));

    % Added mass (same for all modes in this approximation)
    m_a = d * rho_w / (k * tanh(k * H));

    % Modal denominators D_n
    D = EI * beta_all.^4 + d * rho_w * g - omega^2 * (rho_R + m_a);

    % Build S(x_M) and A(x_M)
    %   S = sum over symmetric modes:    W_n(x_M) * W_n(edge) / D_n
    %   A = sum over antisymmetric modes: W_n(x_M) * W_n(edge) / D_n
    coeffs_S = W_at_edge(is_symmetric)  ./ D(is_symmetric);
    coeffs_A = W_at_edge(~is_symmetric) ./ D(~is_symmetric);

    S_of_xM = coeffs_S' * W_at_xM(is_symmetric, :);   % 1 x n_xM
    A_of_xM = coeffs_A' * W_at_xM(~is_symmetric, :);  % 1 x n_xM

    % Find zero crossings of S
    s_count = 0;
    for ix = 1:(n_xM-1)
        if S_of_xM(ix) * S_of_xM(ix+1) < 0
            % Linear interpolation for zero
            t = S_of_xM(ix) / (S_of_xM(ix) - S_of_xM(ix+1));
            xM_zero = xM_grid(ix) + t * (xM_grid(ix+1) - xM_grid(ix));
            s_count = s_count + 1;
            if s_count <= size(S_zeros, 2)
                S_zeros(iw, s_count) = xM_zero;
            end
        end
    end

    % Find zero crossings of A
    a_count = 0;
    for ix = 1:(n_xM-1)
        if A_of_xM(ix) * A_of_xM(ix+1) < 0
            t = A_of_xM(ix) / (A_of_xM(ix) - A_of_xM(ix+1));
            xM_zero = xM_grid(ix) + t * (xM_grid(ix+1) - xM_grid(ix));
            a_count = a_count + 1;
            if a_count <= size(A_zeros, 2)
                A_zeros(iw, a_count) = xM_zero;
            end
        end
    end
end

% ---- Plot ----
figure('Color','w','Units','centimeters','Position',[2 2 24 18]);
ax = gca; hold(ax, 'on');

contourf(ax, O_grid_hz, MP_grid_norm, asymmetry, 50, 'LineStyle', 'none');
caxis(ax, [-1 1]);
colormap(ax, bwr_colormap());
cb = colorbar(ax);
cb.Label.String = 'Asymmetry Factor';
cb.Label.FontSize = 16;

% Data zero contour
contour(ax, O_grid_hz, MP_grid_norm, asymmetry, [0 0], ...
    'LineColor', 'k', 'LineWidth', 2, 'DisplayName', '\alpha=0 (data)');

% S=0 predictions (red)
for col = 1:size(S_zeros, 2)
    vals = S_zeros(:, col) / L_raft;
    valid = ~isnan(vals);
    if any(valid)
        lbl = ''; if col == 1, lbl = 'S=0 (exact)'; end
        plot(ax, freq_fine(valid), vals(valid), 'r.', 'MarkerSize', 6, ...
            'DisplayName', lbl);
    end
end

% A=0 predictions (green)
for col = 1:size(A_zeros, 2)
    vals = A_zeros(:, col) / L_raft;
    valid = ~isnan(vals);
    if any(valid)
        lbl = ''; if col == 1, lbl = 'A=0 (exact)'; end
        plot(ax, freq_fine(valid), vals(valid), 'g.', 'MarkerSize', 6, ...
            'DisplayName', lbl);
    end
end

xlabel(ax, 'Frequency (Hz)', 'FontSize', 16);
ylabel(ax, 'Motor Position / Raft Length', 'FontSize', 16);
title(ax, 'Asymmetry zeros: data vs modal prediction (S\cdotA = 0)');
legend(ax, 'show', 'Location', 'northwest', 'FontSize', 11);
set(ax, 'FontSize', 14, 'LineWidth', 0.75, 'TickDir', 'out', 'Box', 'on');
grid(ax, 'on');
xlim(ax, [4 100]); ylim(ax, [0 0.5]);

% Save
outfile = fullfile(saveDir, 'second_family_test.pdf');
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [24 18], ...
    'PaperPosition', [0 0 24 18]);
print(gcf, outfile, '-dpdf', '-painters', '-r300');
fprintf('Saved to %s\n', outfile);

end

% ============ Helpers ============

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

function cmap = bwr_colormap(n_colors, gamma)
    if nargin < 1 || isempty(n_colors), n_colors = 256; end
    if nargin < 2 || isempty(gamma), gamma = 1.3; end
    rgb_anchors = [0.99 0.35 0.00; 1 1 1; 0 0.35 0.80];
    data_points = [-1 0 1];
    lab_anchors = rgb2lab(rgb_anchors);
    t_lin = linspace(-1,1,n_colors).';
    a = abs(t_lin);
    t_nl = sign(t_lin) .* (a.^gamma);
    lab_interp = interp1(data_points.', lab_anchors, t_nl, 'linear');
    cmap = lab2rgb(lab_interp);
    cmap = max(min(cmap,1),0);
end
