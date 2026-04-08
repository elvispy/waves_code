function analyze_second_family(saveDir)
% Analyze the second family of asymmetry-factor zeros using S*A phase condition.
%
% Key insight: eta(left_edge) = S - A,  eta(right_edge) = S + A
% where S = symmetric contribution, A = antisymmetric contribution.
% So  S = (eta_end + eta_1)/2,   A = (eta_end - eta_1)/2
% (eta_1 = left edge, eta_end = right edge in the data convention)
%
% alpha=0  <=>  |eta_1|^2 = |eta_end|^2  <=>  Re(S*conj(A)) = 0

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

L_raft = St(1).args.L_raft;

% Reshape into grids
eta_1   = reshape([St.eta_1],   n_omega, n_mp);   % left edge (x = -L/2)
eta_end = reshape([St.eta_end], n_omega, n_mp);    % right edge (x = +L/2)

% Compute S and A from edge values
% Convention: eta(-L/2) = S - A,  eta(+L/2) = S + A
% eta_1 is labeled as left edge in the solver (x(1)), eta_end as right edge (x(end))
% But we need to check: in the solver, x goes from -L_domain/2 to +L_domain/2
% eta_1 = eta at x = -L_domain/2 (FAR left, outside raft)
% eta_end = eta at x = +L_domain/2 (FAR right, outside raft)
%
% Actually, eta_1 and eta_end are the DOMAIN edges, not the RAFT edges.
% The asymmetry of outgoing waves at the domain edges still reflects
% the raft's S/A decomposition since waves radiate outward.

S_grid = (eta_end + eta_1) / 2;
A_grid = (eta_end - eta_1) / 2;

% Asymmetry factor
eta_1_sq   = abs(eta_1).^2;
eta_end_sq = abs(eta_end).^2;
asymmetry  = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

O_hz    = omega_list / (2*pi);
MP_norm = mp_list / L_raft;

% Phase angle between S and A
SA_phase = rad2deg(angle(S_grid ./ A_grid));
ReSAstar = real(S_grid .* conj(A_grid));

% ---- Extract zero contour (second family) ----
% contourc wants Z(n_mp, n_omega) with X=O_hz(1:n_omega), Y=MP_norm(1:n_mp)
C = contourc(O_hz, MP_norm, asymmetry.', [0 0]);
curves = {};
idx = 1;
while idx < size(C, 2)
    npts = C(2, idx);
    cx = C(1, (idx+1):(idx+npts));
    cy = C(2, (idx+1):(idx+npts));
    curves{end+1} = struct('freq_hz', cx, 'mp_norm', cy); %#ok<AGROW>
    idx = idx + npts + 1;
end

fprintf('Found %d zero-contour segments\n', numel(curves));
for ic = 1:numel(curves)
    freq_span = max(curves{ic}.freq_hz) - min(curves{ic}.freq_hz);
    mp_span   = max(curves{ic}.mp_norm) - min(curves{ic}.mp_norm);
    fprintf('  Curve %d: %d pts, freq [%.1f, %.1f] Hz, mp [%.3f, %.3f], spans: f=%.1f mp=%.3f\n', ...
        ic, numel(curves{ic}.freq_hz), ...
        min(curves{ic}.freq_hz), max(curves{ic}.freq_hz), ...
        min(curves{ic}.mp_norm), max(curves{ic}.mp_norm), freq_span, mp_span);
end

% ---- Plot 1: Phase space with angle(S/A) overlay ----
fig1 = figure('Color','w','Units','centimeters','Position',[2 2 28 22]);

% Panel 1: Asymmetry factor with zero contour
subplot(2,2,1); hold on;
contourf(O_hz, MP_norm, asymmetry.', 50, 'LineStyle', 'none');
contour(O_hz, MP_norm, asymmetry.', [0 0], 'LineColor', 'k', 'LineWidth', 2);
caxis([-1 1]); colormap(gca, bwr_colormap()); colorbar;
xlabel('Frequency (Hz)'); ylabel('x_M / L');
title('Asymmetry factor \alpha'); set(gca, 'FontSize', 11);

% Panel 2: angle(S/A) — should be ±90° on the second family
subplot(2,2,2); hold on;
contourf(O_hz, MP_norm, SA_phase.', 50, 'LineStyle', 'none');
contour(O_hz, MP_norm, asymmetry.', [0 0], 'LineColor', 'k', 'LineWidth', 2);
contour(O_hz, MP_norm, SA_phase.', [90 90], 'LineColor', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
contour(O_hz, MP_norm, SA_phase.', [-90 -90], 'LineColor', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
caxis([-180 180]); colorbar;
xlabel('Frequency (Hz)'); ylabel('x_M / L');
title('angle(S/A) [deg]  — should be \pm90° on \alpha=0'); set(gca, 'FontSize', 11);

% Panel 3: Re(S*A*) — should be zero on the contour
subplot(2,2,3); hold on;
% Normalize for visualization
ReSA_norm = ReSAstar ./ max(abs(ReSAstar(:)));
contourf(O_hz, MP_norm, ReSA_norm.', 50, 'LineStyle', 'none');
contour(O_hz, MP_norm, asymmetry.', [0 0], 'LineColor', 'k', 'LineWidth', 2);
contour(O_hz, MP_norm, ReSA_norm.', [0 0], 'LineColor', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
caxis([-1 1]); colormap(gca, bwr_colormap()); colorbar;
xlabel('Frequency (Hz)'); ylabel('x_M / L');
title('Re(S \cdot A^*) normalized — zero = \alpha=0'); set(gca, 'FontSize', 11);

% Panel 4: |S| vs |A| ratio
subplot(2,2,4); hold on;
SA_ratio = abs(S_grid) ./ (abs(A_grid) + eps);
contourf(O_hz, MP_norm, log10(SA_ratio).', 50, 'LineStyle', 'none');
contour(O_hz, MP_norm, asymmetry.', [0 0], 'LineColor', 'k', 'LineWidth', 2);
caxis([-2 2]); colorbar;
xlabel('Frequency (Hz)'); ylabel('x_M / L');
title('log_{10}(|S|/|A|) — magnitude ratio'); set(gca, 'FontSize', 11);

sgtitle('Second family: S \cdot A phase analysis', 'FontSize', 14, 'FontWeight', 'bold');

outfile = fullfile(saveDir, 'second_family_analysis.pdf');
set(fig1, 'PaperUnits', 'centimeters', 'PaperSize', [28 22], 'PaperPosition', [0 0 28 22]);
print(fig1, outfile, '-dpdf', '-painters', '-r300');
fprintf('\nSaved to %s\n', outfile);

% ---- Also print a few values along the second family ----
% Interpolate S, A, and angle along the second family curve
% Pick the longest non-vertical curve
best = 0; best_idx = 0;
for ic = 1:numel(curves)
    score = (max(curves{ic}.freq_hz) - min(curves{ic}.freq_hz)) * ...
            (max(curves{ic}.mp_norm) - min(curves{ic}.mp_norm));
    if score > best, best = score; best_idx = ic; end
end

if best_idx > 0
    c2 = curves{best_idx};
    n_pts = numel(c2.freq_hz);
    sample_idx = round(linspace(1, n_pts, min(12, n_pts)));

    fprintf('\n=== Values along the second family (curve %d) ===\n', best_idx);
    fprintf('%-8s | %-8s | %-12s | %-12s | %-12s | %-12s\n', ...
        'f (Hz)', 'x_M/L', '|S|', '|A|', 'angle(S/A)', 'Re(SA*)');
    fprintf('%s\n', repmat('-', 1, 75));

    for is = sample_idx
        f_val  = c2.freq_hz(is);
        mp_val = c2.mp_norm(is);

        % Interpolate from grid
        S_val = interp2(O_hz, MP_norm, S_grid.', f_val, mp_val);
        A_val = interp2(O_hz, MP_norm, A_grid.', f_val, mp_val);

        fprintf('%7.1f  | %7.4f  | %10.3e   | %10.3e   | %+10.1f°  | %+10.3e\n', ...
            f_val, mp_val, abs(S_val), abs(A_val), ...
            rad2deg(angle(S_val / A_val)), real(S_val * conj(A_val)));
    end
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
