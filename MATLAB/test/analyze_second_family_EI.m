function analyze_second_family_EI(saveDir)
% Analyze the second family in the (EI, motorPosition) plane.
% Uses d=0.03 dataset (sweepMotorPositionEI.mat).
%
% S = (eta_end + eta_1)/2,  A = (eta_end - eta_1)/2
% alpha=0  <=>  Re(S*conj(A)) = 0

if nargin < 1, saveDir = 'data'; end
addpath('../src');

% ---- Load sweep data (d=0.03) ----
Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L_raft = St(1).args.L_raft;
mp_list = unique([St.motor_position]);
EI_list = unique([St.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);

fprintf('Grid: %d motor positions x %d EI values\n', n_mp, n_EI);
fprintf('d = %g, omega = %g rad/s (%.1f Hz)\n', St(1).args.d, St(1).args.omega, St(1).args.omega/(2*pi));

% Reshape into grids (S(ip,ie): mp is rows, EI is cols)
EI_grid = reshape([St.EI], n_mp, n_EI);
MP_grid = reshape([St.motor_position], n_mp, n_EI) / L_raft;

eta_1   = reshape([St.eta_1],   n_mp, n_EI);   % left domain edge
eta_end = reshape([St.eta_end], n_mp, n_EI);    % right domain edge

% S and A from domain edges
S_grid = (eta_end + eta_1) / 2;
A_grid = (eta_end - eta_1) / 2;

% Asymmetry factor
eta_1_sq   = abs(eta_1).^2;
eta_end_sq = abs(eta_end).^2;
asymmetry  = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

% Derived quantities
SA_phase   = rad2deg(angle(S_grid ./ A_grid));
ReSAstar   = real(S_grid .* conj(A_grid));
SA_mag_ratio = abs(S_grid) ./ (abs(A_grid) + eps);

% Normalize Re(SA*) for visualization
ReSA_norm = ReSAstar ./ max(abs(ReSAstar(:)));

% ---- Contourc needs Z(n_EI, n_mp) with X=EI_list, Y=mp_list/L_raft ----
MP_norm_list = mp_list / L_raft;

% ---- Plot: 4-panel diagnostic ----
fig = figure('Color','w','Units','centimeters','Position',[1 1 32 24]);

% Panel 1: Asymmetry factor
subplot(2,2,1); ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, asymmetry, 50, 'LineStyle','none');
contour(ax, EI_grid, MP_grid, asymmetry, [0 0], 'LineColor','k','LineWidth',1.5);
set(ax,'XScale','log');
caxis([-1 1]); colormap(ax, bwr_colormap()); colorbar;
xlabel('EI (N m^4)'); ylabel('x_M / L');
title('\alpha (asymmetry factor)'); set(ax,'FontSize',11);

% Panel 2: angle(S/A)
subplot(2,2,2); ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, SA_phase, 50, 'LineStyle','none');
contour(ax, EI_grid, MP_grid, asymmetry, [0 0], 'LineColor','k','LineWidth',1.5);
contour(ax, EI_grid, MP_grid, SA_phase, [90 90], 'LineColor','r','LineWidth',1.5,'LineStyle','--');
contour(ax, EI_grid, MP_grid, SA_phase, [-90 -90], 'LineColor','r','LineWidth',1.5,'LineStyle','--');
set(ax,'XScale','log');
caxis([-180 180]); colorbar;
xlabel('EI (N m^4)'); ylabel('x_M / L');
title('angle(S/A) [deg]'); set(ax,'FontSize',11);

% Panel 3: Re(S*A*) normalized
subplot(2,2,3); ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, ReSA_norm, 50, 'LineStyle','none');
contour(ax, EI_grid, MP_grid, asymmetry, [0 0], 'LineColor','k','LineWidth',1.5);
contour(ax, EI_grid, MP_grid, ReSA_norm, [0 0], 'LineColor','r','LineWidth',1.5,'LineStyle','--');
set(ax,'XScale','log');
caxis([-1 1]); colormap(ax, bwr_colormap()); colorbar;
xlabel('EI (N m^4)'); ylabel('x_M / L');
title('Re(S \cdot A^*) normalized'); set(ax,'FontSize',11);

% Panel 4: log10(|S|/|A|)
subplot(2,2,4); ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, log10(SA_mag_ratio), 50, 'LineStyle','none');
contour(ax, EI_grid, MP_grid, asymmetry, [0 0], 'LineColor','k','LineWidth',1.5);
set(ax,'XScale','log');
caxis([-2 2]); colorbar;
xlabel('EI (N m^4)'); ylabel('x_M / L');
title('log_{10}(|S|/|A|)'); set(ax,'FontSize',11);

sgtitle('motorPosition-EI plane (d=0.03): S \cdot A analysis', 'FontSize',14,'FontWeight','bold');

outfile = fullfile(saveDir, 'second_family_EI_analysis.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[32 24],'PaperPosition',[0 0 32 24]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('Saved to %s\n', outfile);

% ---- Extract some values along the second family curves ----
% Use contourc on asymmetry to get alpha=0 contours
C = contourc(EI_list, MP_norm_list, asymmetry, [0 0]);
curves = {};
idx = 1;
while idx < size(C, 2)
    npts = C(2, idx);
    cx = C(1, (idx+1):(idx+npts));
    cy = C(2, (idx+1):(idx+npts));
    curves{end+1} = struct('EI', cx, 'mp_norm', cy); %#ok<AGROW>
    idx = idx + npts + 1;
end

fprintf('\nFound %d alpha=0 contour segments:\n', numel(curves));
for ic = 1:numel(curves)
    EI_span = max(curves{ic}.EI)/min(curves{ic}.EI);  % ratio for log scale
    mp_span = max(curves{ic}.mp_norm) - min(curves{ic}.mp_norm);
    fprintf('  Curve %d: %d pts, EI [%.2e, %.2e] (ratio %.1f), mp [%.3f, %.3f] (span %.3f)\n', ...
        ic, numel(curves{ic}.EI), ...
        min(curves{ic}.EI), max(curves{ic}.EI), EI_span, ...
        min(curves{ic}.mp_norm), max(curves{ic}.mp_norm), mp_span);
end

% Classify: vertical = small mp_span relative to EI span; non-vertical = large mp_span
fprintf('\n=== Sampling non-vertical curves (second family) ===\n');
fprintf('%-6s | %-10s | %-8s | %-10s | %-10s | %-10s | %-10s\n', ...
    'Curve', 'EI', 'x_M/L', '|S|', '|A|', '|S|/|A|', 'angle(S/A)');
fprintf('%s\n', repmat('-', 1, 80));

for ic = 1:numel(curves)
    mp_span = max(curves{ic}.mp_norm) - min(curves{ic}.mp_norm);
    if mp_span < 0.05, continue; end  % skip vertical lines

    n_pts = numel(curves{ic}.EI);
    sample_idx = round(linspace(1, n_pts, min(6, n_pts)));

    for is = sample_idx
        ei_val = curves{ic}.EI(is);
        mp_val = curves{ic}.mp_norm(is);

        S_val = interp2(EI_list, MP_norm_list, S_grid, ei_val, mp_val);
        A_val = interp2(EI_list, MP_norm_list, A_grid, ei_val, mp_val);

        fprintf('%5d  | %9.2e  | %7.4f  | %9.2e  | %9.2e  | %9.2e  | %+8.1f°\n', ...
            ic, ei_val, mp_val, abs(S_val), abs(A_val), ...
            abs(S_val)/(abs(A_val)+eps), rad2deg(angle(S_val/A_val)));
    end
    fprintf('%s\n', repmat('-', 1, 80));
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
