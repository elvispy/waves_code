function replot_omega_EI_sweep(S, saveDir)
% Create omega-EI sweep figures (MATLAB 2018b-compatible, publication-ready)
%
% Usage:
%   % First, run your simulation to get S
%   S = thrust_vs_omega_EI_sweep;
%
%   % Then, run this function
%   replot_omega_EI_sweep(S)
%   replot_omega_EI_sweep(S, 'my_output_dir')

if nargin < 2, saveDir = 'data'; end
if ~exist(saveDir,'dir'), mkdir(saveDir); end

% Handle S being a table (as produced at the end of Script 2) or struct
D = load('data/sweeperEIOmega.mat');
S = D.S;

% ---- Parameters (from replot_EI_sweep) ----
BASE_FONT       = 'Times';
FIGSIZE_3D_CM   = [18 15]; % [Width Height] in centimeters
LINE_W          = 1.4;
MK              = 5.5;
GRID_ALPHA      = 0.2;
FONT_SIZE_AXIS  = 10;
FONT_SIZE_TITLE = 11;
% Color matrix C is not needed for surf plots, which use a colormap

% ---- Load / Process Data ----
% Get unique omega and EI values to build the grid
omega_list = unique([S.omega]);
EI_list    = unique([S.EI]);
n_omega    = numel(omega_list);
n_EI       = numel(EI_list);

% Reshape data for surf plots
% Note: Script 2 saved S(iw, ie), so omega is rows (1st dim), EI is cols (2nd dim)
EI_grid         = reshape([S.EI], n_omega, n_EI);
Omega_grid_rad  = reshape([S.omega], n_omega, n_EI);
Omega_grid_hz   = Omega_grid_rad / (2*pi);

% Data for Plot 1: Eta Edge Ratio
eta_ratio         = reshape([S.eta_edge_ratio], n_omega, n_EI);
log10_eta_ratio   = log10(eta_ratio);
log10_eta_ratio(isinf(log10_eta_ratio)) = NaN; % Handle potential -Inf

% Data for Plot 2: Asymmetry Factor
eta_1_sq     = abs(reshape([S.eta_1], n_omega, n_EI)).^2;
eta_end_sq   = abs(reshape([S.eta_end], n_omega, n_EI)).^2;
asymmetry_factor = (eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

% ---- Find "Surferbot" reference point ----
% These values are from the 'base' struct in thrust_vs_omega_EI_sweep
EIsurf_val     = 3.0e9 * 3e-2 * (9.9e-4)^3 / 12;
Omega_surf_val = 2*pi*80;

% Find the closest indices in our lists
[~, idx_EI]    = min(abs(EI_list - EIsurf_val));
[~, idx_omega] = min(abs(omega_list - Omega_surf_val));

% Get the exact values at this grid point for the scatter plot
EI_surferbot       = EI_grid(idx_omega, idx_EI);
Omega_surferbot_hz = Omega_grid_hz(idx_omega, idx_EI);
log10_eta_surferbot= log10_eta_ratio(idx_omega, idx_EI);
asymm_surferbot    = asymmetry_factor(idx_omega, idx_EI);

% ====================== FIGURE 1 (Eta Edge Ratio) ======================
fig1 = figure('Color','w', 'Units','centimeters');
set(fig1, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_3D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_3D_CM]);
set(fig1, 'Position', [5 5 FIGSIZE_3D_CM]); % On-screen position

ax1 = gca;
hold(ax1, 'on');

% The main surface plot
surf(ax1, EI_grid, Omega_grid_hz, log10_eta_ratio, 'DisplayName', 'log_{10}(|\eta_1 / \eta_{end}|)', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);

% The reference 'Surferbot' point
scatter3(ax1, EI_surferbot, Omega_surferbot_hz, log10_eta_surferbot, 100, ...
    'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', 'Surferbot');

% --- Styling ---
set(ax1, 'XScale', 'log');
xlabel(ax1, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax1, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
zlabel(ax1, 'log_{10}(|\eta_1 / \eta_{end}|)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
title(ax1, 'Tail Amplitude Ratio', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

caxis(ax1, [-1 1]);
cb = colorbar(ax1);
cb.Label.String = 'Log Ratio';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

legend(ax1, 'show', 'Location', 'southoutside', 'Box', 'off', 'FontName', BASE_FONT);
view(ax1, -30, 25); % Set 3D view angle

% Apply custom helper style
style_axes_3d(ax1, BASE_FONT, FONT_SIZE_AXIS, GRID_ALPHA);

% --- Save ---
print(fig1, fullfile(saveDir,'fig1_omega_EI_eta_ratio.pdf'), '-dpdf','-painters','-r300');
print(fig1, fullfile(saveDir,'fig1_omega_EI_eta_ratio.png'), '-dpng','-r300');

% ====================== FIGURE 2 (Asymmetry Factor) ======================
fig2 = figure('Color','w', 'Units','centimeters');
set(fig2, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_3D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_3D_CM]);
set(fig2, 'Position', [6 6 FIGSIZE_3D_CM]); % Offset on-screen position

ax2 = gca;
hold(ax2, 'on');

% The main surface plot
surf(ax2, EI_grid, Omega_grid_hz, asymmetry_factor, 'DisplayName', 'Asymmetry Factor', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);

% The reference 'Surferbot' point
scatter3(ax2, EI_surferbot, Omega_surferbot_hz, asymm_surferbot, 100, ...
    'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', 'Surferbot');

% --- Styling ---
set(ax2, 'XScale', 'log');
xlabel(ax2, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax2, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
zlabel(ax2, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
title(ax2, 'Thrust Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

caxis(ax2, [-1 1]);
cb = colorbar(ax2);
cb.Label.String = 'Factor';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

legend(ax2, 'show', 'Location', 'southoutside', 'Box', 'off', 'FontName', BASE_FONT);
view(ax2, -30, 25); % Set 3D view angle

% Apply custom helper style
style_axes_3d(ax2, BASE_FONT, FONT_SIZE_AXIS, GRID_ALPHA);

% --- Save ---
print(fig2, fullfile(saveDir,'fig2_omega_EI_asymmetry.pdf'), '-dpdf','-painters','-r300');
print(fig2, fullfile(saveDir,'fig2_omega_EI_asymmetry.png'), '-dpng','-r300');

end

% ====================== Helper Function ======================
function style_axes_3d(ax, baseFont, fontSize, gridAlpha)
% Adapted from your original style_axes for 3D plots
set(ax, 'FontName', baseFont, 'FontSize', fontSize, 'LineWidth', 0.75, ...
    'TickDir', 'out', ...
    'Box', 'on'); % 'Box' on is standard for 3D plots to see all edges
ax.GridAlpha = gridAlpha;
ax.MinorGridAlpha = gridAlpha;
set(ax, 'XMinorTick', 'on', 'YMinorTick', 'on', 'ZMinorTick', 'on');
grid(ax, 'on'); % Ensure grid is visible
end