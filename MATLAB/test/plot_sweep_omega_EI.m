function replot_omega_EI_sweep(saveDir, saveVar)
% Create omega-EI sweep figures (MATLAB 2018b-compatible, publication-ready)
%
% Usage:
%   % First, run your simulation to get S
%   S = thrust_vs_omega_EI_sweep;
%
%   % Then, run this function
%   replot_omega_EI_sweep(S)
%   replot_omega_EI_sweep(S, 'my_output_dir')

if nargin < 1, saveDir = 'data'; end
if nargin < 2, saveVar = true; end
if ~exist(saveDir,'dir'), mkdir(saveDir); end

% Handle S being a table (as produced at the end of Script 2) or struct
D = load('data/sweeperEIOmega.mat');
S = D.S;

% ---- Parameters (from replot_EI_sweep) ----
BASE_FONT       = 'Times';
FIGSIZE_3D_CM   = [18 15]; % [Width Height] in centimeters
FIGSIZE_2D_CM   = [18 15];
LINE_W          = 1.4;
MK              = 5.5;
GRID_ALPHA      = 0.2;
FONT_SIZE_AXIS  = 14;
FONT_SIZE_TITLE = 16;
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
if saveVar
    print(fig1, fullfile(saveDir,'fig1_omega_EI_eta_ratio.pdf'), '-dpdf','-painters','-r300');
    print(fig1, fullfile(saveDir,'fig1_omega_EI_eta_ratio.png'), '-dpng','-r300');
end

% ====================== FIGURE 2 (Asymmetry Factor) ======================
fig2 = figure('Color','w', 'Units','centimeters');
set(fig2, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_3D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_3D_CM]);
set(fig2, 'Position', [6 6 FIGSIZE_3D_CM]); % Offset on-screen position

ax2 = gca;
hold(ax2, 'on');
xlim([min(EI_grid(:)) max(EI_grid(:))]); ylim([min(Omega_grid_hz(:)) max(Omega_grid_hz(:))]);
% The main surface plot
surf(ax2, EI_grid, Omega_grid_hz, asymmetry_factor, 'DisplayName', 'Asymmetry Factor', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);

% The reference 'Surferbot' point
scatter3(ax2, EI_surferbot, Omega_surferbot_hz, asymm_surferbot, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', 'Surferbot');

% --- Styling ---
set(ax2, 'XScale', 'log'); 
xlabel(ax2, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax2, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
zlabel(ax2, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
title(ax2, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

caxis(ax2, [-1 1]);
colormap(ax2, bwr_colormap());
cb = colorbar(ax2); shading interp
cb.Label.String = 'Factor';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

legend(ax2, 'show', 'Location', 'southoutside', 'Box', 'off', 'FontName', BASE_FONT);
view(ax2, -30, 25); % Set 3D view angle

% Apply custom helper style
style_axes_3d(ax2, BASE_FONT, FONT_SIZE_AXIS, GRID_ALPHA);


% --- Save ---
if saveVar
    print(fig2, fullfile(saveDir,'fig2_omega_EI_asymmetry.pdf'), '-dpdf','-painters','-r300');
    print(fig2, fullfile(saveDir,'fig2_omega_EI_asymmetry.png'), '-dpng','-r300');
end

% ====================== FIGURE 3 (2D Contour Plot) ======================
fig3 = figure('Color','w', 'Units','centimeters');
set(fig3, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig3, 'Position', [7 7 FIGSIZE_2D_CM]); % Offset on-screen position

ax3 = gca;
hold(ax3, 'on');

% Set 20 levels for a smooth-ish fill, and no lines
n_fill_levels = 50;
contourf(ax3, EI_grid, Omega_grid_hz, asymmetry_factor, n_fill_levels, ...
    'LineStyle', 'none', 'HandleVisibility', 'off');

% Overlay 10 explicit contour lines in dark gray
%n_line_levels = 10;
yline(80, 'k--', 'LineWidth', 2);

% The reference 'Surferbot' point (now 2D)
% Made the marker black with a white edge for visibility on the colormap
scatter(ax3, EI_surferbot, Omega_surferbot_hz, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, 'DisplayName', 'Surferbot');

% --- Styling ---
set(ax3, 'XScale', 'log');
xlim(ax3, [min(EI_grid(:)) max(EI_grid(:))]); 
ylim(ax3, [min(Omega_grid_hz(:)) max(Omega_grid_hz(:))]);

xlabel(ax3, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax3, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
title(ax3, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

caxis(ax3, [-1 1]);
colormap(ax3, bwr_colormap()); % Use the blue-white-red map
cb = colorbar(ax3);
cb.Label.String = 'Factor';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

% Only show legend for the scatter plot
text(ax3, EI_surferbot * 1.15, 0.98*Omega_surferbot_hz, ' Surferbot', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS, ...
    'Color', 'white', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center');
%htext = legend(ax3, 'show', 'FontSize', 15, 'Location', 'southeast',  ...
%    'Box', 'off', 'FontName', BASE_FONT);
%set(htext, 'Color', 'white');
% Apply the 2D helper style (see Step 3)
style_axes(ax3, BASE_FONT, GRID_ALPHA);

% --- Save ---
if saveVar
    print(fig3, fullfile(saveDir,'fig3_omega_EI_contour.pdf'), '-dpdf','-painters','-r300');
    print(fig3, fullfile(saveDir,'fig3_omega_EI_contour.png'), '-dpng','-r300');
end
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

function style_axes(ax,baseFont,gridAlpha)
set(ax,'FontName',baseFont,'FontSize',15,'LineWidth',0.75,...
    'TickDir','out','Box','off');
ax.GridAlpha = gridAlpha;
ax.MinorGridAlpha = gridAlpha;
set(ax,'XMinorTick','on','YMinorTick','on');
grid(ax, 'on'); % Turn grid on
end


% ----- NEW HELPER FUNCTION -----
function cmap = bwr_colormap(n_colors)
    % Creates a custom Blue-White-Red colormap
    % n_colors: (Optional) number of entries, defaults to 256
    if nargin < 1, n_colors = 256; end
    
    % Define the "anchor" colors
    colors_in = [0 0 1;  % Blue
                 1 1 1;  % White
                 1 0 0]; % Red
                 
    % Define the "anchor" data points (from -1 to 1)
    data_points = [-1, 0, 1];
    
    % Linearly interpolate to create the full n_colors map
    % This finds the R, G, and B values for each of the n_colors
    % steps between -1 and 1.
    cmap = interp1(data_points, colors_in, linspace(-1, 1, n_colors));
end