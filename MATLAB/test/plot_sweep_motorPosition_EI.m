function plot_sweep_motorPosition_EI(saveDir, export)
% Create motorPosition–EI sweep figures (MATLAB 2018b-compatible, publication-ready)
%
% Usage (suggested):
%   % After running sweep_motorPosition_EI and saving S:
%   %   save('data/sweepΩMotorPositionEI.mat','S');
%   plot_sweep_motorPosition_EI;
%   plot_sweep_motorPosition_EI('my_output_dir', true);

if nargin < 1, saveDir = 'data'; end
if nargin < 2, export = false; end
if ~exist(saveDir,'dir'), mkdir(saveDir); end

% Handle S being a table or struct; edit filename if needed
D = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
S = D.S;
if istable(S)
    S = table2struct(S);
end

% ---- Parameters (copied from plot_sweep_omega_EI) ----
BASE_FONT       = 'Times';
FIGSIZE_3D_CM   = [18 15]; % [Width Height] in centimeters
FIGSIZE_2D_CM   = [18 15];
FIGSIZE_CM_LINE = [18 16]; %#ok<NASGU> % Reserved if needed
LINE_W          = 1.5;
MK              = 5.5;     %#ok<NASGU> % Reserved if needed
GRID_ALPHA      = 0.2;
FONT_SIZE_AXIS  = 18;
FONT_SIZE_TITLE = 20;

C = [0.00 0.45 0.70;
     0.85 0.33 0.10;
     0.00 0.60 0.50;
     0.95 0.90 0.25;
     0.80 0.47 0.65;
     0.00 0.00 0.00];

% ---- Load / Process Data ----
% Get unique motor positions and EI values to build the grid
mp_list  = unique([S.motor_position]/S(1).args.L_raft);
EI_list  = unique([S.EI]);
n_mp     = numel(mp_list);
n_EI     = numel(EI_list);

% Reshape data for surf / contour plots
% S(ip,ie): motor_position is rows (1st dim), EI is cols (2nd dim)
EI_grid        = reshape([S.EI],             n_mp, n_EI);
MP_grid        = reshape([S.motor_position]./S(1).args.L_raft, n_mp, n_EI);

% Data for Plot 1: Eta Edge Ratio
eta_ratio       = reshape([S.eta_edge_ratio], n_mp, n_EI);
log10_eta_ratio = log10(eta_ratio);
log10_eta_ratio(isinf(log10_eta_ratio)) = NaN; % Handle potential -Inf

% Data for Plot 2: Asymmetry Factor
eta_1_sq   = abs(reshape([S.eta_1],   n_mp, n_EI)).^2;
eta_end_sq = abs(reshape([S.eta_end], n_mp, n_EI)).^2;
asymmetry_factor = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

% ---- Find "Surferbot" reference point ----
% These values mirror the 'base' struct in sweep_motorPosition_EI
L_raft_val      = 0.05; % same as in the sweep script
EIsurf_val      = 3.0e9 * 3e-2 * (9.9e-4)^3 / 12;
MP_surf_val     = 0.24 / 2; % base.motor_position

all_args   = [S.args]; 
all_power  = [all_args.power];

thrust_grid = reshape([S.thrust_N], n_mp, n_EI);
power_grid  = reshape(all_power,   n_mp, n_EI);

% Assuming S.args.power is power dissipation (negative), so we plot T / (-P)
thrust_over_power = thrust_grid ./ power_grid;

max_abs_val = max(abs(thrust_over_power(:)), [], 'omitnan');
thrust_over_power_norm = thrust_over_power / max_abs_val;

% Data for Plot 5: Sxx vs motor_position
Sxx_grid = reshape([S.Sxx], n_mp, n_EI);

% Select 5 EI indices (logarithmically spaced)
idx_EI_5 = round(linspace(1, n_EI, 5));

% Find closest indices in our lists for the "Surferbot"
[~, idx_EI] = min(abs(EI_list - EIsurf_val));
[~, idx_mp] = min(abs(mp_list - MP_surf_val));

MP_surferbot       = MP_grid(idx_mp, idx_EI);
EI_surferbot       = EI_grid(idx_mp, idx_EI);
log10_eta_surferbot= log10_eta_ratio(idx_mp, idx_EI);
asymm_surferbot    = asymmetry_factor(idx_mp, idx_EI);

%% ====================== FIGURE 1 (Eta Edge Ratio) ======================
fig1 = figure('Color','w', 'Units','centimeters');
set(fig1, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_3D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_3D_CM]);
set(fig1, 'Position', [5 5 FIGSIZE_3D_CM]);

ax1 = gca;
hold(ax1, 'on');

surf(ax1, EI_grid, MP_grid, log10_eta_ratio, ...
    'DisplayName', 'log_{10}(|\eta_1 / \eta_{end}|)', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);

scatter3(ax1, EI_surferbot, MP_surferbot, log10_eta_surferbot, 100, ...
    'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
    'DisplayName', 'Surferbot');

set(ax1, 'XScale', 'log');
xlabel(ax1, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax1, 'Motor Position / Raft Length', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
zlabel(ax1, 'log_{10}(|\eta_1 / \eta_{end}|)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

caxis(ax1, [-1 1]);
cb1 = colorbar(ax1);
cb1.Label.String   = 'Log Ratio';
cb1.Label.FontName = BASE_FONT;
cb1.Label.FontSize = FONT_SIZE_AXIS;

legend(ax1, 'show', 'Location', 'southoutside', 'Box', 'off', 'FontName', BASE_FONT);
view(ax1, -30, 25);

style_axes_3d(ax1, BASE_FONT, FONT_SIZE_AXIS, GRID_ALPHA);

if export
    print(fig1, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig1.pdf'), '-dpdf','-painters','-r300');
    print(fig1, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig1.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 2 (Asymmetry Factor) ======================
fig2 = figure('Color','w', 'Units','centimeters');
set(fig2, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_3D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_3D_CM]);
set(fig2, 'Position', [6 6 FIGSIZE_3D_CM]);

ax2 = gca;
hold(ax2, 'on');

xlim([min(EI_grid(:)) max(EI_grid(:))]);
ylim([min(MP_grid(:)) max(MP_grid(:))]);

surf(ax2, EI_grid, MP_grid, asymmetry_factor, ...
    'DisplayName', 'Asymmetry Factor', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85, 'HandleVisibility', 'off');

scatter3(ax2, EI_surferbot, MP_surferbot, asymm_surferbot, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
    'DisplayName', 'Surferbot');

set(ax2, 'XScale', 'log');
xlabel(ax2, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax2, 'Motor Position / Raft Length', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
zlabel(ax2, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

caxis(ax2, [-1 1]);
colormap(ax2, bwr_colormap());
cb2 = colorbar(ax2); shading interp
cb2.Label.String   = 'Asymmetry Factor';
cb2.Label.FontName = BASE_FONT;
cb2.Label.FontSize = FONT_SIZE_AXIS;

legend(ax2, 'show', 'Location', 'best', 'Box', 'off', 'FontName', BASE_FONT);
view(ax2, -30, 25);

style_axes_3d(ax2, BASE_FONT, FONT_SIZE_AXIS, GRID_ALPHA);

if export
    print(fig2, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig2.pdf'), '-dpdf','-painters','-r300');
    print(fig2, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig2.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 3 (2D Contour: Asymmetry) ======================
fig3 = figure('Color','w', 'Units','centimeters');
set(fig3, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig3, 'Position', [7 7 FIGSIZE_2D_CM]);

ax3 = gca;
hold(ax3, 'on');

n_fill_levels = 50;
contourf(ax3, EI_grid, MP_grid, asymmetry_factor, n_fill_levels, ...
    'LineStyle', 'none', 'HandleVisibility', 'off');

% Line at base motor position
%yline(MP_surferbot, 'k--', 'LineWidth', 2);

scatter(ax3, EI_surferbot, MP_surferbot, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, ...
    'DisplayName', 'Surferbot');

set(ax3, 'XScale', 'log');
xlim(ax3, [min(EI_grid(:)) max(EI_grid(:))]);
ylim(ax3, [min(MP_grid(:)) max(MP_grid(:))]);

xlabel(ax3, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax3, 'Motor Position / Raft Length', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

caxis(ax3, [-1 1]);
colormap(ax3, bwr_colormap());
cb3 = colorbar(ax3);
cb3.Label.String   = 'Asymmetry Factor';
cb3.Label.FontName = BASE_FONT;
cb3.Label.FontSize = FONT_SIZE_AXIS;

text(ax3, EI_surferbot * 1.15, 0.98*MP_surferbot, ' Surferbot', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS, ...
    'Color', 'white', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center');

style_axes(ax3, BASE_FONT, GRID_ALPHA);

if export
    print(fig3, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig3.pdf'), '-dpdf','-painters','-r300');
    print(fig3, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig3.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 4 (Thrust/Power 2D Contour) ======================
fig4 = figure('Color','w', 'Units','centimeters');
set(fig4, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig4, 'Position', [8 8 FIGSIZE_2D_CM]);

ax4 = gca;
hold(ax4, 'on');

n_fill_levels = 50;
contourf(ax4, EI_grid, MP_grid, thrust_over_power_norm, n_fill_levels, ...
    'LineStyle', 'none', 'HandleVisibility', 'off');

% Line at base motor position
%yline(MP_surferbot, 'k--', 'LineWidth', 2);

scatter(ax4, EI_surferbot, MP_surferbot, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, ...
    'DisplayName', 'Surferbot');

set(ax4, 'XScale', 'log');
xlim(ax4, [min(EI_grid(:)) max(EI_grid(:))]);
ylim(ax4, [min(MP_grid(:)) max(MP_grid(:))]);

xlabel(ax4, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax4, 'Motor Position / Raft Length', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

caxis(ax4, [-1 1]);
colormap(ax4, bwr_colormap());
cb4 = colorbar(ax4);
cb4.Label.String   = 'Normalized T/P';
cb4.Label.FontName = BASE_FONT;
cb4.Label.FontSize = FONT_SIZE_AXIS;

text(ax4, EI_surferbot * 1.15, 0.98*MP_surferbot, ' Surferbot', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS, ...
    'Color', 'black', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center');

style_axes(ax4, BASE_FONT, GRID_ALPHA);

if export
    print(fig4, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig4.pdf'), '-dpdf','-painters','-r300');
    print(fig4, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig4.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 5 (Thrust and Sxx vs. Motor Position) ======================
fig5 = figure('Color','w', 'Units','centimeters');
set(fig5, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig5, 'Position', [9 9 FIGSIZE_2D_CM]);

ax5 = gca;
hold(ax5, 'on');

for i = 1:numel(idx_EI_5)
    idx_ei = idx_EI_5(i);
    
    % Thrust (dashed)
    plot(ax5, MP_grid(:, idx_ei), thrust_grid(:, idx_ei), ...
        'Color', C(i,:), 'LineWidth', 2*LINE_W, 'LineStyle', '--', ...
        'DisplayName', sprintf('EI = %.1e N m^4', EI_list(idx_ei)), ...
        'HandleVisibility', 'off');
        
    % Sxx (solid)
    plot(ax5, MP_grid(:, idx_ei), Sxx_grid(:, idx_ei), ...
        'Color', C(i,:), 'LineWidth', LINE_W, 'LineStyle', '-', ...
        'HandleVisibility', 'off'); 
end

% Dummy plots for legend entries
plot(ax5, NaN, NaN, 'k--', 'LineWidth', 3*LINE_W, 'DisplayName', 'Thrust');
plot(ax5, NaN, NaN, 'k-',  'LineWidth', LINE_W,  'DisplayName', 'S_{xx}');

xlabel(ax5, 'Motor Position / Raft Length', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax5, 'Force/Length (N/m)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
legend(ax5, 'show', 'Location', 'northwest', 'Box', 'off', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS+1);

style_axes(ax5, BASE_FONT, GRID_ALPHA);
set(ax5, 'XScale', 'linear');
yline(0, 'k:', 'HandleVisibility', 'off');

if export
    print(fig5, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig5.pdf'), '-dpdf','-painters','-r300');
    print(fig5, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig5.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 6 (Correlation Plot: Thrust vs Sxx) ======================
fig6 = figure('Color','w', 'Units','centimeters');
set(fig6, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig6, 'Position', [11 11 FIGSIZE_2D_CM]);

ax6 = gca;
hold(ax6, 'on');

markers = {'o', 's', '^', 'd', 'p'};
if numel(idx_EI_5) ~= numel(markers)
    error('Mismatch between marker count and EI group count');
end

color_data_all = MP_grid;
cmin = min(color_data_all(:));
cmax = max(color_data_all(:));

all_data = [thrust_grid(:); Sxx_grid(:)];
min_val = min(all_data, [], 'omitnan');
max_val = max(all_data, [], 'omitnan');

plot(ax6, [min_val max_val], [min_val max_val], ...
    'k--', 'LineWidth', LINE_W, ...
    'DisplayName', 'Identity (Thrust = S_{xx})', ...
    'HandleVisibility', 'off');

for ig = 1:numel(idx_EI_5)
    idx_ei     = idx_EI_5(ig);
    thrust_data= thrust_grid(:, idx_ei);
    Sxx_data   = Sxx_grid(:, idx_ei);
    color_data = color_data_all(:, idx_ei);
    
    scatter(ax6, thrust_data, Sxx_data, 120, color_data, ...
        'filled', 'Marker', markers{ig}, ...
        'MarkerFaceAlpha', 0.75, ...
        'DisplayName', sprintf('EI = %.1e', EI_list(idx_ei)));
end

xlabel(ax6, 'Thrust (N/m)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax6, 'S_{xx} (N/m)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

hLeg = legend(ax6, 'show', 'Location', 'northwest', 'Box', 'off', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
hLeg.ItemTokenSize(1) = 60;

colormap(ax6, 'jet');
caxis(ax6, [cmin cmax]);
cb6 = colorbar(ax6);
cb6.Label.String   = 'Motor Position / Raft Length';
cb6.Label.FontName = BASE_FONT;
cb6.Label.FontSize = FONT_SIZE_AXIS;

style_axes(ax6, BASE_FONT, GRID_ALPHA);

axis(ax6, 'equal');
xlim(ax6, [min_val max_val]);
ylim(ax6, [min_val max_val]);

if export
    print(fig6, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig6.pdf'), '-dpdf','-painters','-r300');
    print(fig6, fullfile(saveDir,'plot_sweep_motorPosition_EI_fig6.svg'), '-dsvg','-r300');
end

end

% ====================== Helper Functions ======================
function style_axes_3d(ax, baseFont, fontSize, gridAlpha)
set(ax, 'FontName', baseFont, 'FontSize', fontSize, 'LineWidth', 0.75, ...
    'TickDir', 'out', 'Box', 'on');
ax.GridAlpha      = gridAlpha;
ax.MinorGridAlpha = gridAlpha;
set(ax, 'XMinorTick', 'on', 'YMinorTick', 'on', 'ZMinorTick', 'on');
grid(ax, 'on');
end

function style_axes(ax, baseFont, gridAlpha)
set(ax,'FontName',baseFont,'FontSize',15,'LineWidth',0.75,...
    'TickDir','out','Box','off');
ax.GridAlpha      = gridAlpha;
ax.MinorGridAlpha = gridAlpha;
set(ax,'XMinorTick','on','YMinorTick','on');
grid(ax, 'on');
end

function cmap = bwr_colormap(n_colors, gamma)
    if nargin < 1 || isempty(n_colors)
        n_colors = 256;
    end
    if nargin < 2 || isempty(gamma)
        gamma = 1.3;
    end

    rgb_anchors = [0.99 0.35 0.00;  % orange
                   1.00 1.00 1.00;  % white
                   0.00 0.35 0.80]; % blue

    data_points = [-1 0 1];

    lab_anchors = rgb2lab(rgb_anchors);

    t_lin = linspace(-1,1,n_colors).';
    a     = abs(t_lin);
    t_nl  = sign(t_lin) .* (a.^gamma);

    lab_interp = interp1(data_points.', lab_anchors, t_nl, 'linear');
    cmap       = lab2rgb(lab_interp);
    cmap       = max(min(cmap,1),0);
end
