function plot_sweep_omega_EI(saveDir, export)
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
if nargin < 2, export = false; end
if ~exist(saveDir,'dir'), mkdir(saveDir); end

% Ensure src helpers (e.g., dispersion_k) are discoverable.
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, '..', 'src'));

% Handle S being a table (as produced at the end of Script 2) or struct
D = load('data/sweeperEIOmega.mat');
S = D.S;

% ---- Parameters (from replot_EI_sweep) ----
BASE_FONT       = 'Times';
FIGSIZE_3D_CM   = [18 15]; % [Width Height] in centimeters
FIGSIZE_2D_CM   = [18 15];
FIGSIZE_CM_LINE = [18, 16]; 
LINE_W          = 1.5;
MK              = 5.5;
GRID_ALPHA      = 0.2;
FONT_SIZE_AXIS  = 18;
FONT_SIZE_TITLE = 20;

% Color matrix C is not needed for surf plots, which use a colormap
C = [0.00 0.45 0.70;
     0.85 0.33 0.10;
     0.00 0.60 0.50;
     0.95 0.90 0.25;
     0.80 0.47 0.65;
     0.00 0.00 0.00];
 
% ---- Load / Process Data ----
% Get unique omega and EI values to build the grid
all_args = [S.args];
omega_list = unique([S.omega]);
EI_list    = unique([S.EI]);
n_omega    = numel(omega_list);
n_EI       = numel(EI_list);

% Reshape data for surf plots
% Note: Script 2 saved S(iw, ie), so omega is rows (1st dim), EI is cols (2nd dim)
EI_grid         = reshape([S.EI], n_omega, n_EI);
Omega_grid_rad  = reshape([S.omega], n_omega, n_EI);
Omega_grid_hz   = Omega_grid_rad / (2*pi);
k_grid          = reshape([all_args(:).k], n_omega, n_EI);

% Data for Plot 1: Eta Edge Ratio
eta_ratio         = reshape([S.eta_edge_ratio], n_omega, n_EI);
log10_eta_ratio   = log10(eta_ratio);
log10_eta_ratio(isinf(log10_eta_ratio)) = NaN; % Handle potential -Inf

% Data for Plot 2: Asymmetry Factor
eta_1_sq     = abs(reshape([S.eta_1], n_omega, n_EI)).^2;
eta_end_sq   = abs(reshape([S.eta_end], n_omega, n_EI)).^2;
asymmetry_factor = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

% ---- Find "Surferbot" reference point ----
% These values are from the 'base' struct in thrust_vs_omega_EI_sweep
EIsurf_val     = 3.0e9 * 3e-2 * (9.9e-4)^3 / 12;
Omega_surf_val = 2*pi*80;

 
% Then we extract the 'power' field from that array
all_power = [all_args.power];
thrust_grid = reshape([S.thrust_N], n_omega, n_EI);
power_grid = reshape(all_power, n_omega, n_EI);
% ------------------------------------

% Assuming S.args.power is power *dissipation* (negative), so we plot T / (-P)
thrust_over_power = thrust_grid ./ (power_grid);

max_abs_val = max(abs(thrust_over_power(:)), [], 'omitnan');

% Normalize the data
thrust_over_power_norm = thrust_over_power / max_abs_val;


% Data for Plot 5: Sxx and LH vs. Frequency
% Sxx is already calculated and saved in S.Sxx
Sxx_grid = reshape([S.Sxx], n_omega, n_EI);

% Select 5 EI indices (logarithmically spaced)
idx_EI_5 = round(linspace(1, n_EI, 5));

% Find the closest indices in our lists
[~, idx_EI]    = min(abs(EI_list - EIsurf_val));
[~, idx_omega] = min(abs(omega_list - Omega_surf_val));

% Get the exact values at this grid point for the scatter plot
EI_surferbot       = EI_grid(idx_omega, idx_EI);
Omega_surferbot_hz = Omega_grid_hz(idx_omega, idx_EI);
log10_eta_surferbot= log10_eta_ratio(idx_omega, idx_EI);
asymm_surferbot    = asymmetry_factor(idx_omega, idx_EI);

%% ====================== FIGURE 1 (Eta Edge Ratio) ======================
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
%title(ax1, 'Tail Amplitude Ratio', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

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
if export
    print(fig1, fullfile(saveDir,'plot_sweep_omega_EI_fig1.pdf'), '-dpdf','-painters','-r300');
    print(fig1, fullfile(saveDir,'plot_sweep_omega_EI_fig1.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 2 (Asymmetry Factor) ======================
fig2 = figure('Color','w', 'Units','centimeters');
set(fig2, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_3D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_3D_CM]);
set(fig2, 'Position', [6 6 FIGSIZE_3D_CM]); % Offset on-screen position

ax2 = gca;
hold(ax2, 'on');
xlim([min(EI_grid(:)) max(EI_grid(:))]); ylim([min(Omega_grid_hz(:)) max(Omega_grid_hz(:))]);
% The main surface plot
surf(ax2, EI_grid, Omega_grid_hz, asymmetry_factor, 'DisplayName', 'Asymmetry Factor', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85, 'HandleVisibility', 'off');

% The reference 'Surferbot' point
scatter3(ax2, EI_surferbot, Omega_surferbot_hz, asymm_surferbot, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', 'Surferbot');

% --- Styling ---
set(ax2, 'XScale', 'log'); 
xlabel(ax2, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax2, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
zlabel(ax2, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
%title(ax2, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

caxis(ax2, [-1 1]);
colormap(ax2, bwr_colormap());
cb = colorbar(ax2); shading interp
cb.Label.String = 'Asymmetry Factor';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

legend(ax2, 'show', 'Location', 'best', 'Box', 'off', 'FontName', BASE_FONT);
view(ax2, -30, 25); % Set 3D view angle

% Apply custom helper style
style_axes_3d(ax2, BASE_FONT, FONT_SIZE_AXIS, GRID_ALPHA);


% --- Save ---
if export
    print(fig2, fullfile(saveDir,'plot_sweep_omega_EI_fig2.pdf'), '-dpdf','-painters','-r300');
    print(fig2, fullfile(saveDir,'plot_sweep_omega_EI_fig2.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 3 (2D Contour Plot) ======================
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
ylim(ax3, [10 max(Omega_grid_hz(:))]);

xlabel(ax3, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax3, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
%title(ax3, 'Asymmetry Factor', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

caxis(ax3, [-1 1]);
colormap(ax3, bwr_colormap()); % Use the blue-white-red map
cb = colorbar(ax3);
cb.Label.String = 'Asymmetry Factor';
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
if export
    print(fig3, fullfile(saveDir,'plot_sweep_omega_EI_fig3.pdf'), '-dpdf','-painters','-r300');
    print(fig3, fullfile(saveDir,'plot_sweep_omega_EI_fig3.svg'), '-dsvg','-r300');
end


%% ====================== FIGURE 4 (Thrust/Power 2D Contour) ======================
fig4 = figure('Color','w', 'Units','centimeters');
set(fig4, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig4, 'Position', [8 8 FIGSIZE_2D_CM]); % Offset on-screen position

ax4 = gca;
hold(ax4, 'on');

% Set levels for a smooth-ish fill, and no lines
n_fill_levels = 50;
% --- PLOT NORMALIZED DATA ---
contourf(ax4, EI_grid, Omega_grid_hz, thrust_over_power_norm, n_fill_levels, ...
    'LineStyle', 'none', 'HandleVisibility', 'off');

% Overlay the 80 Hz line
yline(80, 'k--', 'LineWidth', 2);

% The reference 'Surferbot' point (now 2D)
scatter(ax4, EI_surferbot, Omega_surferbot_hz, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, 'DisplayName', 'Surferbot');

% --- Styling ---
set(ax4, 'XScale', 'log');
xlim(ax4, [min(EI_grid(:)) max(EI_grid(:))]); 
ylim(ax4, [min(Omega_grid_hz(:)) max(Omega_grid_hz(:))]);

xlabel(ax4, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax4, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
%title(ax4, 'Normalized Thrust / Power', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_TITLE);

% --- USE DIVERGING COLORMAP AND LIMITS ---
caxis(ax4, [-1 1]); % Set limits to -1 and 1
colormap(ax4, bwr_colormap()); % Use the blue-white-red map
cb = colorbar(ax4);
cb.Label.String = 'Normalized T/P'; % Update label
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;
% -----------------------------------------

% Add the text label
text(ax4, EI_surferbot * 1.15, 0.98*Omega_surferbot_hz, ' Surferbot', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS, ...
    'Color', 'black', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center');

% Apply the 2D helper style
style_axes(ax4, BASE_FONT, GRID_ALPHA);

% --- Save ---
if export
    print(fig4, fullfile(saveDir,'plot_sweep_omega_EI_fig4.pdf'), '-dpdf','-painters','-r300');
    print(fig4, fullfile(saveDir,'plot_sweep_omega_EI_fig4.svg'), '-dsvg','-r300');
end


% ====================== FIGURE 5 (Thrust and Sxx vs. Frequency) ======================
% --- Use the 2D figure size from FIGSIZE_2D_CM ---
fig5 = figure('Color','w', 'Units','centimeters');
set(fig5, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig5, 'Position', [9 9 FIGSIZE_2D_CM]); % Offset on-screen position

ax5 = gca;
hold(ax5, 'on');

for i = 1:numel(idx_EI_5)
    idx_ei = idx_EI_5(i);
    
    % Plot Thrust (Solid Line)
    plot(ax5, Omega_grid_hz(:, idx_ei), thrust_grid(:, idx_ei), ...
        'Color', C(i,:), 'LineWidth', 2*LINE_W, 'LineStyle', '--', ...
        'DisplayName', sprintf('EI = %.1e N m^4', EI_list(idx_ei)), ...
        'HandleVisibility', 'off');
        
    % Plot Sxx (Dashed Line)
    plot(ax5, Omega_grid_hz(:, idx_ei), Sxx_grid(:, idx_ei), ...
        'Color', C(i,:), 'LineWidth', LINE_W, 'LineStyle', '-', ...
        'HandleVisibility', 'off'); 
end

% --- Add "dummy" plots for a clean legend ---
% These plots won't be visible, they just create the legend entries
plot(ax5, NaN, NaN, 'k--', 'LineWidth', 3*LINE_W, 'DisplayName', 'Thrust');
plot(ax5, NaN, NaN, 'k-', 'LineWidth', LINE_W, 'DisplayName', 'S_{xx}');
% ---------------------------------------------

% Styling
%title(ax5, 'Thrust vs. S_{xx} (Radiation Stress)');
xlabel(ax5, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax5, 'Force/Length (N/m)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
legend(ax5, 'show', 'Location', 'northwest', 'Box', 'off', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS+1);
style_axes(ax5, BASE_FONT, GRID_ALPHA); % Use 2D style
set(ax5, 'XScale', 'linear'); % Use linear scale for frequency
yline(0, 'k:', 'HandleVisibility', 'off'); % Add a dotted zero line

% --- Save ---
if export
    print(fig5, fullfile(saveDir,'plot_sweep_omega_EI_fig5.pdf'), '-dpdf','-painters','-r300');
    print(fig5, fullfile(saveDir,'plot_sweep_omega_EI_fig5.svg'), '-dsvg','-r300');
end


%%% ====================== FIGURE 6 (Correlation Plot with Omega and EI) ======================
fig6 = figure('Color','w', 'Units','centimeters');
set(fig6, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig6, 'Position', [11 11 FIGSIZE_2D_CM]); % Offset on-screen position

ax6 = gca;
hold(ax6, 'on');

% --- Define 5 markers for 5 EI groups ---
% We will use the 5 EI indices from FIG 5: idx_EI_5
markers = {'o', 's', '^', 'd', 'p'};
if numel(idx_EI_5) ~= numel(markers)
    error('Mismatch between marker count and EI group count');
end

% Get the Frequency data for coloring
color_data_all = Omega_grid_hz;

% Find the min/max Frequency for consistent coloring
cmin = min(color_data_all(:));
cmax = max(color_data_all(:));

% --- Add the y=x "line of identity" ---
all_data = [thrust_grid(:); Sxx_grid(:)];
min_val = min(all_data, [], 'omitnan');
max_val = max(all_data, [], 'omitnan');
plot(ax6, [min_val max_val], [min_val max_val], ...
    'k--', 'LineWidth', LINE_W, 'DisplayName', 'Identity (Thrust = S_{xx})', ...
    'HandleVisibility', 'off'); 

% Loop through each of the 5 EI groups
for ig = 1:numel(idx_EI_5)
    % Get the column index for this EI
    idx_ei = idx_EI_5(ig);
    
    % Get all data points for this EI
    thrust_data = thrust_grid(:, idx_ei);
    Sxx_data    = Sxx_grid(:, idx_ei);
    color_data  = color_data_all(:, idx_ei);
    
    % Plot this group's data
    scatter(ax6, thrust_data, Sxx_data, 120, color_data, ...
        'filled', 'Marker', markers{ig}, ...
        'MarkerFaceAlpha', 0.75, 'DisplayName', sprintf('EI = %.1e', EI_list(idx_ei)));
end



% --- Styling ---
%title(ax6, 'Correlation of Thrust and S_{xx}');
xlabel(ax6, 'Thrust (N/m)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax6, 'S_{xx} (N/m)', 'FontName', 'Times', 'FontSize', FONT_SIZE_AXIS);

% --- REMOVED LEGEND ---
hLeg = legend(ax6, 'show', 'Location', 'northwest', 'Box', 'off', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
hLeg.ItemTokenSize(1) = 60; % Set symbol width (default is 30)

% --- Add colorbar for Frequency (omega) ---
colormap(ax6, 'jet'); % 'parula' or 'jet'
caxis(ax6, [cmin cmax]);
cb = colorbar(ax6);
cb.Label.String = 'Frequency (Hz)';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

style_axes(ax6, BASE_FONT, GRID_ALPHA); % Use 2D style

% Force axes to be equal to show 1:1 relationship clearly
axis(ax6, 'equal'); 
xlim(ax6, [min_val max_val]);
ylim(ax6, [min_val max_val]);

% --- Save ---
if export
    % Using the filenames from your snippet
    print(fig6, fullfile(saveDir,'plot_sweep_omega_EI_fig6.pdf'), '-dpdf','-painters','-r300');
    print(fig6, fullfile(saveDir,'plot_sweep_omega_EI_fig6.svg'), '-dsvg','-r300');
end

%% ====================== FIGURE 7 (Asymmetry with Natural Freq Line) ======================
% Purpose: Same as Fig 3 (Asymmetry), but overlaying the theoretical 
% natural frequency: omega = A * sqrt(EI)

fig7 = figure('Color','w', 'Units','centimeters');
set(fig7, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_2D_CM, ...
    'PaperPosition', [0 0 FIGSIZE_2D_CM]);
set(fig7, 'Position', [12 12 FIGSIZE_2D_CM]); % Offset on-screen position

ax7 = gca;
hold(ax7, 'on');

% --- 1. Background Contour (Same as Fig 3) ---
n_fill_levels = 50;
contourf(ax7, EI_grid, Omega_grid_hz, asymmetry_factor, n_fill_levels, ...
    'LineStyle', 'none', 'HandleVisibility', 'off');

% --- 2. Calculate and Plot Theoretical Curve ---
% Equation: omega = A * sqrt(EI)
% A = 2 * pi * (4.73)^2 / (rho^0.5 * L^2)

% Extract parameters (assuming constant rho/L across the sweep)
% all_args was defined earlier in the script
rho_r = all_args(1).rho_raft; 
L_r   = all_args(1).L_raft;
d     = all_args(1).d;
rho   = all_args(1).rho;
H     = all_args(1).domainDepth;
g     = all_args(1).g;
nu    = all_args(1).nu;
sigma = all_args(1).sigma;
% Calculate A
% Note: 4.73 is approx beta*L for the first free-free beam mode
beta_l = 4.73/L_r;
k = beta_l;
A_val = (beta_l)^2 / sqrt(rho_r);

% Generate Smooth Curve Data
% Use logspace for EI since the X-axis is logarithmic
EI_curve_vec = logspace(log10(min(EI_grid(:))), log10(max(EI_grid(:))), 50);
%Omega_curve_rad = A_val * sqrt(EI_curve_vec);

%k_vec = interp1(EI_grid, k_grid, EI_curve_vec)
Omega_curve_rad = sqrt((EI_curve_vec * beta_l^4 + d * rho * g) ./ ...
    (rho_r + d * rho / (k * tanh(k * H))));
% Convert to Hz for the plot
Freq_curve_hz = Omega_curve_rad / (2*pi);

% Plot the line (Cyan or Green usually pops well against Red/Blue)
plot(ax7, EI_curve_vec, Freq_curve_hz, 'c-', 'LineWidth', 3, ...
    'DisplayName', sprintf('Resonant frequency (mode %.3f)', beta_l), ...
    'HandleVisibility', 'off');

% --- 2b. Implicit resonance curve using k = k(omega) from dispersion ---
% Solve, for each EI:
% EI*beta_l^4 + d*rho*g - omega^2*(rho_r + d*rho/(k(omega)*tanh(k(omega)*H))) = 0
if exist('dispersion_k', 'file') == 2 && true
    omega_min = min(Omega_grid_rad(:));
    omega_max = max(Omega_grid_rad(:));
    Omega_curve_disp_rad = NaN(size(EI_curve_vec));
    omega_scan = linspace(omega_min, omega_max, 80);
    prev_root = NaN;

    for iEI = 1:numel(EI_curve_vec)
        EI_i = EI_curve_vec(iEI);
        f_real = @(om) resonance_residual_real(om, EI_i, beta_l, rho_r, d, rho, g, H, nu, sigma);

        % Prefer continuity in EI by first trying previous root.
        root_i = NaN;
        if isfinite(prev_root)
            try
                r_try = fzero(f_real, prev_root);
                if isfinite(r_try) && r_try > 0
                    root_i = r_try;
                end
            catch
            end
        end

        % If continuation failed, bracket from a coarse scan and solve.
        if ~isfinite(root_i)
            vals = arrayfun(f_real, omega_scan);
            valid = isfinite(vals);
            idx = find(valid(1:end-1) & valid(2:end) & (vals(1:end-1).*vals(2:end) <= 0));
            if ~isempty(idx)
                mids = 0.5 * (omega_scan(idx) + omega_scan(idx+1));
                [~, jbest] = min(abs(mids - Omega_curve_rad(iEI)));
                a = omega_scan(idx(jbest));
                b = omega_scan(idx(jbest)+1);
                try
                    r_try = fzero(f_real, [a, b]);
                    if isfinite(r_try) && r_try > 0
                        root_i = r_try;
                    end
                catch
                end
            end
        end

        if isfinite(root_i)
            Omega_curve_disp_rad(iEI) = root_i;
            prev_root = root_i;
        end
    end

    Freq_curve_disp_hz = Omega_curve_disp_rad / (2*pi);
    plot(ax7, EI_curve_vec, Freq_curve_disp_hz, 'k--', 'LineWidth', 2.2, ...
        'DisplayName', 'Resonant frequency with k(\omega)', ...
        'HandleVisibility', 'off');
else
    warning('dispersion_k.m not found on path. Skipping k(omega) resonance curve.');
end

% --- 3. Surferbot Reference Point ---
scatter(ax7, EI_surferbot, Omega_surferbot_hz, 100, ...
    'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, 'DisplayName', 'Surferbot');

% --- 4. Styling ---
set(ax7, 'XScale', 'log');
xlim(ax7, [min(EI_grid(:)) max(EI_grid(:))]); 
ylim(ax7, [10 max(Omega_grid_hz(:))]);

xlabel(ax7, 'EI (N m^4)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax7, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

caxis(ax7, [-1 1]);
colormap(ax7, bwr_colormap()); 
cb = colorbar(ax7);
cb.Label.String = 'Asymmetry Factor';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

% Add Legend to identify the curve and the dot
legend(ax7, 'show', 'Location', 'southeast', 'Box', 'off', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

% Apply the 2D helper style
style_axes(ax7, BASE_FONT, GRID_ALPHA);

% --- Save ---
if export
    print(fig7, fullfile(saveDir,'plot_sweep_omega_EI_fig7.pdf'), '-dpdf','-painters','-r300');
    print(fig7, fullfile(saveDir,'plot_sweep_omega_EI_fig7.svg'), '-dsvg','-r300');
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

function val = resonance_residual_real(omega, EI, beta_l, rho_r, d, rho, g, H, nu, sigma)
% Real-valued resonance residual using k = real(dispersion_k(omega,...)).
if ~isfinite(omega) || omega <= 0
    val = NaN;
    return;
end


k_complex = dispersion_k(omega, g, H, nu, sigma, rho, beta_l);
k_real = real(k_complex);

T = 0; h = 0.1*rho/rho_r;
k_real = dispersion_k2(omega, EI, g, H, rho, rho_r, h, T);

if ~isfinite(k_real) || (k_real <= 0)
    val = NaN;
    return;
end

added_mass = d * rho / (k_real * tanh(k_real * H));
val = EI * beta_l^4 + d * rho * g - omega^2 * (rho_r + added_mass);
end

function cmap = bwr_colormap(n_colors, gamma)
    % BWR-style diverging colormap with more contrast near endpoints.
    %
    % 0   -> white
    % -1  -> blue-ish
    % +1  -> orange-ish
    %
    % n_colors : number of entries (default 256)
    % gamma    : >1 increases distinguishability near endpoints (default 2)
    %
    % Requires Image Processing Toolbox (rgb2lab, lab2rgb).

    if nargin < 1 || isempty(n_colors)
        n_colors = 256;
    end
    if nargin < 2 || isempty(gamma)
        gamma = 1.;  % >1 => more resolution near -1 and 1
    end

    % Friendly endpoints in sRGB [0,1]
    rgb_anchors = [0.99 0.35 0.00;  % orange
                   1.00 1.00 1.00;   % white (0)
                   0.00 0.35 0.80];   % blue
                   

    % Positions in "data space"
    data_points = [-1 0 1];

    % Convert anchors to Lab (perceptual-ish)
    lab_anchors = rgb2lab(rgb_anchors);

    % Linear parameter in [-1,1]
    t_lin = linspace(-1,1,n_colors).';

    % Nonlinear remap to emphasize endpoints:
    % gamma > 1 => flatter center, steeper near ends
    a = abs(t_lin);
    t_nl = sign(t_lin) .* (a.^gamma);

    % Interpolate in Lab using nonlinear parameter
    lab_interp = interp1(data_points.', lab_anchors, t_nl, 'linear');

    % Back to RGB
    cmap = lab2rgb(lab_interp);

    % Clamp to [0,1] to avoid numerical excursions
    cmap = max(min(cmap,1),0);
end
