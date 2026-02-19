function plot_sweep_omega_motorPosition(saveDir, export)
% Create omega–motorPosition sweep figure (MATLAB 2018b-compatible)
%
% Usage:
%   % First run: S = sweep_omega_motorPosition; save('data/motorPositionOmega.mat','S');
%   plot_sweep_omega_motorPosition;
%   plot_sweep_omega_motorPosition('my_output_dir', true);

if nargin < 1, saveDir = 'data'; end
if nargin < 2, export = false; end
if ~exist(saveDir,'dir'), mkdir(saveDir); end

% Ensure src helpers use same path logic as other scripts
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, '..', 'src'));

% Try to load data
dataFile = fullfile(saveDir, 'sweepOmegaMotorPosition.mat');
if ~exist(dataFile, 'file')
    error('No data file found. Please run sweep_omega_motorPosition and save the result as motorPositionOmega.mat in data/.');
end

if exist(dataFile, 'file')
    D = load(dataFile);
    if isfield(D, 'S')
        S = D.S;
    else
        error('Data file %s does not contain variable S.', dataFile);
    end
else
    error('No data file found. Please run sweep_omega_motorPosition and save the result as sweepOmegaMotorPosition.mat in data/.');
end

% Handle table vs struct
if istable(S)
    S = table2struct(S);
end

% ---- Parameters ----
BASE_FONT       = 'Times';
FIGSIZE_CM      = [18 15]; 
FONT_SIZE_AXIS  = 18;
GRID_ALPHA      = 0.2;

% ---- Load / Process Data ----
% Unique lists
omega_list = unique([S.omega]);
mp_list    = unique([S.motor_position]);
n_omega    = numel(omega_list);
n_mp       = numel(mp_list);

% Reshape Grid
% S(iw, ip) -> omega is rows (1st dim), motor_position is cols (2nd dim)
% Verify orientation by checking values
O_grid_rad = reshape([S.omega], n_omega, n_mp);
O_grid_hz  = O_grid_rad / (2*pi);
MP_grid    = reshape([S.motor_position], n_omega, n_mp);

% Calculate Asymmetry
eta_1_sq   = abs(reshape([S.eta_1],   n_omega, n_mp)).^2;
eta_end_sq = abs(reshape([S.eta_end], n_omega, n_mp)).^2;
asymmetry_factor = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

% ---- Find "Surferbot" reference point ----
% Base params from sweep_omega_motorPosition.m
% L_raft = 0.05;
% motor_position = 0.24 * L_raft / 2;
% omega = 2*pi*80;
if isfield(S(1), 'args') && isfield(S(1).args, 'L_raft')
    L_raft_val = S(1).args.L_raft;
else
    L_raft_val = 0.05; % Default fallback
end

MP_surf_val     = 0.24 * L_raft_val / 2; 
Omega_surf_val  = 2*pi*80;

% Find indices
[~, idx_mp]    = min(abs(mp_list - MP_surf_val));
[~, idx_omega] = min(abs(omega_list - Omega_surf_val));

MP_surferbot       = MP_grid(idx_omega, idx_mp);
Omega_surferbot_hz = O_grid_hz(idx_omega, idx_mp);
asymm_surferbot    = asymmetry_factor(idx_omega, idx_mp);

% Scale Motor Position by Raft Length for the plot (consistent with other plots)
MP_grid_norm       = MP_grid / L_raft_val;
MP_surferbot_norm  = MP_surferbot / L_raft_val;

%% ====================== FIGURE: Phase Space (Asymmetry) ======================
fig1 = figure('Color','w', 'Units','centimeters');
set(fig1, 'PaperUnits', 'centimeters', 'PaperSize', FIGSIZE_CM, ...
    'PaperPosition', [0 0 FIGSIZE_CM]);
set(fig1, 'Position', [5 5 FIGSIZE_CM]);

ax1 = gca;
hold(ax1, 'on');

% Contourf Plot
n_fill_levels = 50;
contourf(ax1, O_grid_hz, MP_grid_norm, asymmetry_factor, n_fill_levels, ...
    'LineStyle', 'none');

% Surferbot Dot
scatter(ax1, Omega_surferbot_hz, MP_surferbot_norm, 120, ...
    'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1, ...
    'DisplayName', 'Surferbot');

% Styling
xlabel(ax1, 'Frequency (Hz)', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);
ylabel(ax1, 'Motor Position / Raft Length', 'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS);

caxis(ax1, [-1 1]);
colormap(ax1, bwr_colormap());
cb = colorbar(ax1);
cb.Label.String   = 'Asymmetry Factor';
cb.Label.FontName = BASE_FONT;
cb.Label.FontSize = FONT_SIZE_AXIS;

% Add label for Surferbot
text(ax1, Omega_surferbot_hz, MP_surferbot_norm*1.05, ' Surferbot', ...
    'FontName', BASE_FONT, 'FontSize', FONT_SIZE_AXIS, ...
    'Color', 'black', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold');

style_axes(ax1, BASE_FONT, GRID_ALPHA);

if export
    print(fig1, fullfile(saveDir, 'plot_sweep_omega_motorPosition.pdf'), '-dpdf','-painters','-r300');
    print(fig1, fullfile(saveDir, 'plot_sweep_omega_motorPosition.svg'), '-dsvg','-r300');
end

end

% ====================== Helper Functions ======================
function style_axes(ax, baseFont, gridAlpha)
set(ax,'FontName',baseFont,'FontSize',15,'LineWidth',0.75,...
    'TickDir','out','Box','on'); % Box on for complete border
ax.GridAlpha      = gridAlpha;
ax.MinorGridAlpha = gridAlpha;
set(ax,'XMinorTick','on','YMinorTick','on');
grid(ax, 'on');
xlim(ax, 'tight');
ylim(ax, 'tight');
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
