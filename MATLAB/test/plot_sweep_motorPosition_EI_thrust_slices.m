function plot_sweep_motorPosition_EI_thrust_slices(saveDir, dataFile, mpTargets)
%PLOT_SWEEP_MOTORPOSITION_EI_THRUST_SLICES Plot thrust vs EI at fixed x_M/L.
%
% This is a slice plot extracted from the precomputed motorPosition-EI sweep.
% Each curve corresponds to one motor-position slice x_M/L and shows
% thrust_N as a function of EI.

if nargin < 1, saveDir = 'data'; end
if nargin < 2 || isempty(dataFile), dataFile = 'sweepMotorPositionEI.mat'; end

D = load(fullfile(saveDir, dataFile));
S = D.S;
if istable(S), S = table2struct(S); end

L_raft = S(1).args.L_raft;
mp_list = unique([S.motor_position]) / L_raft;
EI_list = unique([S.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);

EI_grid = reshape([S.EI], n_mp, n_EI);
MP_grid = reshape([S.motor_position], n_mp, n_EI) / L_raft;
thrust_grid = reshape([S.thrust_N], n_mp, n_EI);

if nargin < 3 || isempty(mpTargets)
    mpTargets = [0.10 0.25 0.40];
end

idx_mp = zeros(size(mpTargets));
for i = 1:numel(mpTargets)
    [~, idx_mp(i)] = min(abs(mp_list - mpTargets(i)));
end
idx_mp = unique(idx_mp, 'stable');

n_plots = numel(idx_mp);
fig_width = 24;
fig_height = max(5.8 * n_plots, 16);
colors = [0.00 0.35 0.80;
          0.85 0.33 0.10;
          0.00 0.55 0.45;
          0.55 0.25 0.70];

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2 2 fig_width fig_height]);
t = tiledlayout(fig, n_plots, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:n_plots
    idx = idx_mp(i);
    ax = nexttile(t, i);
    hold(ax, 'on');

    plot(ax, EI_grid(idx, :), thrust_grid(idx, :), ...
        'Color', colors(mod(i - 1, size(colors, 1)) + 1, :), ...
        'LineWidth', 2.2, ...
        'LineStyle', '-');

    set(ax, 'XScale', 'log');
    ylabel(ax, 'Thrust (N/m)', 'FontSize', 18);
    title(ax, sprintf('x_M / L = %.3f', MP_grid(idx, 1)), 'FontSize', 19, ...
        'FontWeight', 'normal');
    set(ax, 'FontSize', 17, 'LineWidth', 1.0, 'TickDir', 'out', 'Box', 'on');
    grid(ax, 'on');

    if i < n_plots
        set(ax, 'XTickLabel', []);
    else
        xlabel(ax, 'EI (N m^4)', 'FontSize', 18);
    end
end

sgtitle(t, 'Thrust vs EI slices at fixed motor position', 'FontSize', 20, ...
    'FontWeight', 'normal');

outname = fullfile(saveDir, 'motorPositionEI_thrust_slices.pdf');
outname_png = fullfile(saveDir, 'motorPositionEI_thrust_slices.png');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [fig_width fig_height], ...
    'PaperPosition', [0 0 fig_width fig_height]);
print(fig, outname, '-dpdf', '-painters', '-r300');
print(fig, outname_png, '-dpng', '-r200');
fprintf('Saved %s\n', outname);
end
