function plot_sweep_motorPosition_EI_beam_end(saveDir)
%PLOT_SWEEP_MOTORPOSITION_EI_BEAM_END Plot the x_M-EI asymmetry figure
%using beam-end amplitudes from the coupled sweep dataset.

if nargin < 1, saveDir = 'data'; end

D = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
S = D.S;
if istable(S), S = table2struct(S); end

required = {'eta_left_beam','eta_right_beam'};
for i = 1:numel(required)
    if ~isfield(S, required{i})
        error('plot_sweep_motorPosition_EI_beam_end:MissingField', ...
            'Field %s is missing. Regenerate the coupled sweep with the updated sweep script.', ...
            required{i});
    end
end

L_raft = S(1).args.L_raft;
mp_list = unique([S.motor_position]);
EI_list = unique([S.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);

EI_grid = reshape([S.EI], n_mp, n_EI);
MP_grid = reshape([S.motor_position], n_mp, n_EI) / L_raft;

eta_left_sq = abs(reshape([S.eta_left_beam], n_mp, n_EI)).^2;
eta_right_sq = abs(reshape([S.eta_right_beam], n_mp, n_EI)).^2;
asymm = -(eta_left_sq - eta_right_sq) ./ (eta_left_sq + eta_right_sq);

fig = figure('Color','w','Units','centimeters','Position',[2 2 20 16]);
ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, asymm, 50, 'LineStyle','none');
% contour(ax, EI_grid, MP_grid, asymm, [0 0], 'LineColor','k', 'LineWidth', 2);
set(ax, 'XScale', 'log');
caxis(ax, [-1 1]);
colormap(ax, bwr_colormap());
colorbar(ax);
xlabel(ax, 'EI (N m^4)');
ylabel(ax, 'Motor Position / Raft Length');
title(ax, 'd = 0.03 (coupled), beam-end asymmetry', 'FontSize', 14);
set(ax, 'FontSize', 13, 'LineWidth', 0.75, 'TickDir', 'out', 'Box', 'on');
grid(ax, 'on');

outname = fullfile(saveDir, 'motorPositionEI_beam_end.pdf');
outname_png = fullfile(saveDir, 'motorPositionEI_beam_end.png');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [20 16], 'PaperPosition', [0 0 20 16]);
print(fig, outname, '-dpdf', '-painters', '-r300');
print(fig, outname_png, '-dpng', '-r200');
fprintf('Saved %s\n', outname);
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
