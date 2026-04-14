function inviscid_test_postprocess(saveDir)
%INVISCID_TEST_POSTPROCESS Compare viscous and inviscid coarse sweep results.

if nargin < 1, saveDir = 'data'; end

D = load(fullfile(saveDir, 'inviscid_test_sweep.mat'));
results = D.results;

omega_list = results.omega_list;
motor_position_list = results.motor_position_list / results.base.L_raft;
EI_list = results.EI_list;

visc = results.viscous;
inv = results.inviscid;

Sxx_visc = reshape([visc.Sxx], size(visc));
Sxx_inv = reshape([inv.Sxx], size(inv));
alpha_visc = reshape([visc.alpha_beam], size(visc));
alpha_inv = reshape([inv.alpha_beam], size(inv));
etaL_visc = abs(reshape([visc.eta_left_beam], size(visc)));
etaL_inv = abs(reshape([inv.eta_left_beam], size(inv)));
etaR_visc = abs(reshape([visc.eta_right_beam], size(visc)));
etaR_inv = abs(reshape([inv.eta_right_beam], size(inv)));

rel_Sxx = relative_change(Sxx_visc, Sxx_inv, 1e-6 * max(abs([Sxx_visc(:); Sxx_inv(:)])));
rel_alpha = relative_change(alpha_visc, alpha_inv, 0.05);
rel_etaL = relative_change(etaL_visc, etaL_inv, 1e-6 * max([etaL_visc(:); etaL_inv(:)]));
rel_etaR = relative_change(etaR_visc, etaR_inv, 1e-6 * max([etaR_visc(:); etaR_inv(:)]));
rel_eta_beam = max(rel_etaL, rel_etaR);

summary_names = {'rel_Sxx'; 'rel_alpha_beam'; 'rel_eta_beam'};
summary_max = [max(rel_Sxx(:)); max(rel_alpha(:)); max(rel_eta_beam(:))];
summary_median = [median(rel_Sxx(:)); median(rel_alpha(:)); median(rel_eta_beam(:))];
summary_p90 = [prctile(rel_Sxx(:), 90); prctile(rel_alpha(:), 90); prctile(rel_eta_beam(:), 90)];

T = table(summary_names, summary_max, summary_median, summary_p90, ...
    'VariableNames', {'metric', 'max_rel_change', 'median_rel_change', 'p90_rel_change'});
csv_path = fullfile(saveDir, 'inviscid_test_summary.csv');
writetable(T, csv_path);
disp(T);

plot_omega_slices(EI_list, motor_position_list, omega_list, rel_Sxx, ...
    'Relative change in S_{xx}', fullfile(saveDir, 'inviscid_test_rel_Sxx.pdf'));
plot_omega_slices(EI_list, motor_position_list, omega_list, rel_alpha, ...
    'Relative change in \alpha_{beam}', fullfile(saveDir, 'inviscid_test_rel_alpha_beam.pdf'));
plot_omega_slices(EI_list, motor_position_list, omega_list, rel_eta_beam, ...
    'Relative change in beam-end amplitudes', fullfile(saveDir, 'inviscid_test_rel_eta_beam.pdf'));

fprintf('Saved %s\n', csv_path);
end

function rel = relative_change(a, b, floor_value)
if nargin < 3 || isempty(floor_value)
    floor_value = 1e-12;
end
den = max(max(abs(a), abs(b)), floor_value);
rel = abs(a - b) ./ den;
end

function plot_omega_slices(EI_list, mp_list, omega_list, Z, fig_title, outfile)
n_omega = numel(omega_list);
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 32 18]);
t = tiledlayout(fig, 2, ceil(n_omega / 2), 'TileSpacing', 'compact', 'Padding', 'compact');

for iw = 1:n_omega
    ax = nexttile(t, iw);
    contourf(ax, EI_list, mp_list, squeeze(Z(iw, :, :)), 40, 'LineStyle', 'none');
    set(ax, 'XScale', 'log');
    xlabel(ax, 'EI (N m^4)');
    ylabel(ax, 'x_M / L');
    title(ax, sprintf('f = %.1f Hz', omega_list(iw) / (2 * pi)));
    set(ax, 'FontSize', 11, 'LineWidth', 0.75, 'TickDir', 'out', 'Box', 'on');
    colorbar(ax);
end

sgtitle(t, fig_title, 'FontSize', 16, 'FontWeight', 'normal');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [32 18], 'PaperPosition', [0 0 32 18]);
print(fig, outfile, '-dpdf', '-painters', '-r300');
fprintf('Saved %s\n', outfile);
end
