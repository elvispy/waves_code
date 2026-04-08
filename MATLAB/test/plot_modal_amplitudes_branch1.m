function plot_modal_amplitudes_branch1(saveDir)
% Plot modal amplitudes |q_n| vs EI along Branch 1 of the second family.
%
% Branch 1 is the main second-family curve (lowest xM/L at each EI).
% We interpolate to ~50 points on a log-EI grid, run the solver at each,
% decompose into free-free beam modes, and plot |q_n| for n=0..10.

if nargin < 1, saveDir = 'data'; end
addpath('../src');

%% ---- Load pre-computed CSV ----
csvFile = fullfile(saveDir, 'modal_amplitudes_branch1.csv');
Tc = readtable(csvFile);

EI_grid = Tc.EI.';
xM_grid = Tc.xM_L.';
n_pts   = numel(EI_grid);
n_modes = 12;

q_mat = NaN(n_modes, n_pts);
for n = 0:10
    col = sprintf('mode%d', n);
    q_mat(n+1, :) = Tc.(col).';
end

% Mode type labels (rigid vs elastic) from free-free beam:
%   modes 0,1 = rigid body; 2+ = elastic
mode_types = cell(1, n_modes);
for n = 0:(n_modes-1)
    if n < 2
        mode_types{n+1} = 'rigid';
    else
        mode_types{n+1} = 'elastic';
    end
end

fprintf('Loaded %d points from %s\n', n_pts, csvFile);

colors = lines(11);

%% ---- Plot vs EI ----
fig1 = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 10 6]);
hold on;
for n = 0:10
    idx = n + 1;
    amp = abs(q_mat(idx, :));
    valid = ~isnan(amp);
    if ~any(valid), continue; end
    lw = 0.5 + n * 0.25;
    lbl = sprintf('Mode %d (%s)', n, mode_types{idx});
    semilogx(EI_grid(valid), amp(valid), '-o', 'Color', colors(n+1,:), ...
        'MarkerFaceColor', colors(n+1,:), 'MarkerSize', 4, 'LineWidth', lw, 'DisplayName', lbl);
end
xlabel('EI (N·m²)', 'FontSize', 12);
ylabel('|q_n|  (modal amplitude)', 'FontSize', 12);
title('Modal amplitudes along Branch 1 of second family', 'FontSize', 12);
legend('Location', 'southwest', 'FontSize', 9);
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 11);
grid on; box on;
xlim([min(EI_grid)*0.7, max(EI_grid)*1.5]);

figfile = fullfile(saveDir, 'modal_amplitudes_branch1');
print(fig1, [figfile '.png'], '-dpng', '-r150');
print(fig1, [figfile '.svg'], '-dsvg');
saveas(fig1,  [figfile '.fig']);
fprintf('Figure saved to %s.{png,svg,fig}\n', figfile);

%% ---- Plot vs xM/L ----
fig2 = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 10 6]);
hold on;
for n = 0:10
    idx = n + 1;
    amp = abs(q_mat(idx, :));
    valid = ~isnan(amp);
    if ~any(valid), continue; end
    lw = 0.5 + n * 0.25;
    lbl = sprintf('Mode %d (%s)', n, mode_types{idx});
    semilogy(xM_grid(valid), amp(valid), '-o', 'Color', colors(n+1,:), ...
        'MarkerFaceColor', colors(n+1,:), 'MarkerSize', 4, 'LineWidth', lw, 'DisplayName', lbl);
end
xlabel('x_M / L', 'FontSize', 12);
ylabel('|q_n|  (modal amplitude)', 'FontSize', 12);
title('Modal amplitudes along Branch 1 of second family', 'FontSize', 12);
legend('Location', 'southwest', 'FontSize', 9);
set(gca, 'YScale', 'log', 'FontSize', 11);
grid on; box on;

figfile2 = fullfile(saveDir, 'modal_amplitudes_branch1_vs_xM');
print(fig2, [figfile2 '.png'], '-dpng', '-r150');
print(fig2, [figfile2 '.svg'], '-dsvg');
saveas(fig2,  [figfile2 '.fig']);
fprintf('Figure saved to %s.{png,svg,fig}\n', figfile2);


end
