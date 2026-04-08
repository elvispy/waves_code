function plot_all_families(saveDir)
% Plot ALL zero-crossings of asymmetry, colored by family and branch.
%
% Family classification:
%   SA_ratio = log10(|S|/|A|):
%     > 0 → first family (|S| > |A|)
%     < 0 → second family (|S| < |A|)
%
% Within second family, crossings are ordered by xM at each EI to give
% branches 1, 2, 3, ...

if nargin < 1, saveDir = 'data'; end
addpath('../src');

Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L = St(1).args.L_raft;
mp  = unique([St.motor_position]);
EI  = unique([St.EI]);
n_mp = numel(mp); n_EI = numel(EI);

eta_1   = reshape([St.eta_1],   n_mp, n_EI);
eta_end = reshape([St.eta_end], n_mp, n_EI);

asym  = -(abs(eta_1).^2 - abs(eta_end).^2) ./ (abs(eta_1).^2 + abs(eta_end).^2);
SA    = log10(abs((eta_end+eta_1)/2) ./ (abs((eta_end-eta_1)/2) + eps));
MP    = mp / L;

%% ---- Collect all crossings ----
all_EI  = [];
all_xM  = [];
all_SA  = [];

for ie = 1:n_EI
    col    = asym(:, ie);
    sa_col = SA(:, ie);
    for im = 1:(n_mp-1)
        if col(im) * col(im+1) < 0
            t   = col(im) / (col(im) - col(im+1));
            xc  = MP(im) + t*(MP(im+1) - MP(im));
            sac = sa_col(im) + t*(sa_col(im+1) - sa_col(im));
            all_EI(end+1) = EI(ie);  %#ok<AGROW>
            all_xM(end+1) = xc;      %#ok<AGROW>
            all_SA(end+1) = sac;     %#ok<AGROW>
        end
    end
end

%% ---- Assign branch numbers by xM ordering at each EI (no SA filter) ----
[EI_u, ~, ic] = unique(all_EI);
branch = zeros(size(all_EI));
for k = 1:numel(EI_u)
    idx = find(ic == k);
    [~, ord] = sort(all_xM(idx));
    for b = 1:numel(idx)
        branch(idx(ord(b))) = b;
    end
end

max_branch = max(branch);
fprintf('Total branches found: %d\n', max_branch);
for b = 1:max_branch
    sel = branch == b;
    fprintf('  Branch %d: %d points, EI=[%.2e,%.2e], xM/L=[%.3f,%.3f], SA=[%.2f,%.2f]\n', ...
        b, sum(sel), min(all_EI(sel)), max(all_EI(sel)), ...
        min(all_xM(sel)), max(all_xM(sel)), min(all_SA(sel)), max(all_SA(sel)));
end

%% ---- Save CSV ----
T = table(all_EI(:), all_xM(:), all_SA(:), branch(:), ...
    'VariableNames', {'EI','xM_L','SA_ratio','branch'});
writetable(T, fullfile(saveDir, 'all_crossings_classified.csv'));
fprintf('Saved all_crossings_classified.csv\n');

%% ---- Figure ----
fig = figure('Visible','off','Units','inches','Position',[0 0 8 5]);

branch_colors = [0.12 0.47 0.71;   % blue   (branch 1)
                 0.89 0.10 0.11;   % red    (branch 2)
                 0.20 0.63 0.17;   % green  (branch 3)
                 0.99 0.60 0.05];  % orange (branch 4)

hold on;
for b = 1:max_branch
    sel = branch == b;
    clr = branch_colors(min(b, size(branch_colors,1)), :);
    semilogx(all_EI(sel), all_xM(sel), 'o', ...
        'Color', clr, 'MarkerFaceColor', clr, 'MarkerSize', 7, ...
        'DisplayName', sprintf('Branch %d', b));
end

xlabel('EI (N·m²)', 'FontSize', 12);
ylabel('x_M / L', 'FontSize', 12);
title('Asymmetry \alpha=0 crossings: all branches by x_M order', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 10);
set(gca, 'XScale', 'log', 'FontSize', 11);
grid on; box on;
xlim([min(all_EI)*0.7, max(all_EI)*1.5]);
ylim([0, 0.52]);

% Save
figfile = fullfile(saveDir, 'all_families_branches');
print(fig, [figfile '.svg'], '-dsvg');
print(fig, [figfile '.png'], '-dpng', '-r150');
saveas(fig, [figfile '.fig']);
fprintf('Figure saved to %s.{svg,png,fig}\n', figfile);

end
