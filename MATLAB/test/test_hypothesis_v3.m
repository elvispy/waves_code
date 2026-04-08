function test_hypothesis_v3(saveDir)
% Extract alpha=0 crossings column-by-column (fixed EI, scan x_M).
% Classify each crossing as vertical-family or second-family.
% Then check |S|/|A| on second-family crossings.

if nargin < 1, saveDir = 'data'; end
addpath('../src');

Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L_raft = St(1).args.L_raft;
mp_list = unique([St.motor_position]);
EI_list = unique([St.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);

EI_grid = reshape([St.EI], n_mp, n_EI);
MP_grid = reshape([St.motor_position], n_mp, n_EI) / L_raft;
MP_norm_list = mp_list / L_raft;

eta_1   = reshape([St.eta_1],   n_mp, n_EI);
eta_end = reshape([St.eta_end], n_mp, n_EI);

S_grid = (eta_end + eta_1) / 2;
A_grid = (eta_end - eta_1) / 2;

eta_1_sq   = abs(eta_1).^2;
eta_end_sq = abs(eta_end).^2;
asymmetry  = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

SA_ratio = log10(abs(S_grid) ./ (abs(A_grid) + eps));

% ---- Step 1: For each EI column, find all x_M zero crossings ----
% Store as (EI_val, x_M_val, SA_ratio_val) triplets
all_EI = [];
all_mp = [];
all_SA = [];

for ie = 1:n_EI
    col = asymmetry(:, ie);
    for im = 1:(n_mp-1)
        if col(im) * col(im+1) < 0  % sign change
            % Linear interpolation for zero crossing
            t = col(im) / (col(im) - col(im+1));
            mp_zero = MP_norm_list(im) + t * (MP_norm_list(im+1) - MP_norm_list(im));
            sa_zero = SA_ratio(im, ie) + t * (SA_ratio(im+1, ie) - SA_ratio(im, ie));

            all_EI(end+1) = EI_list(ie); %#ok<AGROW>
            all_mp(end+1) = mp_zero; %#ok<AGROW>
            all_SA(end+1) = sa_zero; %#ok<AGROW>
        end
    end
end

fprintf('Found %d zero crossings total\n', numel(all_EI));

% ---- Step 2: Identify vertical-family EI values ----
% Vertical family: at resonance EI, alpha=0 for MANY x_M values.
% Count crossings per EI column.
[unique_EI, ~, ei_idx] = unique(all_EI);
crossings_per_EI = accumarray(ei_idx, 1);

% Vertical lines have many crossings (spanning most of the x_M range)
% Second family: typically 1 crossing per EI column per curve
% Threshold: if an EI value has >= 5 crossings, it's near a vertical resonance
vert_thresh = 5;
is_vert_EI = crossings_per_EI >= vert_thresh;
vert_EI_vals = unique_EI(is_vert_EI);

fprintf('\nVertical-family EI values (>= %d crossings):\n', vert_thresh);
for iv = 1:numel(vert_EI_vals)
    nc = crossings_per_EI(unique_EI == vert_EI_vals(iv));
    fprintf('  EI = %.3e (%d crossings)\n', vert_EI_vals(iv), nc);
end

% ---- Step 3: Classify each crossing ----
% A crossing is "near vertical" if its EI is within 0.15 decades of a vert_EI_val
vert_proximity = 0.15; % decades
is_vertical = false(size(all_EI));
for iv = 1:numel(vert_EI_vals)
    dist = abs(log10(all_EI) - log10(vert_EI_vals(iv)));
    is_vertical = is_vertical | (dist < vert_proximity);
end

n_vert = sum(is_vertical);
n_second = sum(~is_vertical);
fprintf('\nClassified: %d vertical, %d second-family crossings\n', n_vert, n_second);

% ---- Step 4: Report |S|/|A| statistics for second-family crossings ----
sf_SA = all_SA(~is_vertical);
sf_EI = all_EI(~is_vertical);
sf_mp = all_mp(~is_vertical);

fprintf('\n=== Second-family crossings: log10(|S|/|A|) ===\n');
fprintf('  Mean:   %+.2f\n', mean(sf_SA));
fprintf('  Median: %+.2f\n', median(sf_SA));
fprintf('  Std:    %.2f\n', std(sf_SA));
fprintf('  Min:    %+.2f\n', min(sf_SA));
fprintf('  Max:    %+.2f\n', max(sf_SA));
fprintf('  Fraction with |S|/|A| < 1 (S~0): %.1f%%\n', 100*mean(sf_SA < 0));
fprintf('  Fraction with |S|/|A| > 1 (A~0): %.1f%%\n', 100*mean(sf_SA > 0));

% Now group by EI to see if individual second-family curves are consistent
% Sort by EI
[sf_EI_sorted, sort_idx] = sort(sf_EI);
sf_mp_sorted = sf_mp(sort_idx);
sf_SA_sorted = sf_SA(sort_idx);

fprintf('\n=== Second-family crossings sorted by EI ===\n');
fprintf('%-12s | %-8s | %-10s | %-6s\n', 'EI', 'x_M/L', 'log|S|/|A|', 'type');
fprintf('%s\n', repmat('-', 1, 48));
for k = 1:numel(sf_EI_sorted)
    tag = 'S~0';
    if sf_SA_sorted(k) > 0, tag = 'A~0'; end
    fprintf('%11.3e | %7.4f  | %+9.2f   | %s\n', ...
        sf_EI_sorted(k), sf_mp_sorted(k), sf_SA_sorted(k), tag);
end

% ---- Step 5: Plot ----
fig = figure('Color','w','Units','centimeters','Position',[1 1 28 20]);

% Top: phase space with crossings colored by |S|/|A|
subplot(2,1,1); ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, asymmetry, 50, 'LineStyle','none');
caxis([-1 1]); colormap(ax, bwr_colormap());

% Vertical crossings: gray
scatter(ax, all_EI(is_vertical), all_mp(is_vertical), 30, ...
    [0.5 0.5 0.5], 'filled', 'DisplayName', 'Vertical family');

% Second-family crossings: colored by sign of SA
sf_pos = ~is_vertical & all_SA > 0;  % A~0
sf_neg = ~is_vertical & all_SA < 0;  % S~0
scatter(ax, all_EI(sf_neg), all_mp(sf_neg), 50, ...
    'b', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'S \approx 0');
scatter(ax, all_EI(sf_pos), all_mp(sf_pos), 50, ...
    'r', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'A \approx 0');

set(ax,'XScale','log');
xlabel('EI'); ylabel('x_M / L');
title('Zero crossings: gray=vertical, blue=S\approx0, red=A\approx0');
legend('show','Location','northeast','FontSize',10);
set(ax,'FontSize',12);

% Bottom: histogram of |S|/|A| for second-family crossings
subplot(2,1,2); ax = gca; hold(ax,'on');
histogram(ax, sf_SA, linspace(-3,4,30), 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'w');
xline(0, 'r--', 'LineWidth', 2);
xlabel('log_{10}(|S|/|A|)'); ylabel('Count');
title('Distribution of log_{10}(|S|/|A|) on second-family crossings');
set(ax,'FontSize',12); grid on;

sgtitle('Hypothesis test v3: column-by-column zero crossing extraction', 'FontSize',13,'FontWeight','bold');

outfile = fullfile(saveDir, 'hypothesis_test_v3.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[28 20],'PaperPosition',[0 0 28 20]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);

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
