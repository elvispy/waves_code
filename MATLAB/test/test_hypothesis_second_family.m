function test_hypothesis_second_family(saveDir)
% Test hypothesis: along second-family segments, either |S|/|A| << 1
% or |S|/|A| >> 1 consistently (not phase matching).
%
% Strategy: compute |S|/|A| on the full grid, then overlay the alpha=0
% contour. Classify each contour SEGMENT (between intersection points)
% as "vertical" (first family) or "non-vertical" (second family) by its
% local slope in (log10(EI), x_M/L) space. Then check |S|/|A| along
% second-family segments.

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

eta_1   = reshape([St.eta_1],   n_mp, n_EI);
eta_end = reshape([St.eta_end], n_mp, n_EI);

S_grid = (eta_end + eta_1) / 2;
A_grid = (eta_end - eta_1) / 2;

eta_1_sq   = abs(eta_1).^2;
eta_end_sq = abs(eta_end).^2;
asymmetry  = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

SA_ratio = log10(abs(S_grid) ./ (abs(A_grid) + eps));
MP_norm_list = mp_list / L_raft;

% ---- Extract contour and split at intersection points ----
C = contourc(EI_list, MP_norm_list, asymmetry, [0 0]);
raw_curves = {};
idx = 1;
while idx < size(C, 2)
    npts = C(2, idx);
    cx = C(1, (idx+1):(idx+npts));
    cy = C(2, (idx+1):(idx+npts));
    raw_curves{end+1} = struct('EI', cx(:), 'mp', cy(:)); %#ok<AGROW>
    idx = idx + npts + 1;
end

% For each raw contour, split into segments classified by local slope.
% In (log10(EI), x_M/L) space:
%   - vertical family: |d(x_M)/d(log10(EI))| >> 1 (large mp change per EI decade)
%     Actually vertical in original space means constant EI, so d(log10(EI))/d(x_M) ~ 0
%     i.e., slope = d(mp) / d(log10(EI)) -> infinity
%   - second family: finite slope, EI changes significantly
%
% Classify each point by local slope, then split into runs of same class.

all_segments = struct('EI',{},'mp',{},'family',{},'SA_ratio',{});

for ic = 1:numel(raw_curves)
    ei = raw_curves{ic}.EI;
    mp = raw_curves{ic}.mp;
    n = numel(ei);
    if n < 3, continue; end

    % Local slope: d(mp) / d(log10(EI))
    dlogEI = diff(log10(ei));
    dmp    = diff(mp);

    % Classify each segment between consecutive points
    % "vertical" if |dlogEI| < threshold (EI barely changes)
    % "non-vertical" otherwise
    % Use threshold on the ratio: if mp changes more than EI (in log), it's vertical
    is_vertical = abs(dlogEI) < 0.05;  % less than 0.05 decades of EI change between adjacent pts

    % Split into runs of same classification
    run_start = 1;
    for ip = 2:numel(is_vertical)
        if is_vertical(ip) ~= is_vertical(run_start) || ip == numel(is_vertical)
            % End of a run - save segment
            if ip == numel(is_vertical)
                run_end = ip + 1;  % include last point
            else
                run_end = ip;
            end

            seg_ei = ei(run_start:run_end);
            seg_mp = mp(run_start:run_end);

            if numel(seg_ei) >= 2
                % Interpolate |S|/|A| along segment
                seg_SA = zeros(size(seg_ei));
                for k = 1:numel(seg_ei)
                    seg_SA(k) = interp2(EI_list, MP_norm_list, SA_ratio, ...
                        seg_ei(k), seg_mp(k), 'linear', NaN);
                end

                fam = 'V';
                if ~is_vertical(run_start), fam = 'NV'; end

                s = struct('EI', seg_ei, 'mp', seg_mp, ...
                    'family', fam, 'SA_ratio', seg_SA);
                all_segments(end+1) = s; %#ok<AGROW>
            end

            run_start = ip;
        end
    end
end

% ---- Report: for each non-vertical segment, check consistency of |S|/|A| ----
fprintf('\n=== Non-vertical (second family) segments ===\n');
fprintf('%-4s | %-4s | %-12s | %-10s | %-10s | %-10s | %-10s | %-8s\n', ...
    '#', 'Npts', 'EI range', 'mp range', 'mean SA', 'std SA', 'min SA', 'max SA');
fprintf('%s\n', repmat('-', 1, 90));

nv_count = 0;
consistent_count = 0;
for is = 1:numel(all_segments)
    seg = all_segments(is);
    if ~strcmp(seg.family, 'NV'), continue; end
    if numel(seg.SA_ratio) < 2, continue; end

    valid = ~isnan(seg.SA_ratio);
    if sum(valid) < 2, continue; end

    sa = seg.SA_ratio(valid);
    nv_count = nv_count + 1;

    % Check consistency: are all values either > 0 (|S|>|A|) or < 0 (|S|<|A|)?
    all_positive = all(sa > 0);
    all_negative = all(sa < 0);
    consistent = all_positive || all_negative;
    if consistent, consistent_count = consistent_count + 1; end

    tag = '';
    if all_positive, tag = ' [A~0]'; end
    if all_negative, tag = ' [S~0]'; end
    if ~consistent, tag = ' [MIXED]'; end

    fprintf('%3d  | %3d  | [%.1e,%.1e] | [%.2f,%.2f] | %+8.2f   | %8.2f   | %+8.2f   | %+8.2f  %s\n', ...
        nv_count, sum(valid), ...
        min(seg.EI), max(seg.EI), min(seg.mp), max(seg.mp), ...
        mean(sa), std(sa), min(sa), max(sa), tag);
end

fprintf('\nSummary: %d/%d non-vertical segments have consistent sign of log10(|S|/|A|)\n', ...
    consistent_count, nv_count);

% ---- Plot: zoom into one non-vertical segment to visualize ----
fig = figure('Color','w','Units','centimeters','Position',[2 2 28 14]);

% Left: full phase space with segments colored by family
subplot(1,2,1); ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, SA_ratio, linspace(-2,2,50), 'LineStyle','none');
contour(ax, EI_grid, MP_grid, asymmetry, [0 0], 'LineColor',[0.5 0.5 0.5],'LineWidth',0.5);
set(ax,'XScale','log'); caxis([-2 2]); colorbar;

for is = 1:numel(all_segments)
    seg = all_segments(is);
    if strcmp(seg.family, 'V')
        plot(ax, seg.EI, seg.mp, 'k-', 'LineWidth', 2);
    else
        plot(ax, seg.EI, seg.mp, 'r-', 'LineWidth', 2.5);
    end
end
xlabel('EI'); ylabel('x_M / L');
title('log_{10}(|S|/|A|) with V (black) and NV (red) segments');
set(ax,'FontSize',11);

% Right: |S|/|A| along each non-vertical segment
subplot(1,2,2); ax = gca; hold(ax,'on');
nv_idx = 0;
for is = 1:numel(all_segments)
    seg = all_segments(is);
    if ~strcmp(seg.family, 'NV'), continue; end
    if numel(seg.SA_ratio) < 2, continue; end
    nv_idx = nv_idx + 1;
    valid = ~isnan(seg.SA_ratio);
    % parametric distance along segment
    t = cumsum([0; sqrt(diff(log10(seg.EI(valid))).^2 + diff(seg.mp(valid)).^2)]);
    t = t / max(t);
    plot(ax, t, seg.SA_ratio(valid), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Seg %d', nv_idx));
end
yline(0, 'k--', 'LineWidth', 1);
xlabel('Parametric distance along segment'); ylabel('log_{10}(|S|/|A|)');
title('|S|/|A| along non-vertical segments');
legend('show','Location','best','FontSize',8); grid on;
set(ax,'FontSize',11);

sgtitle('Hypothesis test: consistency of |S|/|A| on second-family segments', 'FontSize',13);

outfile = fullfile(saveDir, 'hypothesis_test_second_family.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[28 14],'PaperPosition',[0 0 28 14]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);

end
