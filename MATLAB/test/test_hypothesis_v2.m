function test_hypothesis_v2(saveDir)
% Robust test: classify WHOLE contourc curves as vertical vs second-family
% by their overall EI span. Then trace |S|/|A| along each second-family curve.

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

% ---- Extract contourc curves ----
C = contourc(EI_list, MP_norm_list, asymmetry, [0 0]);
curves = {};
idx = 1;
while idx < size(C, 2)
    npts = C(2, idx);
    cx = C(1, (idx+1):(idx+npts));
    cy = C(2, (idx+1):(idx+npts));
    curves{end+1} = struct('EI', cx(:), 'mp', cy(:)); %#ok<AGROW>
    idx = idx + npts + 1;
end

% ---- Classify whole curves by EI span (in decades) ----
% Vertical family: small EI span (< 0.5 decades)
% Second family: large EI span (> 0.5 decades)
ei_span_thresh = 0.5;  % decades

v_curves  = {};  % vertical family
nv_curves = {};  % second family

fprintf('Contourc produced %d curves:\n', numel(curves));
for ic = 1:numel(curves)
    ei_span = log10(max(curves{ic}.EI)) - log10(min(curves{ic}.EI));
    mp_span = max(curves{ic}.mp) - min(curves{ic}.mp);

    if ei_span < ei_span_thresh
        family = 'V';
        v_curves{end+1} = curves{ic}; %#ok<AGROW>
    else
        family = 'NV';
        nv_curves{end+1} = curves{ic}; %#ok<AGROW>
    end

    fprintf('  Curve %d: %3d pts, EI span=%.1f decades, mp span=%.3f -> %s\n', ...
        ic, numel(curves{ic}.EI), ei_span, mp_span, family);
end

fprintf('\nVertical family: %d curves\n', numel(v_curves));
fprintf('Second family:   %d curves\n', numel(nv_curves));

% ---- For each second-family curve, interpolate |S|/|A| along it ----
fig = figure('Color','w','Units','centimeters','Position',[1 1 30 20]);

% Top: phase space with classified curves
subplot(2,2,[1 2]); ax = gca; hold(ax,'on');
contourf(ax, EI_grid, MP_grid, SA_ratio, linspace(-2,2,50), 'LineStyle','none');
caxis([-2 2]);
cb = colorbar; cb.Label.String = 'log_{10}(|S|/|A|)';
set(ax,'XScale','log');

% Plot vertical family in black
for ic = 1:numel(v_curves)
    plot(ax, v_curves{ic}.EI, v_curves{ic}.mp, 'k-', 'LineWidth', 2.5);
end
% Plot second family in distinct colors
nv_colors = lines(numel(nv_curves));
for ic = 1:numel(nv_curves)
    plot(ax, nv_curves{ic}.EI, nv_curves{ic}.mp, '-', ...
        'Color', nv_colors(ic,:), 'LineWidth', 2.5);
end

xlabel('EI (N m^4)'); ylabel('x_M / L');
title(sprintf('\\alpha=0 curves: %d vertical (black), %d second-family (colored)', ...
    numel(v_curves), numel(nv_curves)));
set(ax,'FontSize',12);
legend_entries = cell(numel(nv_curves),1);
for ic = 1:numel(nv_curves)
    legend_entries{ic} = sprintf('NV curve %d', ic);
end

% Bottom-left: |S|/|A| along each second-family curve vs parametric distance
subplot(2,2,3); ax = gca; hold(ax,'on');
for ic = 1:numel(nv_curves)
    c = nv_curves{ic};
    n = numel(c.EI);

    % Parametric distance in (log10(EI), mp) space
    t = cumsum([0; sqrt(diff(log10(c.EI)).^2 + diff(c.mp).^2)]);
    t = t / max(t);

    % Interpolate SA_ratio along curve
    sa = zeros(n,1);
    for k = 1:n
        sa(k) = interp2(EI_list, MP_norm_list, SA_ratio, c.EI(k), c.mp(k), 'linear', NaN);
    end

    plot(ax, t, sa, 'o-', 'Color', nv_colors(ic,:), 'LineWidth', 1.5, 'MarkerSize', 4, ...
        'MarkerFaceColor', nv_colors(ic,:), 'DisplayName', legend_entries{ic});
end
yline(0, 'k--', 'LineWidth', 1);
xlabel('Parametric distance along curve'); ylabel('log_{10}(|S|/|A|)');
title('|S|/|A| along second-family curves');
legend('show','Location','best','FontSize',9); grid on;
set(ax,'FontSize',11);

% Bottom-right: |S|/|A| along each second-family curve vs log10(EI)
subplot(2,2,4); ax = gca; hold(ax,'on');
for ic = 1:numel(nv_curves)
    c = nv_curves{ic};
    n = numel(c.EI);

    sa = zeros(n,1);
    for k = 1:n
        sa(k) = interp2(EI_list, MP_norm_list, SA_ratio, c.EI(k), c.mp(k), 'linear', NaN);
    end

    plot(ax, log10(c.EI), sa, 'o-', 'Color', nv_colors(ic,:), 'LineWidth', 1.5, 'MarkerSize', 4, ...
        'MarkerFaceColor', nv_colors(ic,:), 'DisplayName', legend_entries{ic});
end

% Mark vertical resonance EI values
for ic = 1:numel(v_curves)
    ei_mid = exp(mean(log(v_curves{ic}.EI)));
    xline(log10(ei_mid), 'k:', 'LineWidth', 1, 'HandleVisibility','off');
end

yline(0, 'k--', 'LineWidth', 1);
xlabel('log_{10}(EI)'); ylabel('log_{10}(|S|/|A|)');
title('|S|/|A| vs EI along second-family curves');
legend('show','Location','best','FontSize',9); grid on;
set(ax,'FontSize',11);

sgtitle('Hypothesis test v2: |S|/|A| along whole second-family curves', 'FontSize',13, 'FontWeight','bold');

outfile = fullfile(saveDir, 'hypothesis_test_v2.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[30 20],'PaperPosition',[0 0 30 20]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);

end
