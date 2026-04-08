addpath('../src');

for ds = 1:2
    if ds == 1
        D = load('data/sweepMotorPositionEI.mat');
        label = 'd = 0.03 (coupled)';
    else
        D = load('data/sweepMotorPositionEI2.mat');
        label = 'd = 0 (uncoupled)';
    end
    S = D.S;
    if istable(S), S = table2struct(S); end

    L_raft = S(1).args.L_raft;
    mp_list = unique([S.motor_position]);
    EI_list = unique([S.EI]);
    n_mp = numel(mp_list);
    n_EI = numel(EI_list);

    EI_grid = reshape([S.EI], n_mp, n_EI);
    MP_grid = reshape([S.motor_position], n_mp, n_EI) / L_raft;

    eta_1_sq   = abs(reshape([S.eta_1],   n_mp, n_EI)).^2;
    eta_end_sq = abs(reshape([S.eta_end], n_mp, n_EI)).^2;
    asymm = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);

    fig = figure('Color','w','Units','centimeters','Position',[2+20*(ds-1) 2 20 16]);
    ax = gca; hold(ax,'on');
    contourf(ax, EI_grid, MP_grid, asymm, 50, 'LineStyle','none');
    contour(ax, EI_grid, MP_grid, asymm, [0 0], 'LineColor','k','LineWidth',2);
    set(ax,'XScale','log');
    caxis([-1 1]); colormap(ax, bwr_colormap()); colorbar;
    xlabel('EI (N m^4)'); ylabel('Motor Position / Raft Length');
    title(label, 'FontSize', 14);
    set(ax,'FontSize',13,'LineWidth',0.75,'TickDir','out','Box','on');
    grid on;

    outname = sprintf('data/motorPositionEI_d%d.pdf', ds);
    set(fig,'PaperUnits','centimeters','PaperSize',[20 16],'PaperPosition',[0 0 20 16]);
    print(fig, outname, '-dpdf','-painters','-r300');
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
