function plot_sweep_EI(saveDir, export)
% Recreate EI_sweep figures (MATLAB 2018b-compatible, publication-ready)
% Usage:
%   replot_EI_sweep_pub_2018b
%   replot_EI_sweep_pub_2018b('outdir')

if nargin < 1, saveDir = 'data'; end
if ~exist(saveDir,'dir'), mkdir(saveDir); end
if nargin < 2, export = false; end

% ---- Parameters ----
BASE_FONT   = 'Times';
FIGSIZE_CM1 = [18 18.5];
FIGSIZE_CM2 = [20 7.5];
LINE_W      = 1.4;
MK          = 5.5;
GRID_ALPHA  = 0.2;
C = [0.00 0.45 0.70;
     0.85 0.33 0.10;
     0.00 0.60 0.50;
     0.95 0.90 0.25;
     0.80 0.47 0.65;
     0.00 0.00 0.00];

% ---- Load ----
D = load('data/EI_sweep.mat');
S = D.S;

EIsurf = 3.0e9 * 3e-2 * 9e-4^3 / 12;
EI   = S.EI;
EI0  = 1; %EI(1);
xEI  = EI ./ EI0;
Tthr = S.thrust_N;
Ppow = S.power;
E1   = S.eta_1;
Eend = S.eta_end;
symFactor = (E1.^2 - Eend.^2) ./ (E1.^2 + Eend.^2);

% ====================== FIGURE 1 ======================
fig1 = figure('Units','centimeters','Position',[2 2 FIGSIZE_CM1], 'Color','w');

subplot(3,1,1); hold on;
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
yline(0, '-', 'HandleVisibility', 'off', 'LineWidth', 0.5);
plot(xEI, Tthr, '-','LineWidth',LINE_W,'MarkerSize',MK,...
    'Color',C(1,:),'MarkerFaceColor',C(1,:),'MarkerEdgeColor','w', 'HandleVisibility', 'off');
set(gca,'XScale','log'); xlim([min(xEI) max(xEI)]); grid on;

xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',10)
ylabel('Thrust (N/m)','FontName',BASE_FONT,'FontSize',10)
title('Thrust','FontName',BASE_FONT,'FontSize',11)
legend('Location','best','Box','off')
style_axes(gca,BASE_FONT,GRID_ALPHA)

subplot(3,1,2)
plot(xEI, -Ppow, 'o-','LineWidth',LINE_W,'MarkerSize',MK,...
    'Color',C(2,:),'MarkerFaceColor',C(2,:),'MarkerEdgeColor','w', 'HandleVisibility', 'off');
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
set(gca,'XScale','log','YScale','log'); xlim([min(xEI) max(xEI)]); grid on;
xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',10)
ylabel('Power (W)','FontName',BASE_FONT,'FontSize',10)
title('Power','FontName',BASE_FONT,'FontSize',11)
legend('Location','best','Box','off')
style_axes(gca,BASE_FONT,GRID_ALPHA)

subplot(3,1,3); hold on;

yline(0, '-', 'HandleVisibility', 'off', 'LineWidth', 0.5);
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
plot(xEI, symFactor, 'o-','LineWidth',LINE_W,'MarkerSize',MK,...
    'Color',C(3,:),'MarkerFaceColor',C(3,:),'MarkerEdgeColor','w','HandleVisibility','off'); 
set(gca,'XScale','log'); xlim([min(xEI) max(xEI)]); grid on;
xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',10)
ylabel('y','FontName',BASE_FONT,'FontSize',10)
title('Normalization coefficient','FontName',BASE_FONT,'FontSize',11)
legend('Location','best','Box','off')
style_axes(gca,BASE_FONT,GRID_ALPHA)

if export
    % Save
    print(fig1, fullfile(saveDir,'fig1_EI_sweep.pdf'), '-dpdf','-painters','-r300');
    print(fig1, fullfile(saveDir,'fig1_EI_sweep.svg'), '-dsvg','-r300');
end

% ====================== FIGURE 2 ======================
unitmax = @(y) (max(abs(y))>0) .* (y ./ max(abs(y))) + (max(abs(y))==0).*y;
Tn    = unitmax(Tthr);
%Pn    = unitmax(Ppow);
TnPn  = unitmax(Tthr ./ Ppow);
E1n   = unitmax(E1);
Eendn = unitmax(Eend);

fig2 = figure('Units','centimeters','Position',[2 2 FIGSIZE_CM2], 'Color','w');
plot(xEI, Tn,   '-','LineWidth',1.8,'Color',C(1,:),'DisplayName','Thrust (Normalized)'); 
hold on;
plot(xEI, TnPn, '--','LineWidth',1.4,'Color',C(2,:),'DisplayName','Thrust/Power');
plot(xEI, symFactor,  '-','LineWidth',1.8,'Marker','o','MarkerSize',MK,...
    'MarkerFaceColor',C(3,:),'MarkerEdgeColor','w','Color',C(3,:),'DisplayName','Symmetry Factor');
%plot(xEI, Eendn,'-','LineWidth',1.4,'Marker','s','MarkerSize',MK,...
%    'MarkerFaceColor',C(5,:),'MarkerEdgeColor','w','Color',C(5,:),'DisplayName','|eta(end)|');
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
set(gca,'XScale','log'); ylim([-1.05 1.05]); xlim([min(xEI) max(xEI)]); grid on;
xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',10)
ylabel('y (Normalized)','FontName',BASE_FONT,'FontSize',10)
title('Scaled metrics overlay','FontName',BASE_FONT,'FontSize',11)
legend('Location','southeast','Box','off')
style_axes(gca,BASE_FONT,GRID_ALPHA)

if export
    % Save
    print(fig2, fullfile(saveDir,'fig2_overlay.pdf'), '-dpdf','-painters','-r300');
    print(fig2, fullfile(saveDir,'fig2_overlay.svg'), '-dsvg','-r300');
end

% ====================== TABLE ======================
%T = table(EI, S.N_x, S.M_z, S.thrust_N, S.power, S.tail_flat_ratio, S.disp_res, ...
%    'VariableNames', {'EI','N_x','M_z','thrust_N','power','tail_flat_ratio','dispersion_resid'});
%disp('=== EI sweep results ==='); disp(T);

end

% ===== Helper =====
function style_axes(ax,baseFont,gridAlpha)
set(ax,'FontName',baseFont,'FontSize',10,'LineWidth',0.75,...
    'TickDir','out','Box','off');
ax.GridAlpha = gridAlpha;
ax.MinorGridAlpha = gridAlpha;
set(ax,'XMinorTick','on','YMinorTick','on');
end
