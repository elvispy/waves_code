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
symFactor = -(E1.^2 - Eend.^2) ./ (E1.^2 + Eend.^2);

unitmax = @(y) (max(abs(y))>0) .* (y ./ max(abs(y))) + (max(abs(y))==0).*y;
Tn    = unitmax(Tthr);
%Pn    = unitmax(Ppow);
TnPn  = unitmax(Tthr ./ Ppow);
%E1n   = unitmax(E1);
%Eendn = unitmax(Eend);

% ====================== FIGURE 1 ======================
fig1 = figure('Units','centimeters','Position',[2 2 FIGSIZE_CM1], 'Color','w');

subplot(3,1,1); hold on;
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
yline(0, '-', 'HandleVisibility', 'off', 'LineWidth', 0.5);
plot(xEI, Tthr, '-','LineWidth',LINE_W,'MarkerSize',MK,...
    'Color',C(1,:),'MarkerFaceColor',C(1,:),'MarkerEdgeColor','w', 'HandleVisibility', 'off');
set(gca,'XScale','log'); xlim([min(xEI) max(xEI)]); grid on;

xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',14)
ylabel('Thrust (N/m)','FontName',BASE_FONT,'FontSize',14)
%title('Thrust','FontName',BASE_FONT,'FontSize',11)
legend('Location','best','Box','off')
style_axes(gca,BASE_FONT,GRID_ALPHA)

subplot(3,1,2)
plot(xEI, -Ppow, 'o-','LineWidth',LINE_W,'MarkerSize',MK,...
    'Color',C(2,:),'MarkerFaceColor',C(2,:),'MarkerEdgeColor','w', 'HandleVisibility', 'off');
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
set(gca,'XScale','log','YScale','log'); xlim([min(xEI) max(xEI)]); grid on;
xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',14)
ylabel('Power (W)','FontName',BASE_FONT,'FontSize',14)
%title('Power','FontName',BASE_FONT,'FontSize',14)
legend('Location','best','Box','off')
style_axes(gca,BASE_FONT,GRID_ALPHA)

subplot(3,1,3); hold on;

yline(0, '-', 'HandleVisibility', 'off', 'LineWidth', 0.5);
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
plot(xEI, symFactor, 'o-','LineWidth',LINE_W,'MarkerSize',MK,...
    'Color',C(3,:),'MarkerFaceColor',C(3,:),'MarkerEdgeColor','w',...
    'DisplayName', '$\alpha$', 'HandleVisibility','off'); 
%plot(xEI, TnPn, '--','LineWidth',1.4,'Color',C(2,:),'DisplayName','$\eta$');
set(gca,'XScale','log'); xlim([min(xEI) max(xEI)]); grid on;
xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',14)
ylabel('$\alpha$','FontName',BASE_FONT,'FontSize',18, 'Interpreter', 'latex')
%title('Normalization coefficient','FontName',BASE_FONT,'FontSize',14)
legend('Location','best','Box','off')
style_axes(gca,BASE_FONT,GRID_ALPHA)

if export
    % Save
    print(fig1, fullfile(saveDir,'EI_sweep_fig1.pdf'), '-dpdf','-painters','-r300');
    print(fig1, fullfile(saveDir,'EI_sweep_fig1.svg'), '-dsvg','-r300');
end

% ====================== FIGURE 2 ======================


fig2 = figure('Units','centimeters','Position',[2 2 FIGSIZE_CM2], 'Color','w');
plot(xEI, Tn,   '-','LineWidth',1.8,'Color',C(1,:),'DisplayName','Thrust'); 
hold on;
plot(xEI, TnPn, '--','LineWidth',1.4,'Color',C(2,:),'DisplayName','$\eta$');
plot(xEI, symFactor,  '-','LineWidth',1.8,'Marker','o','MarkerSize',MK,...
    'MarkerFaceColor',C(3,:),'MarkerEdgeColor','w','Color',C(3,:),'DisplayName','$\alpha$');
%plot(xEI, Eendn,'-','LineWidth',1.4,'Marker','s','MarkerSize',MK,...
%    'MarkerFaceColor',C(5,:),'MarkerEdgeColor','w','Color',C(5,:),'DisplayName','|eta(end)|');
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
set(gca,'XScale','log'); ylim([-1.05 1.05]); xlim([min(xEI) max(xEI)]); grid on;
xlabel('EI (N m^4)','FontName',BASE_FONT,'FontSize',14)
ylabel('Normalized metric','FontName',BASE_FONT,'FontSize',14)
%title('Scaled metrics overlay','FontName',BASE_FONT,'FontSize',11)
legend('Location','southeast','Box','off', 'Interpreter', 'latex')
style_axes(gca,BASE_FONT,GRID_ALPHA)

if export
    % Save
    print(fig2, fullfile(saveDir,'EI_sweep_fig2.pdf'), '-dpdf','-painters','-r300');
    print(fig2, fullfile(saveDir,'EI_sweep_fig2.svg'), '-dsvg','-r300');
end

% ====================== TABLE ======================
%T = table(EI, S.N_x, S.M_z, S.thrust_N, S.power, S.tail_flat_ratio, S.disp_res, ...
%    'VariableNames', {'EI','N_x','M_z','thrust_N','power','tail_flat_ratio','dispersion_resid'});
%disp('=== EI sweep results ==='); disp(T);



% ====================== FIGURE 3 ======================
fig3 = figure('Units','centimeters','Position',[4 4 FIGSIZE_CM1], 'Color','w');

% Pre-create axes so we can control positions later
ax1 = subplot(3,1,1); hold(ax1,'on');
ax2 = subplot(3,1,2); hold(ax2,'on');
ax3 = subplot(3,1,3); hold(ax3,'on');

% ---------- AXIS 1: Thrust ----------
axes(ax1); % make sure
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2, ...
    'HandleVisibility', 'off');
yline(0, '-', 'HandleVisibility', 'off', 'LineWidth', 0.5);
plot(xEI, Tthr, '-', ...
    'LineWidth',LINE_W,'MarkerSize',MK, ...
    'Color',C(1,:),'MarkerFaceColor',C(1,:), ...
    'MarkerEdgeColor','w','HandleVisibility','off');
set(ax1,'XScale','log');
xlim(ax1,[min(xEI) max(xEI)]);
grid(ax1,'on');
ylabel(ax1,'Thrust (N/m)','FontName',BASE_FONT,'FontSize',14, 'interpreter', 'latex');
legend(ax1,'Location','best','Box','off');
style_axes(ax1,BASE_FONT,GRID_ALPHA);

% ---------- AXIS 2: Power ----------
axes(ax2);
plot(xEI, -Ppow, 'o-', ...
    'LineWidth',LINE_W,'MarkerSize',MK, ...
    'Color',C(2,:),'MarkerFaceColor',C(2,:), ...
    'MarkerEdgeColor','w','HandleVisibility','off');
xline(EIsurf, '--', 'DisplayName', 'Surferbot', 'LineWidth', 2);
set(ax2,'XScale','log','YScale','log');
xlim(ax2,[min(xEI) max(xEI)]);
grid(ax2,'on');
ylabel(ax2,'Power (W)','FontName',BASE_FONT,'FontSize',14, 'Interpreter', 'latex');
legend(ax2,'Location','northeast','Box','off', 'FontSize', 16);
style_axes(ax2,BASE_FONT,GRID_ALPHA);

% ---------- AXIS 3: alpha ----------
axes(ax3); hold on;
yline(0, '-', 'HandleVisibility','off','LineWidth',0.5);
xline(EIsurf, '--', 'DisplayName','Surferbot','LineWidth',2, ...
    'HandleVisibility', 'off');
plot(xEI, symFactor, 'o-', ...
    'LineWidth',LINE_W,'MarkerSize',MK, ...
    'Color',C(3,:),'MarkerFaceColor',C(3,:), ...
    'MarkerEdgeColor','w',...
    'DisplayName', '$\alpha$', 'HandleVisibility','on');
plot(xEI, TnPn, '--','LineWidth',1.4,'Color',C(2,:),'DisplayName','$\eta$');
set(ax3,'XScale','log');
xlim(ax3,[min(xEI) max(xEI)]);
grid(ax3,'on');
ylabel(ax3,'Normalized Metric','FontName',BASE_FONT,'FontSize',20, 'Interpreter', 'latex');
legend(ax3,'Location','southeast','Box','off', 'Interpreter', 'latex', 'Fontsize', 20);
style_axes(ax3,BASE_FONT,GRID_ALPHA);

% ---------- Make it compact & "fused" ----------
left   = 0.12;
width  = 0.80;
h      = 0.25;   % height of each axis
gap    = 0.00;   % vertical gap between axes (0 = touching)

bottom3 = 0.14;              % enough space for the bottom xlabel
bottom2 = bottom3 + h + gap;
bottom1 = bottom2 + h + gap;

set(ax3,'Position',[left bottom3 width h]);
set(ax2,'Position',[left bottom2 width h]);
set(ax1,'Position',[left bottom1 width h]);

% Remove x tick labels on top two axes (but keep ticks)
set(ax1,'XTickLabel',[]);
set(ax2,'XTickLabel',[]);

% Ticks inside, axes "fused"
set([ax1 ax2 ax3], ...
    'TickDir','in', ...           % ticks inward
    'TickLength',[0.015 0.015], ...
    'Box','on', ...               % frame around each axis
    'Layer','top');               % draw ticks/box on top of plots

setYL(ax1);
%setYL(ax2);
setYL(ax3);


% Shared x-limits (if not already done)
linkaxes([ax1 ax2 ax3],'x');

% Bottom xlabel (will be visible because we left bottom margin)
xlabel(ax3,'EI (N m^4)','FontName',BASE_FONT,'FontSize',14);


% ---------- Export ----------
if export
    print(fig3, fullfile(saveDir,'EI_sweep_fig3.pdf'), '-dpdf','-painters','-r300');
    print(fig3, fullfile(saveDir,'EI_sweep_fig3.svg'), '-dsvg','-r300');
end


end

% ===== Helper =====
function style_axes(ax,baseFont,gridAlpha)
set(ax,'FontName',baseFont,'FontSize',14,'LineWidth',0.75,...
    'TickDir','out','Box','on');
ax.GridAlpha = gridAlpha;
ax.MinorGridAlpha = gridAlpha;
set(ax,'XMinorTick','on','YMinorTick','on');
end


function setYL(ax)
    % get current limits
    yl = ylim(ax);
    ymid = mean(yl);
    yrng = (yl(2)-yl(1))/2;

    % shrink to 80% of height around center
    new_half = 1.2 * yrng;
    new_limits = [ymid-new_half, ymid+new_half];
    ylim(ax, new_limits);

    % create exactly 3 ticks: low, mid, high
    %set(ax, 'YTick', linspace(new_limits(1), new_limits(2), 3));
end

