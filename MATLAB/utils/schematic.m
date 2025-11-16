%% 2D fluid + flexible raft schematic with A1(x) cos(kx)
clear; close all; clc;

%% -----------------------
%  Parameters
%  -----------------------
L        = 10;      % total domain length in x (from -L/2 to L/2)
H        = 2.0;     % domain depth
L_raft   = 5.0;     % raft length
nWaves   = 4;       % desired # of waves per side
A0_left  = 0.25;    % base amplitude for left side
A0_right = 0.10;    % base amplitude for right side

% Aesthetics
lw_free   = 3.0;           % free-surface line width
lw_raft   = 4.0;           % raft line width
lw_box    = 2.0;           % box/domain line width
col_free  = [0.1 0.3 0.8]; % blue
col_raft  = [0.0 0.0 0.0]; % black
col_box   = [0.3 0.3 0.3]; % darker gray

%% -----------------------
%  Geometry
%  -----------------------
xL_dom  = -L/2;
xR_dom  =  L/2;

x1   = -L_raft/2;     % left end of raft
x2   =  L_raft/2;     % right end of raft


L_side = x1 - xL_dom; % length of each side region

% Wavenumber: nWaves per side
k = 2*pi * nWaves / L_side;

%% -----------------------
%  Amplitude functions A1(x), A2(x)
%  -----------------------
% You said A1(x) cos(kx). Here A1(x) is a *slowly varying* envelope.
% You can change these two handles to whatever envelope you want.

% Left side amplitude A1(x)
% Example: mildly decaying envelope from left wall to raft
A1fun = @(x) A0_left .* (1 + 0.5 * (x - xL_dom) / L_side);  % A1(x)

% Right side amplitude A2(x)
% Example: symmetric envelope
A2fun = @(x) A0_right .* (1 + 0.5 * (xR_dom - x) / L_side); % A2(x)

%% -----------------------
%  Free surface and raft definitions
%  -----------------------
% Left free surface: A1(x) * cos(k * (x - x1))
yLfun = @(x) A1fun(x) .* cos(k * (x - x1));

% Right free surface: A2(x) * cos(k * (x - x2))
yRfun = @(x) A2fun(x) .* cos(k * (x - x2));

% Match raft height at endpoints
y1 = yLfun(x1);
y2 = yRfun(x2);

% Raft center slightly above these values -> small curvature
humpAmp = -0.05;
xmid    = -0.0;          % middle of raft
y_mid   = 0.5*(y1 + y2) + humpAmp;

% Fit quadratic y = a x^2 + b x + c through (x1,y1), (xmid,y_mid), (x2,y2)
Xmat = [x1^2 x1 1;
        xmid^2 xmid 1;
        x2^2 x2 1];
Yvec = [y1; -y_mid; y2];
abc  = Xmat \ Yvec;
a = abc(1); b = abc(2); c = abc(3);
yRaft = @(x) a*x.^2 + b*x + c;

%% -----------------------
%  Discretization
%  -----------------------
Nx_side = 400;
Nx_raft = 400;

xL = linspace(xL_dom, x1, Nx_side);   % left free surface region
xM = linspace(x1,    x2, Nx_raft);    % raft region
xR = linspace(x2,  xR_dom, Nx_side);  % right free surface region

yL = yLfun(xL);
yM = yRaft(xM);
yR = yRfun(xR);

yTopMax = max([yL, yM, yR]);
yTopMin = min([yL, yM, yR]);

%% -----------------------
%  Plot
%  -----------------------
fig = figure('Color','w');
ax  = axes('Parent',fig);
set(ax,'Color','w');
hold(ax,'on');

% Fluid box: bottom + vertical sides only (no straight top line)
plot(ax,[xL_dom xR_dom],[-H -H],           'Color',col_box,'LineWidth',lw_box);
plot(ax,[xL_dom xL_dom],[-H yLfun(xL(1))], '--', 'Color',2*col_box,'LineWidth',lw_box);
plot(ax,[xR_dom xR_dom],[-H yRfun(xR(end))], '--', 'Color',2*col_box,'LineWidth',lw_box);

% Free surface left
plot(ax, xL, yL, 'Color', col_free, 'LineWidth', lw_free);

% Raft
plot(ax, xM, yM, 'Color', col_raft, 'LineWidth', lw_raft);

% Free surface right
plot(ax, xR, yR, 'Color', col_free, 'LineWidth', lw_free);

% Optional: mark raft center
% plot(ax, xmid, yRaft(xmid), 'ko', 'MarkerFaceColor','k','MarkerSize',5);

% Axes formatting
xlim(ax,[xL_dom, xR_dom]);
ylim(ax,[-H-0.1, yTopMax + 0.1]);
set(ax,'XTick',[],'YTick',[],'Box','off', 'XColor', 'none', 'YColor', 'none');
pbaspect(ax,[2 1 1]);  % nice aspect ratio

%title(ax,'A_1(x)\cos(kx) free surface + flexible raft','FontName','Arial','FontSize',12);

print(gcf, fullfile('figures/schematic.pdf'), '-dpdf','-vector','-r300');
