function thrust_vs_d_test
% Goal: characterize sensitivity to raft half-length d.
% 1) Shape similarity: does |eta(x)| collapse as |eta|(xi) with xi=x/d?
% 2) Power law: thrust_N ~ d^p  (report fitted p)
%
% Only 'd' changes. Everything else is fixed.

addpath('../src');

% --- base setup ---
L_raft = 0.05;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.4*L_raft/2, 'd',1, ...
    'EI',100*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*10.0, ...
    'domainDepth',0.5, 'L_domain', 3*L_raft, 'n',101, 'M',200, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*5, 'ooa', 4);

% --- sweep in d ---
d_list = base.d * [1/4 1/2 1 2 4];  % broad span
proto = struct('x',[],'eta',[],'d',NaN,'thrust_N',NaN,'args',struct(),'LH',nan);
S = repmat(proto,1,numel(d_list));
for i = 1:numel(d_list)
    p = base; p.d = d_list(i);
    S(i) = run_case_eta_only(p);
end

% --- convenience ---
A = S(1).args;  % take grids/const from first run
k = A.k;

% --- plot |eta(x)| for each d ---
figure(1); clf;
set(gcf,'Units','pixels','Position',[80 80 1200 500]);

ax = axes('Position',[0.07 0.12 0.68 0.80]); hold(ax,'on');
for i = 1:numel(S)
    semilogy(ax, S(i).x, abs(S(i).eta), 'DisplayName', sprintf('d=%.3g m', S(i).d));
end
xlabel(ax,'x (m)'); ylabel(ax,'|{\eta}(x)|');
legend(ax,'show','Location','best'); grid(ax,'on'); set(ax,'FontSize',13);
title(ax,'Free-surface amplitude vs x for varying d');

% right info panel
info = sprintf([ ...
    '(k*dx)^-1 = %.3g\n' ...
    'H/dz      = %.3g\n' ...
    'omega     = %.3g 1/s\n' ...
    'L         = %.3g m\n' ...
    'L_{domain}= %.3g m\n' ...
    'EI        = %.3g\n' ...
    'Gamma     = %.3g\n' ...
    'Fr        = %.3g\n' ...
    'kappa     = %.3g\n' ...
    'Lambda    = %.3g\n' ...
    'Weber     = %.3g\n' ...
    'Reynolds  = %.3g'], ...
    2*pi/(A.dx*A.k), A.domainDepth/A.dz, ...
    A.omega, A.L_raft, A.L_domain, A.EI, ...
    A.nd_groups.Gamma, A.nd_groups.Fr, A.nd_groups.kappa, ...
    A.nd_groups.Lambda, A.nd_groups.We, A.nd_groups.Re);
annotation('textbox',[0.78 0.12 0.20 0.80], ...
    'String', sprintf('Run constants\n\n%s', info), ...
    'Interpreter','none','EdgeColor','none', ...
    'VerticalAlignment','top','FontSize',16,'FontName','FixedWidth');

% --- similarity collapse: normalize and replot vs xi = x/d ---
% normalize by local amplitude near x=0 for each case
xi_max = inf;
for i = 1:numel(S)
    xi_i = S(i).x / S(i).d;
    xi_max = min(xi_max, max(abs(xi_i)));
end
xi = linspace(-xi_max, xi_max, 1000);   % common xi grid

eta_norm = zeros(numel(S), numel(xi));
for i = 1:numel(S)
    x  = S(i).x;
    xi_i = x / S(i).d;
    % reference scale: amplitude at x ~ 0
    [~,i0] = min(abs(x));
    s_i = max(abs(S(i).eta(i0)), eps);
    % interpolate magnitude on common xi grid
    mag_i = abs(S(i).eta) / s_i;
    eta_norm(i,:) = interp1(xi_i, mag_i, xi, 'linear', 'extrap');
end

figure(2); clf; hold on;
for i = 1:numel(S)
    plot(xi, eta_norm(i,:), 'DisplayName', sprintf('d=%.3g', S(i).d));
end
xlabel('\xi = x/d'); ylabel('|{\eta}| / |{\eta}(0)|');
title('Similarity collapse test: normalized amplitude vs \xi'); grid on; legend('show');

% quantify collapse error vs reference d_ref (closest to base.d)
[~,iref] = min(abs(d_list - base.d));
ref = eta_norm(iref,:);
collErr = zeros(1,numel(S));
for i = 1:numel(S)
    num = norm(eta_norm(i,:) - ref);
    den = max(norm(eta_norm(i,:)), eps);
    collErr(i) = num/den;
end

% --- thrust scaling: fit power law F ~ d^p ---
dcol = [S.d].'; FT = [S.thrust_N].';
valid = FT>0;
p_fit = polyfit(log(dcol(valid)), log(FT(valid)), 1); % slope = p
p = p_fit(1);

figure(3); clf; 
loglog(dcol, FT, 'o-','LineWidth',1.5,'DisplayName','FT'); hold on;
loglog(dcol, [S.LH].', 's--','LineWidth',1.2,'DisplayName','LH');
% overlay fitted ~ d^p line through reference point
dref = dcol(iref); FTref = FT(iref);
dline = [min(dcol) max(dcol)];
FTline = FTref * (dline/dref).^p;
loglog(dline, FTline, 'k:','LineWidth',1.2, 'DisplayName', sprintf('fit ~ d^{%.2f}', p));
xlabel('d (m)'); ylabel('Thrust (N)'); grid on; legend('show','Location','best');
title('Thrust vs d (log-log) with power-law fit');

% --- console summary ---
T = table(dcol, k*dcol, FT, [S.LH].', collErr.', ...
    'VariableNames', {'d','k_d','thrust_N','LH','shapeCollapseErr'});
disp('=== d-sweep summary ==='); disp(T);
fprintf('Power-law fit FT ~ d^p: p = %.3f\n', p);
fprintf('Suggested tolerances: shapeCollapseErr < 5e-2 on shared xi-range; |p-2| < 0.3 if small-kd theory expects FT~d^2.\n');

end

% ---- helpers ----
function S = run_case_eta_only(p)
[~, x, ~, ~, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'n',p.n,'M',p.M,'motor_inertia',p.motor_inertia,'BC',p.BC);

S = struct('x',x,'eta',eta,'d',p.d, ...
           'thrust_N',args.thrust,'args',args, ...
           'LH', 1/4 * p.rho * p.omega^2 / args.k * (abs(eta(1))^2 - abs(eta(end))^2));
end
