function S = thrust_vs_EI_test
% THRUST_VS_EI_TEST
% Purpose:
%   Sweep flexural rigidity (EI) for a “surferbot” model, solve each case,
%   and compare free-surface amplitude profiles and thrust vs EI.
%
% How to run:
%   - Ensure ../src contains FLEXIBLE_SURFERBOT_V2 and dependencies.
%   - Call the function: S = thrust_vs_EI_test;  % returns results table
%
% What it does:
%   - Defines a base parameter set (fluid, geometry, drive, BCs).
%   - Builds EI_list over 10^(-3)–10^(+1) × EI0.
%   - For each EI: solves the model, stores fields, thrust, and diagnostics.
%
% Outputs:
%   - Figures: (1) |η(x)| for all EI; (2) Thrust/d vs EI/EI0 with two estimates.
%   - Console: run constants and a summary table per EI.
%   - Return value S: table with x, z, φ, η, N_x, M_z, thrust_N, Sxx, ratios, residuals, EI.
%
% Key diagnostics:
%   - thrust_N = args.thrust/args.d (pressure integral per depth).
%   - Sxx: alternative thrust proxy from surface energy flux.
%   - tail_flat_ratio: variability near the interior “tail.”
%   - disp_res: |ω^2 − gk| dispersion residual.


addpath('../src');

L_raft = 0.05;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0*1e-6, 'g',9.81, ...
        'L_raft',L_raft, 'motor_position',0.24*L_raft/2, 'd',0.03, ...
        'EI',3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.052, ...
        'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
        'omega',2*pi*80, 'ooa', 4);

% EI values to test (multiplicative sweep)
EI_list = base.EI * (10.^linspace(-3, 1, 100));
EI_list = unique(EI_list, 'stable');

% Preallocate
proto = struct('x',[],'z',[],'phi',[],'eta',[], ...
               'N_x',0,'M_z',0, ...
               'thrust_N',NaN,'tail_flat_ratio',NaN,'disp_res',NaN, ...
               'args', struct(), 'EI', NaN, 'Sxx', NaN);
S = repmat(proto, 1, numel(EI_list));

% Run sweep
for i = 1:numel(EI_list)
    p = base; p.EI = EI_list(i);
    S(i) = run_case(p);
end

% --- constants snapshot (first run)
A = S(1).args;

% |eta(x)| per EI
figure(1); clf;
set(gcf,'Units','pixels','Position',[80 80 1200 500]);  % wide
ax = axes('Position',[0.07 0.12 0.68 0.80]); hold(ax,'on');
for i = 1:numel(S)
    semilogy(ax, S(i).x, abs(S(i).eta), 'DisplayName', sprintf('EI=%.3g', S(i).EI));
end
xlabel(ax,'x (m)'); ylabel(ax,'|{\eta}(x)|');
legend(ax,'show','Location','best'); grid(ax,'on'); set(ax,'FontSize',14);
title(ax,'Free-surface amplitude vs x for varying EI');

% constants textbox (from first run)
info = sprintf([ ...
    '(k*dx)^-1 = %.3g\n' ...
    'H/dz      = %.3g\n' ...
    'omega     = %.3g 1/s\n' ...
    'L         = %.3g m\n' ...
    'L_{domain}= %.3g m\n' ...
    'd         = %.3g m\n' ...
    'EI        = %.3g (varied)\n' ...
    'Gamma     = %.3g\n' ...
    'Fr        = %.3g\n' ...
    'kappa     = %.3g\n' ...
    'Lambda    = %.3g\n' ...
    'Weber     = %.3g \n' ...
    'Reynolds  = %.3g'], ...
    2*pi/(A.dx*A.k), A.domainDepth/A.dz, ...
    A.omega, A.L_raft, A.L_domain, A.d, A.EI, ...
    A.nd_groups.Gamma, A.nd_groups.Fr, A.nd_groups.kappa, ...
    A.nd_groups.Lambda, A.nd_groups.We, A.nd_groups.Re);

annotation('textbox',[0.78 0.12 0.20 0.80], ...
    'String', sprintf('Run constants\n\n%s', info), ...
    'Interpreter','none','EdgeColor','none', ...
    'VerticalAlignment','top','FontSize',16,'FontName','FixedWidth');

% Thrust vs EI
figure(2); clf;
plot([S.EI]./base.EI, [S.thrust_N], 'o-','MarkerSize',6,'LineWidth',1.2, 'DisplayName', 'Pressure integral'); hold on;
plot([S.EI]./base.EI, [S.Sxx],      '--','MarkerSize',6,'LineWidth',2.0, 'DisplayName', 'LH');
xlabel('EI/EI0'); ylabel('Thrust/d (N/m)');
set(gca,'FontSize',14); grid on; legend; set(gca, 'YScale', 'linear', 'XScale', 'log');
title('Thrust vs EI');

% Console table
T = table([S.EI].', [S.N_x].', [S.M_z].', ...
          [S.thrust_N].', [S.Sxx].', [S.tail_flat_ratio].', [S.disp_res].', ...
    'VariableNames', {'EI','N_x','M_z','thrust_N', 'LH', 'tail_flat_ratio','dispersion_resid'});
disp('=== EI sweep results ==='); disp(T);

S = struct2table(S);

end

% ---- helper ----
function S = run_case(p)
[~, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft, ...
    'motor_inertia',p.motor_inertia,'BC',p.BC);

k    = real(args.k);
res  = abs(args.omega^2 - args.g*k);

% Optional interior-tail flatness metric
idx0 = max(1, ceil(0.05*numel(eta)));
tail = abs(eta(1:idx0));
tail_ratio = std(tail) / max(eps, mean(tail));

Sxx = (args.rho*args.g/4 + 3/4*args.sigma*args.k^2) ...
    * (abs(eta(1))^2 - abs(eta(end))^2);

S = struct('x',x,'z',z,'phi',phi,'eta',eta, ...
           'N_x',args.N,'M_z',args.M, ...
           'thrust_N',args.thrust/args.d, ...
           'tail_flat_ratio',tail_ratio, ...
           'args', args, ...
           'disp_res',res, ...
           'EI', p.EI, 'Sxx', Sxx);
end
