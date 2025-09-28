function S = thrust_vs_Ldomain_test
% L-domain study: vary L_domain, plot |eta(x)| per case and thrust vs L_domain

addpath('../src');

L_raft = 0.05;
base = struct( ...
    'sigma',0, 'rho',1000, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'L_domain', 0.50, ...          % <-- base horizontal domain length [m]
    'motor_position',0.3*L_raft/2, 'd',L_raft/2, ...
    'EI',100*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*3.0, ...
    'domainDepth',0.2, 'n',301, 'M',300, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*10, 'ooa', 2);

% L_domain values to test
L_list = base.L_domain * [0.5, 1, 2, 4];
L_list = unique(L_list,'stable');

% Preallocate
proto = struct('x',[],'z',[],'phi',[],'eta',[], ...
               'N_x',0,'M_z',0,'L_domain',0, ...
               'thrust_N',NaN, 'tail_flat_ratio',NaN, 'disp_res',NaN, ...
               'args',[]);
S = repmat(proto, 1, numel(L_list));

% Run sweep
for i = 1:numel(L_list)
    p = base; p.L_domain = L_list(i);
    S(i) = run_case(p);
end

% Plot |eta(x)| per case
figure(1); clf; hold on;
for i = 1:numel(S)
    semilogy(S(i).x, abs(S(i).eta), 'DisplayName', sprintf('L=%.3g m', S(i).L_domain));
end
xlabel('x (m)'); ylabel('|{\eta}|'); legend('show','Location','best');
set(gca,'FontSize',14); set(gcf, 'Position',[100 100 900 420]);
title(sprintf('Free-surface amplitude vs x for varying L\_domain at dx/k = %.2g', S(1).args.dx/S(1).args.k));

% Plot thrust vs L_domain
figure(2); clf;
plot([S.L_domain], [S.thrust_N], 'o-','MarkerSize',6,'LineWidth',1.2);
xlabel('L\_domain (m)'); ylabel('Thrust (N)');
set(gca,'FontSize',14);
title('Thrust vs L\_domain');

% Console table
T = table([S.L_domain].', [S.N_x].', [S.M_z].', ...
          [S.thrust_N].', [S.tail_flat_ratio].', [S.disp_res].', ...
    'VariableNames', {'L_domain_m','N_x','M_z','thrust_N','tail_flat_ratio','dispersion_resid'});
disp('=== L_domain sweep results ==='); disp(T);

end

% ---- helpers ----
function S = run_case(p)
% Pass L_domain through to the solver (solver must honor it)
[~, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'L_domain',p.L_domain, ...
    'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'n',p.n,'M',p.M,'motor_inertia',p.motor_inertia,'BC',p.BC);

% Diagnostics
k   = real(args.k);
H   = p.domainDepth;
res = abs(args.omega^2 - args.g*k*tanh(k*H));   % finite-depth dispersion residual

% Optional interior-tail flatness metric
idx0 = max(1, ceil(0.75*numel(eta)));
tail = abs(eta(idx0:end));
tail_ratio = std(tail) / max(eps, mean(tail));

S = struct('x',x,'z',z,'phi',phi,'eta',eta, ...
           'N_x',args.N,'M_z',args.M, ...
           'L_domain',p.L_domain, ...
           'thrust_N',args.thrust, ...
           'tail_flat_ratio',tail_ratio, ...
           'args', args, ...
           'disp_res',res);
end
