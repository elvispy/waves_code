function test_quick_checks
% CLI smoke tests and readable output

addpath('../src');

base = struct( ...
    'sigma',0, 'rho',1000, 'nu',0, 'g',10*9.81, ...
    'L_raft',0.1, 'motor_position',0.5*0.1/2, 'd',0.1/2, ...
    'EI',100*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*3.0, ...
    'domainDepth',0.2, 'n',201, 'M',100, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*10);

% Labeled cases
cases = struct([]);
cases(1).label = "baseline";          cases(1).p = base;
tmp = base; tmp.domainDepth = 2*base.domainDepth;
cases(2).label = "depthx2";          cases(2).p = tmp;
tmp = base; tmp.n = 2*ceil(3/4*base.n)+1;          % ~1.5x, force odd
cases(3).label = "biggern";       cases(3).p = tmp;
tmp = base; tmp.M = 2*ceil(3/4*base.M)+1;          % ~1.5x, force odd
cases(4).label = "biggerM";       cases(4).p = tmp;

% Run
for i = 1:numel(cases)
    cases(i).S = run_case(cases(i).p);
end
A = cases(1).S;

% Pack results
labels          = string({cases.label}).';
domainDepth_m   = arrayfun(@(c) c.S.args.domainDepth, cases).';
N_x             = arrayfun(@(c) c.S.args.N,           cases).';
M_z             = arrayfun(@(c) c.S.args.M,           cases).';
thrust_N        = arrayfun(@(c) c.S.args.thrust,      cases).';
k_minv          = arrayfun(@(c) real(c.S.args.k),     cases).';
dispersion_resid= arrayfun(@(c) c.S.disp_res,         cases).';
LH_flux         = arrayfun(@(c) c.S.LH,               cases).';
momentum_flux   = arrayfun(@(c) c.S.mom,              cases).';
flux_diff       = LH_flux - momentum_flux;
tail_flat_ratio = arrayfun(@(c) c.S.tail_ratio,       cases).';

T = table(labels, domainDepth_m, N_x, M_z, thrust_N, k_minv, ...
          dispersion_resid, LH_flux, momentum_flux, flux_diff, tail_flat_ratio, ...
          'VariableNames', {'caseLabel','domainDepth_m','N_x','M_z','thrust_N', ...
                            'k_minv','dispersion_resid','LH_flux','momentum_flux', ...
                            'flux_diff','tail_flat_ratio'});

disp('=== Flexible Surferbot quick checks ===');
disp(T);

% Relative thrust changes vs baseline
for i = 2:numel(cases)
    rel = abs(cases(i).S.args.thrust - A.args.thrust) / max(1e-12, abs(A.args.thrust));
    fprintf('%s vs baseline thrust change: %.3g\n', cases(i).label, rel);
end

fprintf('Max |dispersion residual|: %.3e\n', max(T.dispersion_resid));
fprintf('Tail flatness (std/mean) per case: %s\n', strjoin( ...
    arrayfun(@(c) sprintf('%s %.3g', c.label, c.S.tail_ratio), cases, 'UniformOutput', false), '  '));
fprintf('(target tail_flat_ratio < 0.1)\n');

% Plot
figure(1); clf; hold on;
for i = 1:numel(cases)
    semilogy(cases(i).S.x, abs(cases(i).S.eta), 'DisplayName', cases(i).label);
end
xlabel('x (m)'); ylabel('|η|'); legend('show','Location','best');
set(gca,'FontSize',14); pbaspect([3 1 1]);
end

function S = run_case(p)
% Execute core solver and compute diagnostics
[~, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'n',p.n,'M',p.M,'motor_inertia',p.motor_inertia,'BC',p.BC);

% Dispersion residual (gravity-only form)
k    = real(args.k);
res  = abs(args.omega^2 - args.g*k);

% Flux diagnostics
dx = abs(x(2)-x(1)); dz = abs(z(2)-z(1));
[Dx, ~] = getNonCompactFDmatrix2D(args.M,args.N,dx,dz,1,args.ooa);
u  = reshape(Dx * reshape(phi, args.M*args.N,1), args.M, args.N);

rho = args.rho; omega = args.omega;
LH  = 0.25 * rho * omega^2 / k * (abs(eta(2))^2 - abs(eta(end-1))^2);
mom = rho * trapz(z, abs(u(:,2)).^2 - abs(u(:,end-1)).^2);

% Tail flatness over last quarter of domain
idx0 = max(1, ceil(0.75*numel(eta)));
tail = abs(eta(idx0:end));
tail_ratio = std(tail) / max(eps, mean(tail));

S = struct('x',x,'z',z,'phi',phi,'eta',eta,'args',args, ...
           'disp_res',res,'LH',LH,'mom',mom,'tail_ratio',tail_ratio);
end
