function S = sweep_EI
% EI sweep: vary flexural rigidity EI and analyze response.
% Plots:
%   Fig 1: 1�3 tiled panels vs EI/EI0 ? (a) Thrust, (b) Power, (c) |?(1)| and |?(end)|
%   Fig 2: Overlay all series after unit-max normalization for shape comparison.
% Console:
%   Table with EI, N, M, thrust, power, tail ratio, dispersion residual.

addpath('../src');

L_raft = 0.05;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0*1e-6, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.24*L_raft/2, 'd',0.03, ...
    'EI',3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.052, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*80, 'ooa', 4);

% EI values to test (multiplicative sweep)
EI_list = 10.^linspace(-5, -2, 300); %base.EI * (10.^linspace(-3, 1, 100));
EI_list = unique(EI_list, 'stable');

% Preallocate
proto = struct('x',[],'z',[],'eta',[], ...
               'N_x',0,'M_z',0, ...
               'thrust_N',NaN,'tail_flat_ratio',NaN,'disp_res',NaN, ...
               'args', struct(), 'EI', NaN, 'power', NaN, ...
               'eta_1', NaN, 'eta_end', NaN);
S = repmat(proto, 1, numel(EI_list));

% Run sweep
for i = 1:numel(EI_list)
    p = base; p.EI = EI_list(i);
    S(i) = run_case(p);
end

% --- constants snapshot (first run)
A = S(1).args;

% ---------- FIGURE 1: three panels ----------
xEI = [S.EI]./base.EI;
Tthr = [S.thrust_N];
Ppow = [S.power];
E1   = [S.eta_1];
Eend = [S.eta_end];

figure; clf;
set(gcf,'Units','pixels','Position',[80 80 1300 1000]);  % wide window

% (a) Thrust vs EI/EI0
subplot(3,1,1);
plot(xEI, Tthr, 'o-','LineWidth',1.2,'MarkerSize',5);
set(gca,'XScale','log'); grid on; set(gca,'FontSize',13);
xlabel('EI/EI_0'); ylabel('Thrust/d (N/m)');
title('Thrust');

% (b) Power vs EI/EI0
subplot(3,1,2);
plot(xEI, -Ppow, 'o-','LineWidth',1.2,'MarkerSize',5);
set(gca,'XScale','log', 'Yscale', 'log'); grid on; set(gca,'FontSize',13);
xlabel('EI/EI_0'); ylabel('Power (W)');
title('Power');

% (c) Endpoint amplitudes vs EI/EI0
subplot(3,1,3);
plot(xEI, E1,   'o-','LineWidth',1.2,'MarkerSize',5,'DisplayName','|eta(1)|'); hold on;
plot(xEI, Eend, 's-','LineWidth',1.2,'MarkerSize',5,'DisplayName','|eta(end)|');
set(gca,'XScale','log'); grid on; set(gca,'FontSize',13);
xlabel('EI/EI_0'); ylabel('Amplitude (m)');
title('|eta(1)| and |eta(end)|'); legend('Location','best');


% ---------- FIGURE 2: overlay with unit-max normalization ----------
% unit-max normalization (safe if all nonnegative; guards against all-zero)
unitmax = @(y) (max(abs(y))>0) .* (y ./ max(abs(y))) + (max(abs(y))==0).*y;

Tn = unitmax(Tthr);
Pn = unitmax(Ppow);
TnPn = unitmax(Tthr./Ppow);
E1n = unitmax(E1);
Eendn = unitmax(Eend);

figure; clf;
set(gcf,'Units','pixels','Position',[120 120 1000 420]);
plot(xEI, Tn,     '--','LineWidth',3,'MarkerSize',5,'DisplayName','Thrust/d'); hold on;
plot(xEI, TnPn,   '--','LineWidth',2,'MarkerSize',5,'DisplayName','Thrust/Power');
plot(xEI, E1n,     '-','LineWidth',3,'MarkerSize',10,'DisplayName','|eta(1)|');
plot(xEI, Eendn,   '-','LineWidth',1.5,'MarkerSize',5,'DisplayName','|eta(end)|');
set(gca,'XScale','log'); ylim([-1.05 1.05]);
grid on; set(gca,'FontSize',13);
xlabel('EI/EI_0'); ylabel('Normalized');
title('Overlay: shapes vs EI (scaled)');
legend('Location','best');

% ---------- Console table ----------
T = table([S.EI].', [S.N_x].', [S.M_z].', ...
          [S.thrust_N].', [S.power].', [S.tail_flat_ratio].', [S.disp_res].', ...
    'VariableNames', {'EI','N_x','M_z','thrust_N', 'power', 'tail_flat_ratio','dispersion_resid'});
disp('=== EI sweep results ==='); disp(T);

S = struct2table(S);
save('data/EI_sweep.mat', 'S');
end

% ---- helper ----
function S = run_case(p)
[~, x, z, ~, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft, ...
    'motor_inertia',p.motor_inertia,'BC',p.BC);

k    = real(args.k);
res  = abs(args.omega^2 - args.g*k);

% interior "head" window for your tail/flat metric as you had it
idx0 = max(1, ceil(0.05*numel(eta)));
tail = abs(eta(1:idx0));
tail_ratio = std(tail) / max(eps, mean(tail));

args = rmfield(args, 'phi_z');
S = struct('x',x,'z',z,'eta',eta, ...
           'N_x',args.N,'M_z',args.M, ...
           'thrust_N',args.thrust/args.d, ...
           'tail_flat_ratio',tail_ratio, ...
           'eta_1', abs(eta(1)), 'eta_end', abs(eta(end)), ...
           'args', args, ...
           'disp_res',res, ...
           'EI', p.EI, 'power', args.power);
end
