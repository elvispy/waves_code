function S = sweep_EI_experiment
% Sweep over driving frequency omega and motor_position.
% Plots:
%   2D color map: Thrust/d (N/m) as a function of (omega, motor_position).
%
% Console:
%   Displays size and ranges of the sweep; no file export.

addpath('../src');

% --- base parameters (same as before where possible) ---
L_raft = 0.02;
E      = 330e3;     % Pa
h      = 0.0015;     % m, thickness
d      = 0.003;     % m, depth
EI0    = E * d * h^3/12;
omega0 = 2*pi*10;   % rad/s (reference)
rho_oomoo = 1.34e3;   % kg/m3

% Base parameter struct (omega and motor_position will be overwritten in sweep)
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',1e-6, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position', 0.25*L_raft/2, ... % placeholder
    'd',d, 'EI',EI0, ...
    'rho_raft',rho_oomoo * d * h, ...
    'motor_force',50e-6, 'BC','radiative', ...
    'omega',omega0, 'ooa',4);

% ---------- SWEEP DEFINITION ----------
% Edit these two lines to change the sweep ranges/resolution.
omega_list      = omega0 * (1:10);   % rad/s
motor_frac_list = linspace(0, 0.5, 10);            % fraction of L_raft

n_omega = numel(omega_list);
n_mp    = numel(motor_frac_list);

% Preallocate results
proto = struct('x',[],'z',[],'eta',[], ...
               'N_x',0,'M_z',0, ...
               'thrust_N',NaN,'tail_flat_ratio',NaN,'disp_res',NaN, ...
               'args', struct(), ...
               'EI', EI0, 'power', NaN, ...
               'eta_1', NaN, 'eta_end', NaN, ...
               'omega', NaN, 'motor_position', NaN);

S = repmat(proto, n_mp, n_omega);      % 2D array indexed by (motor_pos, omega)
Thrust = NaN(n_mp, n_omega);           % Thrust/d (N/m)
Asymmetry = NaN(n_mp, n_omega);           % Thrust/d (N/m)

% ---------- RUN SWEEP ----------
parfor i_omega = 1:n_omega
    for j_mp = 1:n_mp
        p = base;
        p.omega          = omega_list(i_omega);
        p.motor_position = motor_frac_list(j_mp) * L_raft;

        S(j_mp, i_omega) = run_case(p);
        Thrust(j_mp, i_omega) = S(j_mp, i_omega).thrust_N;
        Asymmetry(j_mp, i_omega) = ...
            (S(j_mp, i_omega).eta_1.^2 - S(j_mp, i_omega).eta_end.^2) / ...
            (S(j_mp, i_omega).eta_1.^2 + S(j_mp, i_omega).eta_end.^2);
        S(j_mp, i_omega).omega = p.omega;
        S(j_mp, i_omega).motor_position = p.motor_position;
    end
end

% ---------- PLOT: THRUST AS FUNCTION OF (omega, motor_position) ----------
%[OmegaGrid, MPfracGrid] = meshgrid(omega_list, motor_frac_list);

figure; clf;
set(gcf,'Units','pixels','Position',[80 80 1000 700]);

% 2D color map (view from above)
imagesc(omega_list/(2*pi), motor_frac_list, Asymmetry);
set(gca, 'YDir', 'normal', 'FontSize', 13);
xlabel('Frequency f (Hz)');
ylabel('Motor position / L_{raft}');
title('Thrust/d (N/m) as a function of frequency and motor position');
cb = colorbar;
ylabel(cb, 'Thrust/d (N/m)');

grid on;

% ---------- Console summary ----------
fprintf('=== omega-motor sweep ===\n');
fprintf('omega: [%g, %g] rad/s, %d points\n', ...
    min(omega_list), max(omega_list), n_omega);
fprintf('freq:  [%g, %g] Hz\n', min(omega_list/(2*pi)), max(omega_list/(2*pi)));
fprintf('motor position: [%g, %g] * L_raft, %d points\n', ...
    min(motor_frac_list), max(motor_frac_list), n_mp);

end

% ---- helper (unchanged except for adding omega/motor_position to S above) ----
function S = run_case(p)
[~, x, z, ~, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft, ...
    'motor_force',p.motor_force,'BC',p.BC);

k    = real(args.k);
res  = abs(args.omega^2 - args.g*k);

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
           'EI', p.EI, ...
           'power', args.power, ...
           'omega', p.omega, ...
           'motor_position', p.motor_position);
end
