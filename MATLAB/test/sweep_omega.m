function S = sweep_omega
% OMEGA sweep: vary angular frequency (omega) and analyze response.
% Plots:
%   Fig 1: 3 tiled panels vs Frequency (Hz) -> (a) Thrust, (b) Power, (c) Amplitudes
%   Fig 2: Overlay all series after unit-max normalization.
% Console:
%   Table with Freq, Omega, Thrust, Power, etc.

addpath('../src');

L_raft = 0.05;
d = 0.003;  % m
h = 1e-3;   % m
E = 330e3;  % Pa
rho = 1340; % kg/m3
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0*1e-6, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0*L_raft/2, 'd',0.03, ...
    'EI',E*d*h^3/12, 'rho_raft',rho * d * h, ...
    'motor_force',1e-6, 'BC','radiative', ...
    'omega', NaN, 'ooa', 4); % Omega is now NaN in base, set in loop

% Frequency sweep setup
% Sweeping from 10 Hz to 120 Hz
freq_Hz_list = linspace(10, 90, 50); 
omega_list   = 2 * pi * freq_Hz_list;

% Preallocate
proto = struct('x',[],'z',[],'eta',[], ...
               'N_x',0,'M_z',0, ...
               'thrust_N',NaN,'tail_flat_ratio',NaN,'disp_res',NaN, ...
               'args', struct(), 'omega', NaN, 'freq_Hz', NaN, ...
               'power', NaN, 'eta_1', NaN, 'eta_end', NaN);
S = repmat(proto, 1, numel(omega_list));

% Run sweep
fprintf('Running Omega sweep over %d points...\n', numel(omega_list));
for i = 1:numel(omega_list)
    p = base; 
    p.omega = omega_list(i);
    S(i) = run_case(p);
end

% ---------- FIGURE 1: three panels ----------
xFreq = [S.freq_Hz]; % X-axis is now Frequency in Hz
Tthr  = [S.thrust_N];
Ppow  = [S.power];
E1    = [S.eta_1];
Eend  = [S.eta_end];

figure; clf;
set(gcf,'Units','pixels','Position',[80 80 1300 1000]);  % wide window

% (a) Thrust vs Frequency
subplot(3,1,1);
plot(xFreq, Tthr, 'o-','LineWidth',1.2,'MarkerSize',3);
set(gca,'FontSize',13); grid on;
% Note: Frequency is usually linear, but response (Y) is often Log
set(gca, 'YScale', 'linear'); 
xlabel('Frequency (Hz)'); ylabel('Thrust/d (N/m)');
title('Thrust vs Frequency');

% (b) Power vs Frequency
subplot(3,1,2);
plot(xFreq, -Ppow, 'o-','LineWidth',1.2,'MarkerSize',3);
set(gca,'Yscale', 'log'); grid on; set(gca,'FontSize',13);
xlabel('Frequency (Hz)'); ylabel('Power (W)');
title('Power input (Log Scale)');

% (c) Endpoint amplitudes vs Frequency
subplot(3,1,3);
plot(xFreq, E1,   'o-','LineWidth',1.2,'MarkerSize',3,'DisplayName','|eta(1)|'); hold on;
plot(xFreq, Eend, 's-','LineWidth',1.2,'MarkerSize',3,'DisplayName','|eta(end)|');
grid on; set(gca,'FontSize',13);
set(gca, 'YScale', 'log'); % Log scale often better for resonance peaks
xlabel('Frequency (Hz)'); ylabel('Amplitude (m)');
title('|eta(1)| and |eta(end)|'); legend('Location','best');


% ---------- FIGURE 2: overlay with unit-max normalization ----------
unitmax = @(y) (max(abs(y))>0) .* (y ./ max(abs(y))) + (max(abs(y))==0).*y;

Tn    = unitmax(Tthr);
% Pn  = unitmax(Ppow); % (Unused in plot, but good to have)
TnPn  = unitmax(Tthr./Ppow);
E1n   = unitmax(E1);
Eendn = unitmax(Eend);

figure; clf;
set(gcf,'Units','pixels','Position',[120 120 1000 420]);
plot(xFreq, Tn,      '--','LineWidth',3,'MarkerSize',5,'DisplayName','Thrust/d'); hold on;
plot(xFreq, TnPn,    '--','LineWidth',2,'MarkerSize',5,'DisplayName','Thrust/Power');
plot(xFreq, E1n,     '-','LineWidth',3,'MarkerSize',10,'DisplayName','|eta(1)|');
plot(xFreq, Eendn,   '-','LineWidth',1.5,'MarkerSize',5,'DisplayName','|eta(end)|');
ylim([-1.05 1.05]);
grid on; set(gca,'FontSize',13);
xlabel('Frequency (Hz)'); ylabel('Normalized Unit-Max');
title('Overlay: Shapes vs Frequency (Normalized)');
legend('Location','best');

% ---------- Console table ----------
T = table([S.freq_Hz].', [S.omega].', ...
          [S.thrust_N].', [S.power].', [S.eta_end].', ...
    'VariableNames', {'Freq_Hz','Omega','thrust_N', 'power', 'eta_end'});
disp('=== Omega sweep results ==='); 
disp(T(1:10:end, :)); % Display every 10th row to save space

S = struct2table(S);
save('data/Omega_sweep.mat', 'S');
end

% ---- helper ----
function S = run_case(p)
    [~, x, z, ~, eta, args] = flexible_surferbot_v2( ...
        'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
        'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
        'EI',p.EI,'rho_raft',p.rho_raft, ...
        'motor_force',p.motor_force,'BC',p.BC);

    k    = real(args.k);
    res  = abs(args.omega^2 - args.g*k);

    % interior "head" window for your tail/flat metric as you had it
    idx0 = max(1, ceil(0.05*numel(eta)));
    tail = abs(eta(1:idx0));
    tail_ratio = std(tail) / max(eps, mean(tail));

    args = rmfield(args, 'phi_z');
    
    % Store Omega and Freq_Hz instead of EI in the top level struct
    S = struct('x',x,'z',z,'eta',eta, ...
               'N_x',args.N,'M_z',args.M, ...
               'thrust_N',args.thrust/args.d, ...
               'tail_flat_ratio',tail_ratio, ...
               'eta_1', abs(eta(1)), 'eta_end', abs(eta(end)), ...
               'args', args, ...
               'disp_res',res, ...
               'omega', p.omega, ...
               'freq_Hz', p.omega/(2*pi), ...
               'power', args.power);
end