% sweep_thrust_and_Sxx_vs_frequency.m
%
% Nondimensional plot:
%   x:  ω √(L/g)
%   y:  F / (ρ g L^2)
% Curves: numerical thrust and your S_xx expression based on eta.

addpath '../src/'
% Frequency range (Hz)
f_values     = 5:5:25;
omega_values = 2*pi*f_values;

% Preallocate
thrust_values = zeros(size(omega_values));
Sxx_values    = zeros(size(omega_values));
LH_values     = zeros(size(omega_values));

for i = 1:numel(omega_values)
    fprintf("%d; ", i);
    omega = omega_values(i);

    % Run simulation (defaults elsewhere)
    [~, ~, ~, ~, eta, args] = flexible_surferbot_v2('omega', omega, ...
        'sigma', 0, 'nu', 0, 'domainDepth', 0.2, 'L_raft', 0.05, 'EI', 0.1, 'g', 100);

    % Thrust from solver
    thrust_values(i) = args.thrust;

    % Your Sxx using ends of eta
    rho   = args.rho;
    g     = args.g;
    sigma = args.sigma;
    k     = real(args.k);
    Sxx_values(i) = (rho*g/4 + 3/4*sigma*k^2) * (abs(eta(1))^2 - abs(eta(end))^2);
    LH_values(i)  = 1/2 * rho * omega^2 / k *   (abs(eta(1))^2 - abs(eta(end))^2);
end
disp('');

% --- Nondimensionalization ---
L     = args.L_raft;
rho   = args.rho;
g     = args.g;

omega_star  = omega_values .* sqrt(L/g);     % ω √(L/g)
F_scale     = rho * g * L^2;                 % ρ g L^2
thrust_star = thrust_values ./ F_scale;
Sxx_star    = Sxx_values    ./ F_scale;
LH_star     = LH_values     ./ F_scale;

% --- Plot (log–log) ---
figure(1); clf;
semilogx(omega_star, thrust_star, 'k-',  'LineWidth', 2); hold on;   % numerical thrust
semilogx(omega_star, LH_star,     'b--', 'LineWidth', 2); hold on;
semilogx(omega_star, Sxx_star,    'r--', 'LineWidth', 2); hold off;


grid on; set(gca, 'Box','on', 'TickDir','out');
set(gca, 'FontSize', 16)
xlabel('$\omega \sqrt{L/g}$', 'Interpreter','Latex');
ylabel('$F /\rho g L^2$', 'Interpreter','Latex');
title('Nondimensional thrust vs. frequency');

legend({'Numerical thrust', 'S_{xx} scaling'}, 'Location','best', 'Interpreter','tex');

% (optional) print a quick table to console
fprintf('\n  f [Hz]   ω√(L/g)     F_T/(ρgL^2)      Sxx/(ρgL^2)\n');
for i = 1:numel(f_values)
    fprintf('%7.2f   %8.3g    %12.4e   %12.4e\n', ...
        f_values(i), omega_star(i), thrust_star(i), Sxx_star(i));
end
