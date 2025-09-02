% sweep_thrust_and_Sxx_vs_frequency.m
%
% Nondimensional plot:
%   x:  ω √(L/g)
%   y:  F / (ρ g L^2)
% Curves: numerical thrust and your S_xx expression based on eta.

addpath '../src/'
% Frequency range (Hz)
f_values     = 5:3:15;
omega_values = 2*pi*f_values;

% Preallocate
thrust_values   = zeros(size(omega_values));
momentum_values = zeros(size(omega_values));
Sxx_values      = zeros(size(omega_values));
LH_values       = zeros(size(omega_values));

for i = 1:numel(omega_values)
    
    omega = omega_values(i);

    % Run simulation (defaults elsewhere)
    [~, x, z, phi, eta, args] = flexible_surferbot_v2('omega', omega, ...
        'sigma', 0, 'nu', 0, 'domainDepth', 1.5, 'L_raft', 0.5, ...
        'motor_position', 0.5 * 0.3, 'motor_inertia', 'EI', 10, 'g', 9.81);

    % Thrust from solver
    thrust_values(i) = args.thrust;

    % Your Sxx using ends of eta
    rho   = args.rho;
    g     = args.g;
    sigma = args.sigma;
    k     = real(args.k);
    
    %Sxx_values(i) = (rho*g/4 + 3/4*sigma*k^2) * (abs(eta(1))^2 - abs(eta(end))^2);
    LH_values(i)  = 1/4 * rho * omega^2 / k *   (abs(eta(2))^2 - abs(eta(end-1))^2);

    dx = abs(x(1) - x(2)); dz = abs(z(1) - z(2));
    [Dx, ~] = getNonCompactFDmatrix2D(args.M,args.N,dx,dz,1,args.ooa);
    u = reshape(Dx * reshape(phi, args.M * args.N, 1), args.M, args.N); 

    momentum_values(i) = rho * trapz(z, abs(u(:, 2)).^2 - abs(u(:, end-1)).^2);

    fprintf("%d, %.2e;", i, args.omega^2 - k * g);

end
disp('');

% --- Nondimensionalization ---
L     = args.L_raft;
rho   = args.rho;
g     = args.g;

omega_star  = omega_values .* sqrt(L/g);     % ω √(L/g)
F_scale     = rho * g * L^2;                 % ρ g L^2
thrust_star = thrust_values   ./ F_scale;
%Sxx_star    = Sxx_values      ./ F_scale;
LH_star     = LH_values       ./ F_scale;
mom_star    = momentum_values ./ F_scale;
% --- Plot (log–log) ---
figure(1); clf;
semilogx(omega_star, thrust_star, 'k-',  'LineWidth', 2); hold on;   % numerical thrust
semilogx(omega_star, LH_star,     'b--', 'LineWidth', 2); hold on;
semilogx(omega_star, mom_star,    'r--', 'LineWidth', 2); hold off;
%semilogx(omega_star, Sxx_star,   'r--', 'LineWidth', 2); hold off;


grid on; set(gca, 'Box','on', 'TickDir','out');
set(gca, 'FontSize', 16)
xlabel('$\omega \sqrt{L/g}$', 'Interpreter','Latex');
ylabel('$F /\rho g L^2$', 'Interpreter','Latex');
title('Nondimensional thrust vs. frequency');

legend({'Numerical thrust', 'LH', 'Momentum'}, 'Location','best', 'Interpreter','tex');

% (optional) print a quick table to console
fprintf('\n  f [Hz]   ω√(L/g)     M/(ρgL^2)     F_T/(ρgL^2)      LH\n');
for i = 1:numel(f_values)
    fprintf('%7.2f   %8.3g    %12.4e    %12.4e   %12.4e\n', ...
        f_values(i), omega_star(i), mom_star(i), thrust_star(i), LH_star(i));
end
