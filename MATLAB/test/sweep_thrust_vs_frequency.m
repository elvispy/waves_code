% sweep_thrust_and_Sxx_vs_frequency.m
%
% Nondimensional plot:
%   x:  ω √(L/g)
%   y:  F / (ρ g L^2)
% Curves: numerical thrust and your S_xx expression based on eta.

addpath '../src/'
% Frequency range (Hz)
f_values     = 80:80;
omega_values = 2*pi*f_values;
L_raft       = 0.05;

% Preallocate
thrust_values   = zeros(size(omega_values));
momentum_values = zeros(size(omega_values));
Sxx_values      = zeros(size(omega_values));
LH_values       = zeros(size(omega_values));

for ii = 1:numel(omega_values)
    
    omega = omega_values(ii);

    % Run simulation (defaults elsewhere)
    [~, x, z, phi, eta, args] = flexible_surferbot_v2('sigma',72.2e-3, 'rho',1000, 'nu',0*1e-6, 'g',9.81, ...
            'L_raft',L_raft, 'motor_position',0.24*L_raft/2, 'd',0.03, ...
            'EI',10*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*10, ...
            'domainDepth',0.1, 'L_domain', 2.0*L_raft, 'n',1001, 'M',600, ...
            'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
            'omega',omega, 'ooa', 4);
        
        
    
    % Thrust from solver
    thrust_values(ii) = args.thrust/args.d; % To compare it to other methods

    % Your Sxx using ends of eta
    rho   = args.rho;
    g     = args.g;
    sigma = args.sigma;
    k     = real(args.k);
    
    

    % Calculate velocity as gradient of potential
    dx = abs(x(1) - x(2)); dz = abs(z(1) - z(2));
    [Dx, ~] = getNonCompactFDmatrix2D(args.N,args.M,dx,dz,1,args.ooa);
    u = reshape(Dx * reshape(phi, args.M * args.N, 1), args.M, args.N); 

    
    Sxx_values(ii) = (rho*g/4 + 3/4*sigma*k^2) * (abs(eta(1))^2 - abs(eta(end))^2);
    % For gravity waves
    LH_values(ii)  = 1/4 * rho * omega^2 / k *   (abs(eta(1))^2 - abs(eta(end))^2);
    % Integrating the z velocity
    momentum_values(ii) =  rho/2 * trapz(z, abs(u(:, 1)).^2 - abs(u(:, end)).^2);
    
    % We add surface tension contributions
    Dx = getNonCompactFDmatrix(10, 1, 1, args.ooa)/args.dx;
    eta_x_1 = Dx * eta(1:10); eta_x_end = Dx * eta((end-9):end);
    sf_radiation = args.sigma/4 * ( abs(eta_x_end(end))^2 - abs(eta_x_1(1))^2);
    
    momentum_values(ii) = momentum_values(ii) + sf_radiation;
    %thrust_values(ii)   = thrust_values(ii);

    fprintf("%d, %.2e; ", ii, args.omega^2 - k * g);

end
disp('');

% --- Nondimensionalization ---
L     = args.L_raft;
rho   = args.rho;
g     = args.g;

omega_star  = omega_values/(2*pi) ; %.* sqrt(L/g);     
Fm_scale    = 1; %rho * g * L;                 
thrust_star = thrust_values   ./ Fm_scale;
Sxx_star    = Sxx_values      ./ Fm_scale;
LH_star     = LH_values       ./ Fm_scale;
mom_star    = momentum_values ./ Fm_scale;
% --- Plot (loglog) ---
figure(1); %clf;
semilogx(omega_star, thrust_star, 'k-',  'LineWidth', 2); hold on;   % numerical thrust
semilogx(omega_star, LH_star,     'b--', 'LineWidth', 2); hold on;
semilogx(omega_star, mom_star,    'r--', 'LineWidth', 2); hold on;
semilogx(omega_star, Sxx_star,    'g--', 'LineWidth', 2); %hold off;


grid on; set(gca, 'Box','on', 'TickDir','out');
set(gca, 'FontSize', 16)
xlabel('$f$', 'Interpreter','Latex');
ylabel('$F/m$', 'Interpreter','Latex');
title('Nondimensional thrust vs. frequency');

legend({'Numerical thrust', 'LH', 'Momentum', 'Radiation Stress'}, 'Location','best', 'Interpreter','tex');


fprintf('\n   f [Hz]     w*          M (N/m)          FT(N/m)        LH (N/m)         Sxx (N/m)\n');
fprintf('------------------------------------------------------------------------\n');

for ii = 1:numel(f_values)
    fprintf('%8.2f   %8.3g   %14.4e   %14.4e   %14.4e   %14.4e\n', ...
        f_values(ii), omega_star(ii), mom_star(ii), thrust_star(ii), LH_star(ii), Sxx_star(ii));
end

