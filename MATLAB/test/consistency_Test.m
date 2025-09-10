% This script tests that the results have the potential to be independent
% of what it should be:
%  - Depth
%  - Place where ou evaluate wave amplitude
%  - Ensure mesh is of correct order

addpath '../src/'

% Frequency range (Hz)
f_values     = 20;
omega_values = 2*pi*f_values;

% Preallocate
thrust_values   = zeros(size(omega_values));
momentum_values = zeros(size(omega_values));
Sxx_values      = zeros(size(omega_values));
LH_values       = zeros(size(omega_values));

for ii = 1:numel(omega_values)
    
    omega = omega_values(ii);

    % Run simulation (defaults elsewhere)
    L_raft = 0.1; %(meters)
    [~, x, z, phi, eta, args] = flexible_surferbot_v2( ...
        'sigma'         , 0*72.2e-3      , ...   % [N m?¹] surface tension
        'rho'           , 1000.0       , ...   % [kg m?³] water density
        'omega'         , omega         , ...   % [rad s?¹] drive frequency
        'nu'            , 0*1.0e-6       , ...   % [m² s?¹] kinematic viscosity
        'g'             , 10*9.81        , ...   % [m s?²] gravity
        'L_raft'        , L_raft         , ...   % [m] raft length
        'motor_position', 0.5*L_raft/2   , ...   % [m] motor x-position (from -L/2 to L/2)
        'd'             , L_raft/2       , ...   % [m] raft depth (spanwise)
        'EI'            , 100*3.0e9 * 3e-2 * (9.9e-4)^3 / 12 , ...  % [N m²] bending stiffness
        'rho_raft'      , 0.018*3.0    , ...   % [kg m?¹] linear mass
        'domainDepth'   , 0.2          , ...   % [m] water depth
        'n'             , 201          , ...   % grid points in the raft
        'M'             , 100          , ...   % gird points in the z direction
        'motor_inertia' , 0.13e-3*2.5e-3, ...  % [kg m²] motor inertia
        'BC'            , 'radiative'        );% boundary-condition type

    %[~, x, z, phi, eta, args] = flexible_surferbot_v2('omega', omega, ...
    %    'sigma', 0, 'nu', 0, 'domainDepth', 1.5, 'L_raft', 0.1, ...
    %    'motor_position', 0.5 * 0.3, 'EI', 100, 'g', 9.81);

    % Thrust from solver
    thrust_values(ii) = args.thrust;

    % Your Sxx using ends of eta
    rho   = args.rho;
    g     = args.g;
    sigma = args.sigma;
    k     = real(args.k);

    if 2*pi / k / args.dx <= 10
        warning('Maybe timestep in x direction is too big');
    end
    
    %if args.H / args.dz <= 50
    %    warning('Maybe timestep in z direction is too big');
    %end
    
    
    %Sxx_values(i) = (rho*g/4 + 3/4*sigma*k^2) * (abs(eta(1))^2 - abs(eta(end))^2);
    LH_values(ii)  = 1/4 * rho * omega^2 / k *   (abs(eta(2))^2 - abs(eta(end-1))^2);

    % Calculate velocity as gradient of potential
    dx = abs(x(1) - x(2)); dz = abs(z(1) - z(2));
    [Dx, ~] = getNonCompactFDmatrix2D(args.M,args.N,dx,dz,1,args.ooa);
    u = reshape(Dx * reshape(phi, args.M * args.N, 1), args.M, args.N); 

    momentum_values(ii) = rho * trapz(z, abs(u(:, 2)).^2 - abs(u(:, end-1)).^2);

    fprintf("%d, %.2e; \n", ii, args.omega^2 - k * g);

    figure(ii);
    semilogy(x, abs(eta)); xlabel('x (m)'); ylabel('y (um)'); 
    set(gca, 'FontSize', 16)
    pbaspect([3 1 1])   % width : height : depth

end