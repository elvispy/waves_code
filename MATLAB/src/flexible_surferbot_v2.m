function [U, x, z, phi, eta, args] = flexible_surferbot_v2(varargin)
    % Parse inputs
    p = inputParser;
    
    % --- Physical constants ---
    addParameter(p, 'sigma', 72.2e-3);              % [N/m] surface tension of water
    addParameter(p, 'rho', 1000.);                  % [kg/m^3] density of water
    addParameter(p, 'omega', 2*pi*80.);             % [rad/s] driving frequency (80 Hz)
    addParameter(p, 'nu', 1e-6);                    % [m^2/s] kinematic viscosity of water
    addParameter(p, 'g', 9.81);                     % [m/s^2] gravitational acceleration

    % --- Raft properties ---
    addParameter(p, 'L_raft', 0.05);                   % [m] length of the raft
    addParameter(p, 'motor_position', 0.6/2.5 * 0.05); % [m] motor position along the raft (fraction x L_raft)
    addParameter(p, 'd', 0.03);                        % [m] depth of surferbot (third dimension, z-direction)
    addParameter(p, 'EI', 3.0e9 * 3e-2 * 9e-4^3 / 12); % [N m^2] bending stiffness
    addParameter(p, 'rho_raft', 0.018 * 3.);           % [kg/m] mass per unit length of the raft

    % --- Domain settings ---
    addParameter(p, 'L_domain', nan);               % [m] total length of the simulation domain
    addParameter(p, 'domainDepth', 0.1);            % [m] depth of the simulation domain (second dimention, y-direction)

    % --- Solver settings ---
    addParameter(p, 'n', 51);                      % [unitless] number of grid points in x in the raft
    addParameter(p, 'M', 400);                      % [unitless] number of grid points in z
    addParameter(p, 'ooa', 4);                      % [unitless] finite difference order accuracy
    addParameter(p, 'test', false);                 % [boolean]  whether to run self-diagnostic tests
    % --- Motor parameters ---
    addParameter(p, 'motor_inertia', 0.13e-3 * 2.5e-3);  % [kg/m^2] motor rotational inertia
    addParameter(p, 'forcing_width', 0.05);             % [fraction of L_raft] width of Gaussian forcing

    % --- Boundary condition ---
    addParameter(p, 'BC', 'radiative');             % Boundary condition type (radiative, Neuman, Dirichlet)
    
    
    parse(p, varargin{:});
    args = p.Results;
    %args.ooa = 4; % Define finite difference accuracy in space
    
    if isnan(args.L_domain); args.L_domain = args.L_raft * 10; end

    % Derived parameters
    force = args.motor_inertia * args.omega^2;
    L_c   = args.L_raft;
    t_c   = 1 / args.omega;
    m_c   = args.rho_raft * L_c;
    F_c   = m_c * L_c / t_c^2;

    % --- Non-dimensional groups ---
    Gamma  = args.rho * args.L_raft^2 / args.rho_raft;
    Fr     = sqrt(args.L_raft * args.omega^2 / args.g);
    Re     = args.L_raft^2 * args.omega / args.nu;
    kappa  = args.EI / (args.rho_raft * args.L_raft^4 * args.omega^2);
    We     = args.rho_raft * args.L_raft * args.omega^2 / args.sigma;
    Lambda = args.d / args.L_raft;

    nd_groups = struct( ...
        'Gamma',  Gamma, ...
        'Fr',     Fr, ...
        'Re',     Re, ...
        'kappa',  kappa, ...
        'We',     We, ...
        'Lambda', Lambda);

    % Store nondimensional groups for later use
    args.nd_groups = nd_groups;

    % Wavenumber
    k = dispersion_k(args.omega, args.g, args.domainDepth, args.nu, args.sigma, args.rho);
    args.k = k;
    if tanh(args.k * args.domainDepth) < 0.95; warning('Domain depth not enough for dispersison relation'); end
    if 2*pi/k / (args.L_raft / args.n) <= 10 
        
        args.n = ceil(10 / (2*pi/k) * args.L_raft);
        args.n = args.n + mod(args.n, 2) + 1;
        warning('Number of points in x direction too small. Changing n to %d', args.n); 
    end
    % Grid
    L_domain_adim = ceil(args.L_domain / L_c);
    if mod(L_domain_adim, 2) == 0
        L_domain_adim = L_domain_adim + 1;
    end
    N = round(args.n * L_domain_adim / (args.L_raft / L_c));
    M = args.M;

    x = linspace(-L_domain_adim/2, L_domain_adim/2, N);
    dx = x(2) - x(1);
    
    
    z = linspace(-args.domainDepth, 0, M) / L_c;
    dz = abs(z(2) - z(1));

    % Contact and free indices
    x_contact = abs(x) <= args.L_raft / (2 * L_c);
    x_free    = abs(x) >  args.L_raft / (2 * L_c);
    x_free(1) = false; x_free(end) = false;

    % Loads at which the motor applies a force to the raft
    loads = force / F_c * gaussian_load(args.motor_position/L_c, args.forcing_width, x(x_contact));

    % Solve system
    solution = build_system_v2(N, M, dx, dz, x_free, x_contact, loads, args);
    
    %% Post-processing
    phi = reshape(solution(1:(N*M)), M, N);
    phi_z = reshape(solution((N*M+1):end), M, N);

    x = x * L_c;
    z = z * L_c;    
    args.x_contact = x_contact;
    args.loads = loads;
    args.N = N; args.M = M; args.dx = dx * L_c; args.dz = dz * L_c;
    args.t_c = t_c; args.L_c = L_c; args.m_c = m_c;

    [U, power, thrust, eta] = calculate_surferbot_outputs(args, phi, phi_z);
    
    args.k = k; args.power = power; args.thrust = thrust;
    phi = full(reshape(phi * L_c^2 / t_c, M, N));
    args.phi_z = full(reshape(phi_z * L_c / t_c, M, N));
    
end
