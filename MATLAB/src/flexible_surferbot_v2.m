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
    addParameter(p, 'motor_position', 0.6/2.5 * 0.05); % [m] motor position along the raft (fraction × L_raft)
    addParameter(p, 'd', 0.03);                        % [m] depth of surferbot (third dimension, z-direction)
    addParameter(p, 'EI', 3.0e9 * 3e-2 * 9e-4^3 / 12); % [N·m^2] bending stiffness
    addParameter(p, 'rho_raft', 0.018 * 3.);           % [kg/m] mass per unit length of the raft

    % --- Domain settings ---
    addParameter(p, 'L_domain', 0.1);               % [m] total length of the simulation domain
    addParameter(p, 'domainDepth', 0.1);            % [m] depth of the simulation domain (second dimention, y-direction)

    % --- Discretization parameters ---
    addParameter(p, 'n', 201);                      % [unitless] number of grid points in x in the raft
    addParameter(p, 'M', 100);                      % [unitless] number of grid points in z

    % --- Motor parameters ---
    addParameter(p, 'motor_inertia', 0.13e-3 * 2.5e-3);  % [kg·m^2] motor rotational inertia
    addParameter(p, 'forcing_width', 0.05);             % [fraction of L_raft] width of Gaussian forcing

    % --- Boundary condition ---
    addParameter(p, 'BC', 'radiative');             % Boundary condition type (radiative, Neuman, Dirichlet)

    
    parse(p, varargin{:});
    args = p.Results;
    args.ooa = 4; % Define finite difference accuracy in space
    args.test = false;

    % Derived parameters
    force = args.motor_inertia * args.omega^2;
    L_c   = args.L_raft;
    t_c   = 1 / args.omega;
    m_c   = args.rho_raft * L_c;
    F_c   = m_c * L_c / t_c;

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
    xsol = build_system_v2(N, M, dx, dz, x_free, x_contact, loads, args);
    
    %% Post-processing
    phi = reshape(xsol(1:(N*M)), M, N);
    phi_z = reshape(xsol((N*M+1):end), M, N);
    % 
    % ooa = args.ooa;
    % % eta from kinematic condition
    % [Dx, ~]  = getNonCompactFDmatrix2D(N,M,dx,dz,1,ooa);
    % 
    % eta   = (1 / (1i * args.omega * t_c)) * phi_z(end, :).'; 
    % eta_x = (1 / (1i * args.omega * t_c)) * Dx(M:M:end, :) * phi_z(:);
    % 
    % % Pressure
    % 
    % [Dx2, ~]  = getNonCompactFDmatrix2D(N,M,dx,dz,2,ooa);
    % %P1 = (C24 + C26 * Dxx) * phi(x_contact, 1);
    % P1 = (C24*phi(:) + C26 * Dx2*phi(:)); P1 = P1(M:M:end, 1); P1 = P1(x_contact, :);
    % p = C25 * eta(x_contact) + P1; 
    % 
    % %eta_x = Dx * eta(x_contact);
    % weights = simpson_weights(H, dx);
    % thrust = (args.d / L_c) * (weights * (-0.5 * real(p .* eta_x(x_contact)))); disp(norm((imag(eta(x_contact)) .*  (-C23 * loads))));
    % thrust = thrust * F_c;
    % 
    % % NO SURFACE TENSION EFFECT x-dynamics to linear order!
    % %thrust = thrust + args.sigma * args.d * (eta(left_raft_boundary + 1) - eta(left_raft_boundary)) / dx;
    % %thrust = thrust - args.sigma * args.d * (eta(right_raft_boundary) - eta(right_raft_boundary - 1)) / dx; 
    % 
    % % Calculating applied power:
    % power = -(0.5 * args.omega * L_c * F_c) * weights * (imag(eta(x_contact)) .*  (-C23 * loads));
    % 
    % 
    % thrust = real(thrust); 
    % U = (thrust^2 / thrust_factor)^(1/3);
    x = x * L_c;
    z = z * L_c;    
    args.x_contact = x_contact;
    args.loads = loads;
    args.N = N; args.M = M; args.dx = dx * L_c; args.dz = dz * L_c;
    args.t_c = t_c; args.L_c = L_c; args.m_c = m_c;

    [U, power, thrust, eta] = calculate_surferbot_outputs(args, phi, phi_z);
    %disp(norm(U2 - U) + norm(power2-power) + norm(thrust2 - %thrust) - norm(eta2 - eta));
    args.k = k; args.power = power; args.thrust = thrust;
    phi = full(reshape(phi * L_c^2 / t_c, M, N));
    args.phi_z = full(reshape(phi_z * L_c / t_c, M, N));
end
