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
    addParameter(p, 'L_raft', 0.05);                % [m] length of the raft
    addParameter(p, 'motor_position', 0.6/2.5 * 0.05);% [m] motor position along the raft (fraction × L_raft)
    addParameter(p, 'd', 0.03);                     % [m] depth of surferbot (third dimension, z-direction)
    addParameter(p, 'EI', 3.0e9 * 3e-2 * 1e-4^3 / 12); % [N·m^2] bending stiffness (Young?s modulus × I)
    addParameter(p, 'rho_raft', 0.018 * 3.);        % [kg/m] mass per unit length of the raft

    % --- Domain settings ---
    addParameter(p, 'L_domain', 0.1);               % [m] total length of the simulation domain
    addParameter(p, 'domainDepth', 0.1);            % [m] depth of the simulation domain (second dimention, y-direction)

    % --- Discretization parameters ---
    addParameter(p, 'n', 201);                      % [unitless] number of grid points in x
    addParameter(p, 'M', 100);                      % [unitless] number of modes or Fourier components, etc.

    % --- Motor parameters ---
    addParameter(p, 'motor_inertia', 0.13e-3 * 2.5e-3); % [kg·m^2] rotational inertia of motor

    % --- Boundary condition ---
    addParameter(p, 'BC', 'radiative');             % Boundary condition type (radiative, Neuman, Dirichlet)

    
    parse(p, varargin{:});
    args = p.Results;
    args.ooa = 4; % Define finite difference accuracy in space
    args.test = false;

    % Derived parameters
    force = args.motor_inertia * args.omega^2;
    L_c = args.L_raft;
    t_c = 1 / args.omega;
    m_c = args.rho * L_c^3;
    F_c = m_c * L_c / t_c^2;

    % Non-dimensional constants
    C11 = 1.0;
    C12 = -args.sigma / (args.rho * args.g * L_c^2);
    C13 = -args.omega^2 / args.g * L_c;
    C14 = -4i * args.nu * args.omega / (args.g * L_c);

    C21 = args.EI * t_c / (1i * args.omega * m_c * L_c^3);
    C22 = -args.rho_raft * args.omega * L_c * t_c / (1i * m_c);
    C23 = -force / (m_c * L_c / t_c^2);
    C24 = (args.rho * args.d * t_c * 1i * args.omega * L_c^2) / m_c;
    C25 = (args.rho * args.d * t_c * args.g * L_c) / (m_c * 1i * args.omega);
    C26 = -(args.rho * args.d * t_c * 2 * args.nu) / m_c;
    C27 = -args.sigma * args.d * t_c / (1i * args.omega * m_c * L_c);

    % Wavenumber
    k = dispersion_k(args.omega, args.g, args.domainDepth, args.nu, args.sigma, args.rho); args.k = k;
    C31 = 1;
    C32 = 1i * k * L_c;

    thrust_factor = 4/9 * args.nu * (args.rho * args.d)^2 * args.L_raft;

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
    H = sum(x_contact);
    x_free = abs(x) > args.L_raft / (2 * L_c);
    x_free(1) = false; x_free(end) = false;

    %left_raft_boundary = find(x_contact, 1, 'first');
    %right_raft_boundary = find(x_contact, 1, 'last');


    coeffs = struct('C11',C11,'C12',C12,'C13',C13,'C14',C14, ...
               'C21',C21,'C22',C22,'C23',C23,'C24',C24,'C25',C25, ...
               'C26',C26,'C27',C27,'C31',C31,'C32',C32);
    loads = gaussian_load(args.motor_position/L_c,0.05,x(x_contact));
    
    % Solve system
    xsol = build_system_v2(N, M, dx, dz, coeffs, x_free, x_contact, ...
                                 loads, ...
                                 args);
    
    % Post-processing
    if startsWith(args.BC, 'd')
        xsol = cat(2, zeros(1, M, 5), xsol, zeros(1, M, 5));
    end

    phi = reshape(xsol(1:(N*M)), M, N);
    phi_z = reshape(xsol((N*M+1):end), M, N);

    ooa = args.ooa;
    % eta from kinematic condition
    [Dx, ~]  = getNonCompactFDmatrix2D(N,M,dx,dz,1,ooa);
    
    eta   = (1 / (1i * args.omega * t_c)) * phi_z(end, :).';
    eta_x = (1 / (1i * args.omega * t_c)) * Dx(M:M:end, :) * phi_z(:);

    % Pressure
    
    [Dx2, ~]  = getNonCompactFDmatrix2D(N,M,dx,dz,2,ooa);
    %P1 = (C24 + C26 * Dxx) * phi(x_contact, 1);
    P1 = (C24*phi(:) + C26 * Dx2*phi(:)); P1 = P1(M:M:end, 1); P1 = P1(x_contact, :);
    p = C25 * eta(x_contact) + P1;

    %eta_x = Dx * eta(x_contact);
    weights = simpson_weights(H, dx);
    thrust = (args.d / L_c) * (weights * (-0.5 * real(p .* eta_x(x_contact))));
    thrust = thrust * F_c;
    
    % NO SURFACE TENSION AFFECT x-dynamics to linear order!
    %thrust = thrust + args.sigma * args.d * (eta(left_raft_boundary + 1) - eta(left_raft_boundary)) / dx;
    %thrust = thrust - args.sigma * args.d * (eta(right_raft_boundary) - eta(right_raft_boundary - 1)) / dx; 
    
    thrust = real(thrust);
    U = (thrust^2 / thrust_factor)^(1/3);
    x = x * L_c;
    z = z * L_c;
    phi = full(reshape(phi * L_c^2 / t_c, M, N));
    eta = full(eta * L_c);
    args.x_contact = x_contact;
    args.loads = loads;
    args.N = N; args.M = M; args.dx = dx * L_c; args.dz = dz * L_c;
    args.coeffs = coeffs; args.t_c = t_c; args.L_c = L_c; args.m_c = m_c;
    args.phi_z = full(reshape(phi_z * L_c / t_c, M, N));
    args.k = k;
end
