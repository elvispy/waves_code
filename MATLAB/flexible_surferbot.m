function [U, x, z, phi, eta, args] = flexible_surferbot(varargin)
    % Parse inputs
    p = inputParser;
    addParameter(p, 'sigma', 72.2e-3);
    addParameter(p, 'rho', 1000.);
    addParameter(p, 'omega', 2*pi*80.);
    addParameter(p, 'nu', 1e-6);
    addParameter(p, 'g', 9.81);
    addParameter(p, 'L_raft', 0.05);
    addParameter(p, 'motor_position', 0.6/5 * 0.05);
    addParameter(p, 'd', 0.03);
    addParameter(p, 'L_domain', 0.5);
    addParameter(p, 'EI', 3.0e9 * 3e-2 * 1e-4^3 / 12);
    addParameter(p, 'rho_raft', 0.018 * 3.);
    addParameter(p, 'domainDepth', 0.1);
    addParameter(p, 'n', 201);
    addParameter(p, 'M', 100);
    addParameter(p, 'motor_inertia', 0.13e-3 * 2.5e-3);
    addParameter(p, 'BC', 'radiative');
    parse(p, varargin{:});
    args = p.Results;

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
    k = dispersion_k(args.omega, args.g, args.domainDepth, args.nu, args.sigma, args.rho);
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
    z = linspace(0, -args.domainDepth, M) / L_c;
    dz = abs(z(2) - z(1));

    % Contact and free indices
    x_contact = abs(x) <= args.L_raft / (2 * L_c);
    H = sum(x_contact);
    x_free = abs(x) > args.L_raft / (2 * L_c);
    x_free(1) = false; x_free(end) = false;

    left_raft_boundary = floor((N - H)/2) + 1;
    right_raft_boundary = left_raft_boundary + H - 1;


    C = struct('C11',C11,'C12',C12,'C13',C13,'C14',C14, ...
               'C21',C21,'C22',C22,'C23',C23,'C24',C24,'C25',C25, ...
               'C26',C26,'C27',C27,'C31',C31,'C32',C32);
    loads = gaussian_load(args.motor_position/L_c,0.05,x(x_contact));
    [A_sparse, b_flat] = build_system(N, M, dx, dz, C, x_free, x_contact, ...
                                 loads, ...
                                 args.BC);

    % reshape RHS to a column vector (blocks stacked)
    b_sparse = reshape(b_flat.', [], 1);

    % now solve
    xsol = A_sparse \ b_sparse;

    
    % Post-processing
    if startsWith(args.BC, 'd')
        xsol = cat(2, zeros(1, M, 5), xsol, zeros(1, M, 5));
    elseif startsWith(args.BC, 'n')
        xsol = cat(2, xsol(1, :, :), xsol, xsol(end, :, :));
    end

    phi   = xsol(true(N, 1) & true(1, M) & reshape([1 0 0 0 0], 1, 1, 5));
    phi_x = xsol(true(N, 1) & true(1, M) & reshape([0 1 0 0 0], 1, 1, 5));


    prod2D = @(A, B) reshape(reshape(A, N*M, N*M) * reshape(B, N*M, N*M), N, M);


    ooa = 4;
    % eta from kinematic condition
    [~, Dz]  = getNonCompactFDmatrix2D(N,M,dx,dz,1,ooa);
    eta   = (1 / (1i * args.omega * t_c)) * Dz(1:N, :) * phi; % Only first 1:N points = surface
    eta_x = (1 / (1i * args.omega * t_c)) * Dz(1:N, :) * phi_x;

    % Pressure
    Dxx = getNonCompactFDmatrix(sum(x_contact),dx,2,ooa);
    P1 = (C24 + C26 * Dxx) * phi(x_contact, 1);
    p = C25 * eta(x_contact) + P1;

    %eta_x = Dx * eta(x_contact);
    weights = simpson_weights(H, dx);
    thrust = (args.d / L_c) * (weights * (-0.5 * real(p .* eta_x(x_contact))));
    thrust = thrust * F_c;
    thrust = thrust + args.sigma * args.d * (eta(left_raft_boundary + 1) - eta(left_raft_boundary)) / dx;
    thrust = thrust - args.sigma * args.d * (eta(right_raft_boundary) - eta(right_raft_boundary - 1)) / dx;

    U = (thrust^2 / thrust_factor)^(1/3);
    x = x * L_c;
    z = z * L_c;
    phi = full(reshape(phi * L_c^2 / t_c, N, M));
    eta = full(eta * L_c);
    args.x_contact = x_contact;
    args.loads = loads;
end
