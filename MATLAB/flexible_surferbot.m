function [U, x, z, phi, eta] = flexible_surferbot(varargin)
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
    addParameter(p, 'L_domain', 1.0);
    addParameter(p, 'EI', 3.0e9 * 3e-2 * 1e-4^3 / 12);
    addParameter(p, 'rho_raft', 0.018 * 3.);
    addParameter(p, 'domainDepth', 0.1);
    addParameter(p, 'n', 21);
    addParameter(p, 'M', 10);
    addParameter(p, 'motor_inertia', 0.13e-3 * 2.5e-3);
    addParameter(p, 'BC', 'neumann');
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
    z = (args.domainDepth / L_c) * (logspace(0, log10(2), M) - 1);
    dz = z(2) - z(1);

    % Contact and free indices
    x_contact = abs(x) <= args.L_raft / (2 * L_c);
    H = sum(x_contact);
    x_free = abs(x) > args.L_raft / (2 * L_c);
    x_free(1) = false; x_free(end) = false;

    left_raft_boundary = floor((N - H)/2) + 1;
    right_raft_boundary = left_raft_boundary + H - 1;

    % Operators
    d_dx = Diff(0, dx, 2, [N, M]);
    d_dz = Diff(1, dz, 2, [N, M]);

    % I_NM tensor: identity in (i,j)-(k,l)
    I_NM = zeros(N, M, N, M);
    for i = 1:N
        for j = 1:M
            I_NM(i, j, i, j) = 1;
        end
    end

    % Build equations
    % E1: Bernoulli
    E1 = zeros(N, M, N, M, 5);
    for d = 1:5
        if d == 1
            E1(:, :, :, :, d) = C11 * d_dz + C13 * I_NM;
        elseif d == 3
            E1(:, :, :, :, d) = C12 * d_dz + C14 * I_NM;
        end
    end
    % Remove contact equations on surface
    for i = 1:N
        if ~x_free(i)
            E1(i, 1, :, :, :) = 0;
        end
    end
    E1(:, 2:end, :, :, :) = 0;

    % E2: Laplace
    E2 = zeros(N, M, N, M);
    lap = d_dx^2 + d_dz^2;
    E2(:, :, :, :) = lap;
    E2([1, end], :, :, :) = 0;
    E2(:, [1, end], :, :) = 0;
    E2_full = zeros(N, M, N, M, 5);
    E2_full(:, :, :, :, 1) = E2;

    % E3: Bottom BC
    E3 = zeros(N, M, N, M, 5);
    E3(:, end, :, :, 1) = d_dz(:, end, :, :);
    E3([1, end], :, :, :) = 0;

    % E4: Radiative BCs
    E4 = zeros(N, M, N, M, 5);
    for j = 1:M
        E4(1, j, 1, j, 1) = -C32;
        E4(1, j, 1, j, 2) = C31;
        E4(end, j, end, j, 1) = C32;
        E4(end, j, end, j, 2) = C31;
    end

    % Final operator
    A = E1 + E2_full + E3 + E4;

    % b vector (RHS)
    b = zeros(5, N, M);
    weights = gaussian_load(args.motor_position / L_c, 0.05, x(x_contact));
    b(1, x_contact, 1) = -C23 * weights;

    % Boundary trimming
    if startsWith(args.BC, 'd')
        A = A(:, 2:end-1, :, 2:end-1, :);
        b = b(:, 2:end-1, :);
    elseif startsWith(args.BC, 'n')
        A(:, :, :, end-1, :) = A(:, :, :, end-1, :) + A(:, :, :, end, :);
        A(:, :, :, 2, :) = A(:, :, :, 2, :) + A(:, :, :, 1, :);
        A = A(:, 2:end-1, :, 2:end-1, :);
        b = b(:, 2:end-1, :);
    end

    % Solve the system
    xsol = solve_tensor_system(A, b);

    % Post-processing
    if startsWith(args.BC, 'd')
        xsol = cat(2, zeros(1, M, 5), xsol, zeros(1, M, 5));
    elseif startsWith(args.BC, 'n')
        xsol = cat(2, xsol(1, :, :), xsol, xsol(end, :, :));
    end

    phi = xsol(:, :, 1);
    phi_x = xsol(:, :, 2);

    % eta from kinematic condition
    eta = (1 / (1i * args.omega * t_c)) * squeeze(multiprod(d_dz, phi));
    eta_x = (1 / (1i * args.omega * t_c)) * squeeze(multiprod(d_dz, phi_x));

    % Pressure
    d_dx1D = Diff(0, dx, 2, [H, 1]);
    P1 = (C24 + C26 * d_dx1D^2) * phi(x_contact, 1);
    p = C25 * eta(x_contact) + P1;

    eta_x = d_dx1D * eta(x_contact);
    weights = simpson_weights(H, dx);
    thrust = (args.d / L_c) * (weights * (-0.5 * real(p .* eta_x)));
    thrust = thrust * F_c;
    thrust = thrust + args.sigma * args.d * (eta(left_raft_boundary + 1) - eta(left_raft_boundary)) / dx;
    thrust = thrust - args.sigma * args.d * (eta(right_raft_boundary) - eta(right_raft_boundary - 1)) / dx;

    U = (thrust^2 / thrust_factor)^(1/3);
    x = x * L_c;
    z = z * L_c;
    phi = phi * L_c^2 / t_c;
    eta = eta * L_c;
end
