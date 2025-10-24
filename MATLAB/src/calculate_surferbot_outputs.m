% Filename: calculate_surferbot_outputs.m

function [U, power, thrust, eta, p] = calculate_surferbot_outputs(args, phi, phi_z)
%CALCULATE_SURFERBOT_OUTPUTS Computes U, power, and other outputs from simulation results.
%
%   Inputs:
%       args  - Struct with all simulation parameters and constants.
%       phi   - Non-dimensional potential field [M x N matrix].
%       phi_z - Non-dimensional z-derivative of potential [M x N matrix].
%
%   Outputs:
%       U      - Final forward velocity (m/s).
%       power  - Power exerted by the motor (W).
%       thrust - Net thrust generated (N).
%       eta    - Dimensional surface elevation (m).

    % --- Unpack necessary variables from args ---
    N = args.N;
    M = args.M;
    dx_adim = args.dx / args.L_c; % Use non-dimensional dx from the run
    dz_adim = args.dz / args.L_c; % Use non-dimensional dz from the run
    F_c = args.m_c * args.L_c / args.t_c^2;
    
    % --- Main Calculations ---
        % ---- Ensure phi_z is available (backward compatible) ----
    if nargin < 3 || isempty(phi_z)
        [~, Dz] = getNonCompactFDmatrix2D(N, M, dx_adim, dz_adim, 1, args.ooa);
        phi_z = reshape(Dz * phi(:), M, N);
    end

    % ---- Raft-only derivatives on the surface ----
    contact_mask = args.x_contact(:);                 % length-N logical
    Nr   = sum(contact_mask);

    % 1D FD operators on the raft grid only (scale by actual spacing)
    [D1r, ~] = getNonCompactFDmatrix(Nr, 1, 1, args.ooa);   % d/dx
    [D2r, ~] = getNonCompactFDmatrix(Nr, 1, 2, args.ooa);   % d^2/dx^2
    D1r = D1r / dx_adim;
    D2r = D2r / dx_adim^2;

    % Surface elevation ? and ?_x restricted to raft
    eta_adim = (1/(1i*args.omega*args.t_c)) * phi_z(end,:).';  % length-N
    eta_r    = eta_adim(contact_mask);                                  % length-Nr
    eta_x_r  = D1r * eta_r;

    % Surface potential restricted to raft
    phi_s   = phi(end,:).';           % length-N
    phi_s_r = phi_s(contact_mask);            % length-Nr

    % Pressure on raft
    P1_r   = (1i*args.nd_groups.Gamma) * phi_s_r ...
           - (2*args.nd_groups.Gamma/args.nd_groups.Re) * (D2r * phi_s_r);
    p_adim = -1i*args.nd_groups.Gamma/args.nd_groups.Fr^2 * eta_r + P1_r;

    % Raft-only quadrature
    w_r         = simpson_weights(Nr, dx_adim);
    % Minus because of sign convention: if the right of the raft has higher
    % amplitude, the raft moves to the left (negative sign)
    thrust_adim = - (args.d/args.L_c) * (w_r * (-0.5 * real(p_adim .* eta_x_r))); 
    %disp(trapz(real(p_adim .* eta_x_r))* dx_adim);
    %figure(7); plot(linspace(0, 1, Nr), real(p_adim .* eta_x_r)); hold on;
    thrust      = real(thrust_adim * F_c);
    
    % Final Velocity (U) 
    thrust_factor = 4/9 * args.nu * (args.rho * args.d)^2 * args.L_raft; 
    U = (thrust^2 / thrust_factor)^(1/3);

    % Power (loads already raft-only)
    power = -(0.5 * args.omega * args.L_c * F_c) * w_r * ...
            (imag(eta_r) .* (-args.loads(:)));

    % Outputs
    eta = full(eta_adim * args.L_c);
    p   = full(p_adim * F_c / args.L_c^2);

end