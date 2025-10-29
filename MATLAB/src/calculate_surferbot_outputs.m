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
    f_adim = 0*args.loads; % (Motor force loads already adimensional!)
    
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
    eta_adim     = (1/(1i*args.omega*args.t_c)) * phi_z(end,:).';  % length-N
    eta_raft     = eta_adim(contact_mask);                                  % length-Nr
    eta_x_raft   = D1r * eta_raft;
    xL = find(args.x_contact, 1); xR = find(args.x_contact, 1, 'last');
    eta_x_surf_L = D1r(end, (end-9):end) * eta_adim((xL - 9):xL);
    eta_x_surf_R = D1r(1,   1:10        ) * eta_adim(xR     :(xR+9));

    % Surface potential restricted to raft
    phi_surf   = phi(end,:).';           % length-N
    phi_raft = phi_surf(contact_mask);            % length-Nr

    % Pressure on raft
    P1_r   = (1i*args.nd_groups.Gamma) * phi_raft ...
           - (2*args.nd_groups.Gamma/args.nd_groups.Re) * (D2r * phi_raft);
    p_adim = -1i*args.nd_groups.Gamma/args.nd_groups.Fr^2 * eta_raft + P1_r;

    
    % Minus because of sign convention: if the right of the raft has higher
    % amplitude, the raft moves to the left (negative sign). Also, beware
    % of the game of "thrust force applied to the body" and "thrust force
    % as felt by the body"
    w_r         = simpson_weights(Nr, dx_adim);
    Q_adim      = f_adim - args.d / args.L_c * p_adim; % Total load applied on the raft
    thrust_adim = ( - w_r * (real(Q_adim) .* real(eta_x_raft) + imag(Q_adim) .* imag(eta_x_raft))/2); 
    %disp(trapz(real(p_adim .* eta_x_r))* dx_adim);
    %figure(7); plot(linspace(0, 1, Nr), real(p_adim .* eta_x_r)); hold on;
    
    thrust_adim = thrust_adim ...
        + args.sigma * args.d / F_c/4 * (abs(eta_x_surf_L)^2 - abs(eta_x_surf_R)^2); 
    thrust      = thrust_adim * F_c;
    
    % Final Velocity (U) 
    thrust_factor = 4/9 * args.nu * (args.rho * args.d)^2 * args.L_raft; 
    U = (thrust^2 / thrust_factor)^(1/3);

    % Power (loads already raft-only)
    power = -(0.5 * args.omega * args.L_c * F_c) * w_r * ...
            (imag(eta_raft) .* (-args.loads(:)));

    % Outputs
    eta = full(eta_adim * args.L_c);
    p   = full(p_adim * F_c / args.L_c^2);

end