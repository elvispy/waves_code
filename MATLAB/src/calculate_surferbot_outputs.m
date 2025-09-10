% Filename: calculate_surferbot_outputs.m

function [U, power, thrust, eta] = calculate_surferbot_outputs(args, phi, phi_z)
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
    % Get finite difference matrices
    [Dx, Dz] = getNonCompactFDmatrix2D(N, M, dx_adim, dz_adim, 1, args.ooa); 
    [Dx2, ~] = getNonCompactFDmatrix2D(N, M, dx_adim, dz_adim, 2, args.ooa);

    if nargin < 3
        phi_z = reshape(Dz * phi(:), M, N);
    end
    
    % Surface elevation (eta) and its derivative
    eta_adim   = (1 / (1i * args.omega * args.t_c)) * phi_z(end, :).'; 
    eta_x_adim = (1 / (1i * args.omega * args.t_c)) * Dx(M:M:end, :) * phi_z(:);
    
    % Pressure
    P1_adim = (1.0i * args.nd_groups.Gamma * phi(:) ...
        - 2 * args.nd_groups.Gamma / args.nd_groups.Re * Dx2 * phi(:));
    P1_adim = P1_adim(M:M:end, 1);
    P1_adim = P1_adim(args.x_contact, :);
    p_adim = - 1.0i * args.nd_groups.Gamma / args.nd_groups.Fr^2 * eta_adim(args.x_contact) ...
       + P1_adim; %Gamma / Fr^2 = d* g / (L^2 * omega^2)
    
    % Thrust
    weights = simpson_weights(sum(args.x_contact), dx_adim);
    thrust_adim = (args.d / args.L_c) * (weights * (-0.5 * real(p_adim .* eta_x_adim(args.x_contact))));
    thrust = real(thrust_adim * F_c); 
    
    % Power
    power = -(0.5 * args.omega * args.L_c * F_c) * weights * (imag(eta_adim(args.x_contact)) .* (- args.loads));
    
    % Final Velocity (U)
    thrust_factor = 4/9 * args.nu * (args.rho * args.d)^2 * args.L_raft;
    U = (thrust^2 / thrust_factor)^(1/3);
    
    % Dimensionalize eta for output
    eta = full(eta_adim * args.L_c);
end