% sweeper.M
%
% Description:
%   End-to-end demo that:
%     1) calls FLEXIBLE_SURFERBOT_V2 with all key parameters explicitly set,
%     2) visualizes surface deflection η(x) and contact loads,
%     3) visualizes the complex velocity potential φ(x,z),
%     4) verifies PDE, boundary conditions, and beam–fluid coupling by reporting residual norms.
%
% How to run:
%   - Place this script in the project’s examples/ or scripts/ folder.
%   - Ensure ../src contains FLEXIBLE_SURFERBOT_V2 and FD helpers:
%       getNonCompactFDmatrix2D, getNonCompactFDmatrix.
%   - Open MATLAB, cd to this script’s folder, then run it.
%
% What it produces:
%   - Figure 1: η(x) in micrometers with contact nodes highlighted (red) and contact force quivers.
%   - Figure 2: φ(x,z) imaginary part as a color map with contours of Im(φ)/(ω L_raft^2).
%   - Command Window prints:
%       * “Velocity is … mm/s”
%       * Norms of Laplacian residual, Bernoulli condition on the free surface,
%         beam equilibrium residual, bottom no-penetration, and left/right radiation BCs.
%
% Key inputs set here (units in SI unless noted):
%   sigma           Surface tension [N/m]. Use 0*sigma to disable its effect here.
%   rho             Fluid density [kg/m^3].
%   omega           Drive angular frequency [rad/s].
%   nu              Kinematic viscosity [m^2/s]. Using 0*… disables viscous terms here.
%   g               Gravity [m/s^2].
%   L_raft          Raft length [m].
%   motor_position  Motor x-location from −L_raft/2 to +L_raft/2 [m].
%   d               Raft spanwise depth [m].
%   EI              Bending stiffness [N·m^2].
%   rho_raft        Linear mass of raft [kg/m].
%   L_domain        Computational domain length in x [m].
%   domainDepth     Water depth in z [m].
%   n               Grid points along the raft (x-direction).
%   M               Grid points in the vertical (z-direction).
%   motor_inertia   Motor rotational inertia [kg·m^2].
%   BC              Boundary condition type on lateral boundaries, e.g., 'radiative'.
%
% Outputs from FLEXIBLE_SURFERBOT_V2:
%   U       Complex surge velocity magnitude [m/s] (printed in mm/s).
%   x,z     Spatial grids in x and z.
%   phi     Complex velocity potential field φ(x,z).
%   eta     Complex free-surface/raft deflection η(x).
%   args    Struct of derived parameters, masks, operators, and metadata (e.g., k, dx, dz,
%           x_contact, phi_z, EI, rho, d, ω, L_raft). Used for diagnostics and plotting.
%
% Diagnostics performed:
%   - Laplacian(φ) in the fluid bulk → reports ||∇²φ||.
%   - Linearized Bernoulli on the free surface away from edges → reports norm of residual.
%   - Beam equation on contact region, including fluid loading and motor inertia → reports
%     ||beam − J_motor ω² loads||_∞.
%   - Bottom no-penetration: Dz φ = 0 at z = −domainDepth → residual norm.
%   - Left/right radiation-type BCs: φ_x ∓ i k φ = 0 → residual norms.
%
% Adjustable visualization settings:
%   - Axis scaling for η: 'scale' factor (default 1e6 for micrometers) and automatic y-limits.
%   - Colormap and contour levels for φ.
%   - Figure sizes via set(gcf,'Position',…).
%
% Tips:
%   - Start with the provided parameters; then vary omega, EI, or L_raft to study sensitivity.
%   - If residual norms are large, increase n and M or adjust domain size to reduce reflection.
%   - Ensure all inputs use SI units; labels convert η to micrometers only for plotting.

addpath '../src'
[U, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma'         , 0*72.2e-3      , ...   % [N m?�] surface tension
    'rho'           , 1000.0       , ...   % [kg m?�] water density
    'omega'         , 2*pi*10      , ...   % [rad s?�] drive frequency
    'nu'            , 0*1.0e-6       , ...   % [m� s?�] kinematic viscosity
    'g'             , 10*9.81         , ...   % [m s?�] gravity
    'L_raft'        , 0.05         , ...   % [m] raft length
    'motor_position', 1.9e-2/2   , ...   % [m] motor x-position (from -L/2 to L/2)
    'd'             , 0.03         , ...   % [m] raft depth (spanwise)
    'EI'            , 100*3.0e9 * 3e-2 * (9.9e-4)^3 / 12 , ...  % [N m�] bending stiffness
    'rho_raft'      , 0.018*3.0    , ...   % [kg m?�] linear mass
    'L_domain'      , 0.2          , ...   % [m] domain length
    'domainDepth'   , 0.2          , ...   % [m] water depth
    'n'             , 201          , ...   % grid points in the raft
    'M'             , 100          , ...   % gird points in the z direction
    'motor_inertia' , 0.13e-3*2.5e-3, ...  % [kg m�] motor inertia
    'BC'            , 'radiative'        );% boundary-condition type

close all;
scale = 1e+6;
plot(x, real(eta) * scale, 'b'); hold on;
plot(x(args.x_contact), real(eta(args.x_contact)) * scale, 'r', 'LineWidth', 2)
set(gcf, 'Position', [52 557 1632 420]);
set(gca, 'FontSize', 16);
xlabel('x (m)'); xlim([-.1, .1]);
sc = max(abs(eta), [], 'all') * 1e+6 * 1.4; scaleY = round(sc, 1, 'significant');
ylabel('y (um)'); ylim([-scaleY, scaleY]);
quiver(x(args.x_contact), (real(eta(args.x_contact)).') * scale, zeros(1, sum(args.x_contact)), ...
    args.loads.'/5e+4 * scale, 0, 'MaxHeadSize', 1e-6);

figure(2); hold on;
set(gca, 'FontSize', 20);
colormap winter
shading interp
plt = pcolor(x', z, imag(phi));
colorbar;
contour(x', z, imag(phi) / (args.omega* args.L_raft^2), ...
        8, 'k'); 
set(gcf, 'Position', [56 49 1638 424]);
set(plt, 'edgecolor', 'none')
title("Phi field")
fprintf("Velocity is %g mm/s\n", U*1000)


%% Checking that solution satisfies PDE
ooa  = args.ooa;
[Dxx, ~] = getNonCompactFDmatrix2D(args.N, args.M, args.dx, args.dz, 2, ooa);
[Dx, Dz] = getNonCompactFDmatrix2D(args.N, args.M, args.dx, args.dz, 1, ooa);
Lapl = Dxx + Dz*Dz;

bulkIdx = false(args.M, args.N); bulkIdx(2:(end-1), 2:(end-1)) = true; bulkIdx = reshape(bulkIdx, [], 1);
lapl = Lapl * phi(:); lapl = lapl(bulkIdx);
fprintf("Norm of laplacian: %g\n", norm(lapl));


xfreeNb = sum(~args.x_contact)/2+1;
phi_z = reshape(args.phi_z, args.M, args.N);
I_FM = speye(xfreeNb);
[DxxFree, ~] = getNonCompactFDmatrix(xfreeNb, args.dx, 2, ooa);

Qty = args.M * xfreeNb;
phi_free_left    = phi(end, 1:xfreeNb).';
phi_free_right   = phi(end, (end-xfreeNb+1):end).';
phi_z_free_left  = phi_z(end, 1:xfreeNb).';
phi_z_free_right = phi_z(end, (end-xfreeNb+1):end).';
bernoulli = @(phi, phi_z) phi_z - args.sigma/(args.rho * args.g) * DxxFree * phi_z - ...
    args.omega^2/args.g * phi - 4 * 1i * args.nu * args.omega / (args.g ) * DxxFree * phi;
bern = bernoulli(phi_free_left, phi_z_free_left) + bernoulli(phi_free_right, phi_z_free_right);
fprintf("Norm of bernoulli: %g\n", norm(bern(2:end-1)));


[DxxRaft, ~] = getNonCompactFDmatrix(sum(args.x_contact), args.dx, 2, ooa);
[Dx4Raft, ~] = getNonCompactFDmatrix(sum(args.x_contact), args.dx, 4, ooa);
beamIdx = find(repmat(args.x_contact,args.M,1));
beamEqnIdx = find(repmat((1:args.M) == args.M, sum(args.x_contact), 1).');
I_HM = speye(sum(args.x_contact) * args.M);

beam = args.EI / (1i * args.omega) * Dx4Raft * (eta(args.x_contact) * (1i * args.omega)) ...
    - args.rho_raft * args.omega / 1i * (eta(args.x_contact)*(1i * args.omega)) ...
    + args.rho * args.d * ( 1i * args.omega * phi(beamEqnIdx) + ...
    args.g / (1i * args.omega) * (eta(args.x_contact)*(1i * args.omega)) - 2 * args.nu * DxxRaft* phi(beamEqnIdx));
% vars = load('vars.mat');
% S = args.rho * args.d * ( 1i * args.omega * eye(numel(beamEqnIdx)) - 2 * args.nu * DxxRaft);
% norm(S * args.L_c^2 * args.t_c / args.m_c - vars.S2D{1, 1}(vars.contactMask, vars.contactMask), 1)
% S2 = args.EI / (1i * args.omega) * Dx4Raft - args.rho_raft * args.omega / 1i * (eye(sum(args.x_contact)))+ args.rho * args.d * ( args.g / (1i * args.omega) * (eye(sum(args.x_contact))) );
% norm(S2 * args.L_c * args.t_c / args.m_c - vars.S2D{1, 2}(vars.contactMask, vars.contactMask), 1)


if args.test == true
    fprintf("Norm of Dirichlet BC on the top: %g\n", norm(phi(beamEqnIdx) - mean(phi(beamEqnIdx))));
else
    fprintf("Norm of beam: %g\n", norm(beam - args.motor_inertia * args.omega^2 * args.loads, inf));
end


noPenetration = Dz * phi(:); bottomIdx = false(args.M, args.N); bottomIdx(1, 2:(end-1)) = 1; bottomIdx = bottomIdx(:);
fprintf("Norm of no penetration: %g\n", norm(Dz(bottomIdx, :) * phi(:)));

phix = Dx * phi(:); leftIdx = false(args.M, args.N); rightIdx = false(args.M, args.N);
leftIdx(:, 1) = true; rightIdx(:, end) = true;

fprintf("Norm of BC (left): %g\n",  norm(phix(leftIdx)  - 1i * args.k * phi(leftIdx)) );
fprintf("Norm of BC (right): %g\n", norm(phix(rightIdx) + 1i * args.k * phi(rightIdx)) );

