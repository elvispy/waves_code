
%% We call the surferbot
% ------------------------------------------------------------------------
% Example: call flexible_surferbot_v2 with every parameter specified
% ------------------------------------------------------------------------
addpath './src'
[U, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma'         , 0*72.2e-3      , ...   % [N m?¹] surface tension
    'rho'           , 1000.0       , ...   % [kg m?³] water density
    'omega'         , 2*pi*10      , ...   % [rad s?¹] drive frequency
    'nu'            , 0*1.0e-6       , ...   % [m² s?¹] kinematic viscosity
    'g'             , 10*9.81         , ...   % [m s?²] gravity
    'L_raft'        , 0.05         , ...   % [m] raft length
    'motor_position', 1.9e-2/2   , ...   % [m] motor x-position (from -L/2 to L/2)
    'd'             , 0.03         , ...   % [m] raft depth (spanwise)
    'EI'            , 100*3.0e9 * 3e-2 * (9.9e-4)^3 / 12 , ...  % [N m²] bending stiffness
    'rho_raft'      , 0.018*3.0    , ...   % [kg m?¹] linear mass
    'L_domain'      , 0.2          , ...   % [m] domain length
    'domainDepth'   , 0.2          , ...   % [m] water depth
    'n'             , 201          , ...   % grid points in the raft
    'M'             , 100          , ...   % gird points in the z direction
    'motor_inertia' , 0.13e-3*2.5e-3, ...  % [kg m²] motor inertia
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

