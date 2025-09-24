function S = pde_residual_single_test
% PDE_RESIDUAL_SINGLE_TEST Validate PDE solver consistency for flexible surferbot simulation
%
% DESCRIPTION:
%   Performs a single simulation of a flexible surferbot system and computes
%   residual norms to verify that the numerical solution satisfies the governing
%   partial differential equations and boundary conditions. This function serves
%   as a consistency check for the PDE solver implementation.
%
% SYNTAX:
%   S = pde_residual_single_test()
%
% INPUTS:
%   None (uses predefined test parameters)
%
% OUTPUTS:
%   S - Structure containing simulation results and residual analysis with fields:
%       x               - Spatial grid points in x-direction (m)
%       z               - Spatial grid points in z-direction (m) 
%       phi             - Velocity potential field
%       eta             - Free surface elevation
%       N_x             - Number of grid points in x-direction
%       M_z             - Number of grid points in z-direction
%       thrust_N        - Computed thrust force (N)
%       H               - Domain depth (m)
%       lapl_norm       - L2 norm of Laplacian residual in bulk fluid
%       bernoulli_norm  - L2 norm of Bernoulli equation residual at free surface
%       beam_norm       - Infinity norm of beam equation residual
%       no_pen_norm     - L2 norm of no-penetration BC residual at bottom
%       bc_left_norm    - L2 norm of left radiation BC residual  
%       bc_right_norm   - L2 norm of right radiation BC residual
%
% PHYSICS:
%   The function validates the following governing equations:
%   - Laplace equation: ?²? = 0 in the fluid domain
%   - Bernoulli equation at free surface including surface tension effects
%   - Flexible beam equation for raft dynamics with motor forcing
%   - No-penetration boundary condition at bottom: ??/?z = 0
%   - Radiation boundary conditions at domain edges: ??/?x ± ik? = 0
%
% TEST PARAMETERS:
%   - Raft length: 0.05 m
%   - Oscillation frequency: 10 Hz  
%   - Domain depth: 0.2 m
%   - Grid resolution: 201×100 points
%   - Flexible beam with bending stiffness EI = 10 N?m²
%
% VISUALIZATION:
%   Creates two figures:
%   1. Semi-log plot of free surface amplitude |?(x)|
%   2. Bar chart of residual norms for all PDE components
%
% EXAMPLE:
%   S = pde_residual_single_test();
%   fprintf('Maximum beam residual: %.2e\n', S.beam_norm);
%
% SEE ALSO:
%   flexible_surferbot_v2, getNonCompactFDmatrix2D
%
% AUTHOR: [Author name]
% DATE: [Date]

addpath('../src');

L_raft = 0.05;
base = struct( ...
    'sigma',0, 'rho',1000, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.3*L_raft/2, 'd',L_raft/2, ...
    'EI',1e-5, 'rho_raft',0.018*3.0, ...
    'domainDepth',0.5, 'n',401, 'M',200, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*10, 'ooa', 2, 'forcing_width', 0.01);

S = run_case(base);

% Plot |eta|
figure(1); clf; semilogy(S.x, abs(S.eta), 'k-','LineWidth',1.2);
hold on; semilogy(S.x(S.args.x_contact), abs(S.eta((S.args.x_contact))), 'r','LineWidth',1.5);
xlabel('x (m)'); ylabel('|{\eta}|'); grid on;
set(gca,'FontSize',14); set(gcf,'Position',[100 100 800 350]);
title('Free-surface amplitude');

% Residuals bar
figure(2); clf;
vals = [S.lapl_norm, S.bernoulli_norm, S.beam_norm, S.no_pen_norm, S.bc_left_norm, S.bc_right_norm];
bar(vals); set(gca,'YScale','log','XTickLabelRotation',20);
set(gca,'XTick',1:6,'XTickLabel',{'lapl','bernoulli','beam','no\_pen','bc\_left','bc\_right'});
ylabel('Residual norm'); grid on; set(gca,'FontSize',12);
set(gcf,'Position',[120 500 800 350]);
title('PDE and BC residuals');

% Summary table
T = table(S.H, S.thrust_N, S.lapl_norm, S.bernoulli_norm, S.beam_norm, ...
          S.no_pen_norm, S.bc_left_norm, S.bc_right_norm, ...
    'VariableNames', {'H_m','thrust_N','lapl','bernoulli','beam','no_pen','bc_left','bc_right'});
disp('=== PDE-consistency result ==='); disp(T);
end

function S = run_case(p)
[U, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'n',p.n,'M',p.M,'motor_inertia',p.motor_inertia,'BC',p.BC); %#ok<ASGLU>

ooa  = args.ooa;
[Dx, Dz] = getNonCompactFDmatrix2D(args.N, args.M, args.dx, args.dz, 1, ooa);
[Dxx, ~] = getNonCompactFDmatrix2D(args.N, args.M, args.dx, args.dz, 2, ooa);
Lapl     = Dxx + Dz*Dz;

% Bulk Laplacian ||?²?|| on interior
bulkMask = false(args.M, args.N); bulkMask(2:end-1, 2:end-1) = true;
lapl = reshape(Lapl * phi(:), args.M, args.N);
lapl_norm = norm(lapl(bulkMask));

% Free-surface Bernoulli on free nodes (left+right of raft)
xfreeNb = sum(~args.x_contact)/2 + 1;
phi_z   = reshape(args.phi_z, args.M, args.N);
[Dx2FS, ~] = getNonCompactFDmatrix(xfreeNb, args.dx, 2, ooa);
bern_fun = @(ph, phz) phz - args.sigma/(args.rho*args.g) * Dx2FS * phz ...
                    - args.omega^2/args.g * ph ...
                    - 4i*args.nu*args.omega/args.g * Dx2FS * ph;
ph_L  = phi(end, 1:xfreeNb).';      ph_R  = phi(end, end-xfreeNb+1:end).';
phz_L = phi_z(end, 1:xfreeNb).';    phz_R = phi_z(end, end-xfreeNb+1:end).';
bern  = [bern_fun(ph_L, phz_L); bern_fun(ph_R, phz_R)];
bernoulli_norm = norm(bern(2:end-1));

% Beam balance on contact
nc = sum(args.x_contact);
[Dx2c, ~] = getNonCompactFDmatrix(nc, args.dx, 2, ooa);
[Dx4c, ~] = getNonCompactFDmatrix(nc, args.dx, 4, ooa);
beamEqnIdx = sub2ind([args.M, args.N], repmat(args.M,1,nc), find(args.x_contact));
beam_res = ...
    args.EI/(1i*args.omega) * Dx4c * (eta(args.x_contact) * (1i*args.omega)) ...
  - args.rho_raft * args.omega/1i * (eta(args.x_contact) * (1i*args.omega)) ...
  + args.rho * args.d * ( 1i*args.omega * phi(beamEqnIdx).' ...
  + args.g/(1i*args.omega) * (eta(args.x_contact) * (1i*args.omega)) ...
  - 2*args.nu * Dx2c * phi(beamEqnIdx).' ) ...
  - args.motor_inertia * args.omega^2 * args.loads;
beam_norm = norm(beam_res, 2);

% Bottom no-penetration ||?z ?|| at z = -H (interior x)
bottomMask = false(args.M, args.N); bottomMask(1, 2:end-1) = true;
Dzphi = Dz * phi(:);
no_pen_norm = norm(Dzphi(bottomMask(:)));

% Left/right radiation ||?x ? ± i k ?||
phix = Dx * phi(:);
leftMask  = false(args.M, args.N); leftMask(:,1)  = true;
rightMask = false(args.M, args.N); rightMask(:,end)= true;
bc_left_norm  = norm(phix(leftMask(:))  - 1i*args.k*phi(leftMask(:)));
bc_right_norm = norm(phix(rightMask(:)) + 1i*args.k*phi(rightMask(:)));

S = struct('x',x,'z',z,'phi',phi,'eta',eta, ...
           'N_x',args.N,'M_z',args.M, ...
           'args', args, ...
           'thrust_N',args.thrust, ...
           'lapl_norm',lapl_norm, ...
           'bernoulli_norm',bernoulli_norm, ...
           'beam_norm',beam_norm, ...
           'no_pen_norm',no_pen_norm, ...
           'bc_left_norm',bc_left_norm, ...
           'bc_right_norm',bc_right_norm, ...
           'H', p.domainDepth);
end
