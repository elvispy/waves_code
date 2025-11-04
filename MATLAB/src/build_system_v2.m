function [xsol, A, b] = build_system_v2(N, P, dx, dz, x_free, ...
                            x_contact, motor_weights, args)
%BUILD_SYSTEM   Assemble the full linear system for the coupled PDE?ODE
%               model of the flexible surferbot.
%
% This function constructs the full linear system matrix `A` and right-hand
% side vector `b` to solve a coupled fluid?structure interaction problem.
% It includes:
%   - Five primary PDE blocks (e.g., velocity potential, surface elevation)
%   - Euler?Bernoulli beam equation inside the raft
%   - Surface tension corner terms at raft boundaries
%   - Horizontal derivative definition constraints
%
% The system accounts for fluid?structure coupling, raft bending stiffness,
% motor actuation, and surface tension effects. It is discretized on a 2D
% grid with finite differences.
% -------------------------
% Inputs:
%   N              [int]    number of grid points in x-direction
%   P              [int]    number of grid points in z-direction
%   dx             [m]      horizontal grid spacing
%   dz             [m]      vertical grid spacing
%   x_free         [logical N x 1] mask for free-surface regions (true where surface is free)
%   x_contact      [logical N x 1] mask for raft-contact regions (true where surface contacts raft)
%   motor_weights  [N x 1]    vector of distributed motor force weights over raft region
%   args           [struct] parameters structure (see flexible_surferbot) including:
%                   .sigma         [N/m]    surface tension
%                   .rho           [kg/m^3] fluid density
%                   .omega         [rad/s]  actuation frequency
%                   .EI            [N*m^2]  bending stiffness of raft
%                   .rho_raft      [kg/m]   linear density of raft
%                   .motor_inertia [kg*m^2] motor rotational inertia
%                   .BC            boundary condition type ('radiative', etc.)
%
% -------------------------
% Outputs:
%   xsol  [5NM x 1]      solution vector ordering:
%                        [phi; eta = phi_z]
%   A     [5NM x 5NM]    sparse matrix, left-hand side of linear system
%   b     [5NM x 1]      right-hand side vector

% -------------------------
% Notes:
% - The PDEs include potential flow equations (Laplace + kinematic/dynamic BC),
%   as well as structure dynamics for raft deformation.
% - The final system A*xsol = b can be solved using sparse linear solvers.
% - The system respects both free-surface dynamics and rigid/body interactions.
%
% Author: Elvis Aguero
% Date: April, 2025
% ------------------------------------------------------------------------


% 0.  House-keeping
% ------------------------------------------------------------------------
NP     = N*P;
BCtype = args.BC;
Fr     = args.nd_groups.Fr;
Gamma  = args.nd_groups.Gamma;
We     = args.nd_groups.We;
kappa  = args.nd_groups.kappa;
Lambda = args.nd_groups.Lambda;
Re     = args.nd_groups.Re;

nbContact = sum(x_contact);
I_NP = speye(NP); I_CC = speye(nbContact, nbContact);


% ------------------------------------------------------------------------
% 2.  Logical masks  ?  row-index vectors
% ------------------------------------------------------------------------
topMask       = false(P,N); topMask(end, 2:N-1)    = true;
freeMask      = repmat(x_free, P, 1)  & topMask;
contactMask   = repmat(x_contact,P,1) & topMask;

bottomMask    = false(P,N); bottomMask(1, 2:N-1)   = true;
rightEdgeMask = false(P,N); rightEdgeMask(:, end)  = true;
leftEdgeMask  = false(P,N); leftEdgeMask(:, 1)     = true;
bulkMask      = false(P,N); bulkMask(2:P-1, 2:N-1) = true;

idxFreeSurf     = find(freeMask);
idxLeftFreeSurf = [P; idxFreeSurf(1:(end/2)); find(contactMask, 1, 'first')]; %idxLeftFreeSurf = idxLeftFreeSurf(1:(end-1));
nbLeft          = nnz(idxLeftFreeSurf);
idxRightFreeSurf= [find(contactMask, 1, 'last'); idxFreeSurf(((end/2)+1):end); NP]; %idxRightFreeSurf = idxRightFreeSurf(2:(end));
idxContact      = find(contactMask);  

idxBulk       = find(bulkMask);
idxBottom     = find(bottomMask);
idxLeftEdge   = find(leftEdgeMask);
idxRightEdge  = find(rightEdgeMask);


% ------------------------------------------------------------------------
% 3.  Initialise 3 by 3 sparse cell container
% ------------------------------------------------------------------------
S2D = repmat({sparse(NP,NP)}, 3, 3);
S2D{1, 3} = sparse(NP, nbContact);
S2D{2, 3} = sparse(NP, nbContact);
S2D{3, 3} = sparse(nbContact, nbContact);
S2D{3, 1} = sparse(nbContact, NP);
S2D{3, 2} = sparse(nbContact, NP);
%S2D{i, j} refers to the block of equations i, and j = 1 correspond to the
%variable phi, whle j=2 corresponds to the variable phi_z, and j = 3 to M

% ------------------------------------------------------------------------
% 4.  Equation 1: Bernoulli on free surface  
% ------------------------------------------------------------------------
L = idxLeftFreeSurf; 
[DxFree,  ~] = getNonCompactFDmatrix(nbLeft,1,1,args.ooa);
[DxxFree, ~] = getNonCompactFDmatrix(nbLeft,1,2,args.ooa);
% (1:(end-1))
S2D{1,1}(L(2:end-1),L) =  dx^2 * I_NP(L(2:end-1), L) + 4.0j/Re * DxxFree(2:end-1, :);
S2D{1,2}(L(2:end-1),L) =  -dx^2/Fr^2 * I_NP(L(2:end-1), L) + 1/(We * Gamma) * DxxFree(2:end-1, :);

R = idxRightFreeSurf;
S2D{1,1}(R(2:end-1),R) =  dx^2 * I_NP(R(2:end-1), R) + 4.0j/Re * DxxFree(2:end-1, :);
S2D{1,2}(R(2:end-1),R) =  -dx^2/Fr^2 * I_NP(R(2:end-1), R) + 1/(We * Gamma) * DxxFree(2:end-1, :);

% ------------------------------------------------------------------------
% 4b.  Euler-beam equations inside raft  (rows = rowRaft)
%       overwrite the rows that were zeroed out above
% ------------------------------------------------------------------------
CC = idxContact;
%[DxRaft,  ~] = getNonCompactFDmatrix(sum(x_contact),1,1,args.ooa);
[Dx2Raft, ~] = getNonCompactFDmatrix(sum(x_contact),1,2,args.ooa);
[Dx3Raft, ~] = getNonCompactFDmatrix(sum(x_contact),1,3,args.ooa);
%[Dx4Raft, ~] = getNonCompactFDmatrix(sum(x_contact),1,4,args.ooa);


S2D{1, 1}(CC, CC) = 1.0i * Lambda * Gamma * dx^2 * I_NP(CC, CC) + 2*Gamma*Lambda / Re * Dx2Raft;
S2D{1, 2}(CC, CC) = (1.0i - 1.0i * Gamma * Lambda/Fr^2) * dx^2 * I_NP(CC, CC); %+ (-1.0i * kappa/dx^2) * Dx4Raft;
S2D{1, 3}(CC, :)  = -1.0i * Dx2Raft;

S2D{1, 1}(idxContact([1 end]), :) = 0; 
S2D{1, 2}(idxContact([1 end]), :) = 0;

% Boundary conditions: No stress on end
S2D{1, 2}(idxContact(1), CC)   = Dx3Raft(1, :); 
S2D{1, 2}(idxContact(1), L)    = S2D{1, 2}(idxContact(1), L) ...
    - dx^2 * Lambda / (kappa * We) * DxFree(end, :);

S2D{1, 2}(idxContact(end), CC) = Dx3Raft(end, :);
S2D{1, 2}(idxContact(end), R)  = S2D{1, 2}(idxContact(end), R) ...
    - dx^2 * Lambda / (kappa * We) * DxFree(1, :);

% Momentum equation for rigid-case and stability

S2D{3, 2}(:, CC) = + 1.0i * Dx2Raft;
S2D{3, 3} = (dx^2/kappa) * I_CC;

if args.test == true % Dirichlet BC conditions for testing
    S2D{1, 1}(idxContact, :) = 0;
    S2D{1, 2}(idxContact, :) = I_NP(idxContact, :);
    %S4D{1, 2}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 3}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 4}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 5}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
end
% ------------------------------------------------------------------------
% 5.  Equation 2:  Laplace in the bulk  
% ------------------------------------------------------------------------
[Dx,  Dz]  = getNonCompactFDmatrix2D(N,P,1,1,1,args.ooa); 
[Dxx, Dzz] = getNonCompactFDmatrix2D(N,P,1,1,2,args.ooa);

S2D{1,1}(idxBulk,:)  = Dxx(idxBulk,:) + Dzz(idxBulk,:) * (dx/dz)^2;
%S2D{1,2}(idxBulk, :) = Dz(idxBulk, :);

% ------------------------------------------------------------------------
% 6.  Equation 3:  bottom impermeability  
% ------------------------------------------------------------------------
S2D{1,2}(idxBottom,:) = I_NP(idxBottom,:);

% ------------------------------------------------------------------------
% 7.  Equation 4: radiative BC 
% ------------------------------------------------------------------------
S2D{1,2}(idxLeftEdge,:)  = 0;
S2D{1,2}(idxRightEdge,:) = 0;
if startsWith(BCtype,'n','IgnoreCase',true)
    S2D{1,1}(idxLeftEdge,:)  = Dx(idxLeftEdge, :);
    S2D{1,1}(idxRightEdge,:) = Dx(idxRightEdge, :);
elseif startsWith(BCtype,'r','IgnoreCase',true)
    S2D{1,1}(idxLeftEdge,:)  = -1.0i * args.k * args.L_raft*I_NP(idxLeftEdge,:)  * dx + Dx(idxLeftEdge, :);
    S2D{1,1}(idxRightEdge,:) =  1.0i * args.k * args.L_raft*I_NP(idxRightEdge,:) * dx + Dx(idxRightEdge, :);
end
% ------------------------------------------------------------------------
% 8.  Derivative constraints
% ------------------------------------------------------------------------
% row-2:  phi_z  - Dz phi   = 0
S2D{2,1} = Dz;      S2D{2,2} = - dz *I_NP;


% ------------------------------------------------------------------------
% 9.  Assemble RHS
% ------------------------------------------------------------------------
b = zeros(2*NP + nbContact, 1);
if args.test == true
    b(idxContact) = -0.01; 
else
    % OBS: Because sum(motor_weights) = force / F_c, to have units of 
    % N/m we would need to input motor_weights/dx. But we are multiplying
    % this whole equation by dx^2, with an overall contribution of dx. 
    % WE WILL ADD THIS dx after solving for x. 
    b(idxContact) = - dx^2 * motor_weights(:).';   % only on raft contact nodes
    % THe next two equations are boundary conditions:
    b(idxContact(1:2)) = 0;
    b(idxContact((end-1):end)) = 0;
end

% ------------------------------------------------------------------------
% 10.  Boundary conditions 
% ------------------------------------------------------------------------
% 
if startsWith(BCtype,'d','IgnoreCase',true)         % Dirichlet in x
    keep = ~edgeMask(:);
    S2D  = cellfun(@(A) A(keep,keep), S2D,'UniformOutput',false);
    b    = b(:,keep);
elseif startsWith(BCtype,'n','IgnoreCase',true)     % Neumann lumping
    % already incorporated in Dx via one-sided stencils; no action
end

A = cell2mat(S2D);                 % (5*NM) x (5*NM) sparse
b_sparse = reshape(b.', [], 1);    % RHS as column vector
info = whos('A');
if info.bytes > 2* 2147483648; warning('Matrix A is taking %.2g GiB of space. Consider downgrading.', info.bytes/2147483648); end

% WE ADD THIS dx because it was rescaled on the motor weights. 
% See lines 205-208
xsol = solve_system(A, b_sparse);
xsol = xsol(1:(2*NP));


end % end function definition