function [xsol, A, b] = build_system_v2(N, M, dx, dz, x_free, ...
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
%
% -------------------------
% Inputs:
%   N              [int] number of grid points in x-direction
%   M              [int] number of grid points in z-direction
%   dx             [m] horizontal grid spacing
%   dz             [m] vertical grid spacing
%   x_free         [logical N�1] mask for free-surface regions (true where surface is free)
%   x_contact      [logical N�1] mask for raft-contact regions (true where surface contacts raft)
%   motor_weights  [N�1] vector of distributed motor force weights over raft region
%   args           [struct] parameters structure (see flexible_surferbot) including:
%                     .sigma         [N/m] surface tension
%                     .rho           [kg/m^3] fluid density
%                     .omega         [rad/s] actuation frequency
%                     .EI            [N�m^2] bending stiffness of raft
%                     .rho_raft      [kg/m] linear density of raft
%                     .motor_inertia [kg�m^2] motor rotational inertia
%                     .BC            boundary condition type ('radiative', etc.)
%
% -------------------------
% Outputs:
%   xsol  [5NM � 1] solution vector ordering:
%              [phi; eta = phi_z]
%   A     [5NM � 5NM] sparse matrix, left-hand side of linear system
%   b     [5NM � 1] right-hand side vector
%
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
NM   = N*M;
BCtype = args.BC;
Fr = args.nd_groups.Fr;
Gamma = args.nd_groups.Gamma;
We = args.nd_groups.We;
kappa = args.nd_groups.kappa;
Lambda = args.nd_groups.Lambda;
Re = args.nd_groups.Re;

I_NM = speye(NM);

% ------------------------------------------------------------------------
% 1.  Finite-difference operators (1-D ? 2-D via Kronecker) 
% ------------------------------------------------------------------------
[Dx,  Dz] = getNonCompactFDmatrix2D(N,M,dx,dz,1,args.ooa); % ?/?x, ?/?z
[Dxx, ~]  = getNonCompactFDmatrix2D(N,M,dx,dz,2,args.ooa); % ?2/?x2


% ------------------------------------------------------------------------
% 2.  Logical masks  ?  row-index vectors
% ------------------------------------------------------------------------
topMask       = false(M,N); topMask(end, 2:N-1)       = true;
freeMask      = repmat(x_free, M, 1)  & topMask;
contactMask   = repmat(x_contact,M,1) & topMask;

bottomMask    = false(M,N); bottomMask(1, 2:N-1)      = true;
rightEdgeMask = false(M,N); rightEdgeMask(:, end)     = true;
leftEdgeMask  = false(M,N); leftEdgeMask(:, 1)        = true;
bulkMask      = false(M,N); bulkMask(2:M-1, 2:N-1)    = true;

idxFreeSurf     = find(freeMask);
idxLeftFreeSurf = [M; idxFreeSurf(1:(end/2)); find(contactMask, 1, 'first')]; %idxLeftFreeSurf = idxLeftFreeSurf(1:(end-1));
nbLeft          = nnz(idxLeftFreeSurf);
idxRightFreeSurf= [find(contactMask, 1, 'last'); idxFreeSurf(((end/2)+1):end); NM]; %idxRightFreeSurf = idxRightFreeSurf(2:(end));
idxContact      = find(contactMask);  

idxBulk       = find(bulkMask);
idxBottom     = find(bottomMask);
idxLeftEdge   = find(leftEdgeMask);
idxRightEdge  = find(rightEdgeMask);


% ------------------------------------------------------------------------
% 3.  Initialise 5�5 sparse cell container
% ------------------------------------------------------------------------
S2D = repmat({sparse(NM,NM)}, 2, 2);

% ------------------------------------------------------------------------
% 4.  Equation 1: Bernoulli on free surface  
% ------------------------------------------------------------------------
L = idxLeftFreeSurf; 
[DxxFree, ~] = getNonCompactFDmatrix(nbLeft,dx,2,args.ooa);
S2D{1,1}(L(2:end-1),L) =  I_NM(L(2:end-1), L) + 4.0j/Re * DxxFree(2:end-1, :);
S2D{1,2}(L(2:end-1),L) =  -Fr^(-2) * I_NM(L(2:end-1), L) + 1/(We * Gamma)*DxxFree(2:end-1, :);

R = idxRightFreeSurf;
S2D{1,1}(R(2:end-1),R) =  I_NM(R(2:end-1), R) + 4.0j/Re *DxxFree(2:end-1, :);
S2D{1,2}(R(2:end-1),R) =  -1/Fr^2 * I_NM(R(2:end-1), R) + 1/(We * Gamma) * DxxFree(2:end-1, :);

% ------------------------------------------------------------------------
% 4b.  Euler-beam equations inside raft  (rows = rowRaft)
%       overwrite the rows that were zeroed out above
% ------------------------------------------------------------------------
CC = idxContact;
[DxxRaft, ~] = getNonCompactFDmatrix(sum(x_contact),dx,2,args.ooa);
[Dx4Raft, ~] = getNonCompactFDmatrix(sum(x_contact),1 ,4,args.ooa);
S2D{1, 1}(idxContact(1), :) = 0; S2D{1, 1}(idxContact(end), :) = 0;
S2D{1, 2}(idxContact(1), :) = 0; S2D{1, 2}(idxContact(end), :) = 0;
S2D{1, 1}(CC, CC) = -2*Gamma*Lambda / Re * DxxRaft + 1.0i * Lambda * Gamma * I_NM(CC, CC);
S2D{1, 2}(CC, CC) = (1.0i - 1.0i * Gamma * Lambda/Fr^2) * I_NM(CC, CC) + (-1.0i * kappa/dx^4) * Dx4Raft;

% ------------------------------------------------------------------------
% 4c.  Surface-tension corner corrections
% ------------------------------------------------------------------------
%   right raft boundary, x > 0
rightBdry = find(contactMask, 1, 'last');      % right edge of raft

% right raft corner contribution
S2D{1,2}(rightBdry, idxRightFreeSurf) = ...
      S2D{1,2}(rightBdry, idxRightFreeSurf) ...
    + (1.0i * Lambda / (We*dx)) * (I_NM(rightBdry+M, idxRightFreeSurf) - I_NM(rightBdry, idxRightFreeSurf))/dx;

% left raft corner contribution
leftBdry  = find(contactMask, 1, 'first');     % left  edge of raft
S2D{1,2}(leftBdry, idxLeftFreeSurf) = ...
    S2D{1,2}(leftBdry, idxLeftFreeSurf) ...
  + (1.0i * Lambda / (We*dx)) * (I_NM(leftBdry, idxLeftFreeSurf) - I_NM(leftBdry-M, idxLeftFreeSurf))/dx;


if args.test == true % Dirichlet BC conditions for testing
    S2D{1, 1}(idxContact, :) = 0;
    S2D{1, 2}(idxContact, :) = I_NM(idxContact, :);
    %S4D{1, 2}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 3}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 4}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 5}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
end
% ------------------------------------------------------------------------
% 5.  Equation 2:  Laplace in the bulk  
% ------------------------------------------------------------------------
S2D{1,1}(idxBulk,:)  = Dxx(idxBulk,:);
S2D{1,2}(idxBulk, :) = Dz(idxBulk, :);

% ------------------------------------------------------------------------
% 6.  Equation 3:  bottom impermeability  
% ------------------------------------------------------------------------
S2D{1,2}(idxBottom,:) = I_NM(idxBottom,:);

% ------------------------------------------------------------------------
% 7.  Equation 4: radiative BC 
% ------------------------------------------------------------------------
S2D{1,2}(idxLeftEdge,:)  = 0;
S2D{1,2}(idxRightEdge,:) = 0;
if startsWith(BCtype,'n','IgnoreCase',true)
    S2D{1,1}(idxLeftEdge,:)  = Dx(idxLeftEdge, :);
    S2D{1,1}(idxRightEdge,:) = Dx(idxRightEdge, :);
elseif startsWith(BCtype,'r','IgnoreCase',true)
    S2D{1,1}(idxLeftEdge,:)  = -1.0i * args.k * args.L_raft*I_NM(idxLeftEdge,:)  + Dx(idxLeftEdge, :);
    S2D{1,1}(idxRightEdge,:) =  1.0i * args.k * args.L_raft*I_NM(idxRightEdge,:) + Dx(idxRightEdge, :);
end
% ------------------------------------------------------------------------
% 8.  Derivative constraints
% ------------------------------------------------------------------------
% row-2:  phi_z  - Dz phi   = 0
S2D{2,1} = Dz;      S2D{2,2} = -I_NM;


% ------------------------------------------------------------------------
% 9.  Assemble RHS
% ------------------------------------------------------------------------
b = zeros(2*NM, 1);
if args.test == true
    b(idxContact) = -0.01; 
else
    b(idxContact) = - motor_weights(:).';   % only on raft contact nodes
end

% ------------------------------------------------------------------------
% 10.  Optional boundary trimming  (Dirichlet vs Neumann)
% ------------------------------------------------------------------------
if startsWith(BCtype,'d','IgnoreCase',true)         % Dirichlet in x
    keep = ~edgeMask(:);
    S2D  = cellfun(@(A) A(keep,keep), S2D,'UniformOutput',false);
    b    = b(:,keep);
elseif startsWith(BCtype,'n','IgnoreCase',true)     % Neumann lumping
    % already incorporated in Dx via one-sided stencils; no action
end

A = cell2mat(S2D);   % converts the 5�5 cell of NM�NM sparse blocks into one (5*NM)�(5*NM) sparse matrix
b_sparse = reshape(b.', [], 1);
xsol = full(A \ b_sparse);

end


