function [xsol, A, b] = build_system_v2(N, M, dx, dz, C, x_free, ...
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
%   N             ? [int] number of grid points in x-direction
%   M             ? [int] number of grid points in z-direction
%   dx            ? [m] horizontal grid spacing
%   dz            ? [m] vertical grid spacing
%   C             ? [NM × 1] flattened solution vector from previous timestep or iteration
%   x_free        ? [logical N×1] mask for free-surface regions (true where surface is free)
%   x_contact     ? [logical N×1] mask for raft-contact regions (true where surface contacts raft)
%   motor_weights ? [N×1] vector of distributed motor force weights over raft region
%   args          ? [struct] parameters structure (see flexible_surferbot) including:
%                     .sigma        ? [N/m] surface tension
%                     .rho          ? [kg/m^3] fluid density
%                     .omega        ? [rad/s] actuation frequency
%                     .EI           ? [N·m^2] bending stiffness of raft
%                     .rho_raft     ? [kg/m] linear density of raft
%                     .motor_inertia? [kg·m^2] motor rotational inertia
%                     .BC           ? boundary condition type ('radiative', etc.)
%
% -------------------------
% Outputs:
%   xsol ? [5NM × 1] solution vector ordering:
%              [phi; eta; eta_x; u_beam; u_beam_xx]
%   A    ? [5NM × 5NM] sparse matrix, left-hand side of linear system
%   b    ? [5NM × 1] right-hand side vector
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
I_NM = speye(NM);

% ------------------------------------------------------------------------
% 1.  Finite-difference operators (1-D ? 2-D via Kronecker) 
% ------------------------------------------------------------------------
[Dx,  Dz] = getNonCompactFDmatrix2D(N,M,dx,dz,1,args.ooa); % ?/?x, ?/?z
[Dxx, ~]  = getNonCompactFDmatrix2D(N,M,dx,dz,2,args.ooa); % ?2/?x2


% ------------------------------------------------------------------------
% 2.  Logical masks  ?  row-index vectors
% ------------------------------------------------------------------------
topMask       = false(M,N); topMask(1, 2:N-1)         = true;
freeMask      = repmat(x_free, M, 1)  & topMask;
contactMask   = repmat(x_contact,M,1) & topMask;

bottomMask    = false(M,N); bottomMask(end, 2:N-1)    = true;
rightEdgeMask = false(M,N); rightEdgeMask(:, end)     = true;
leftEdgeMask  = false(M,N); leftEdgeMask(:, 1)        = true;
bulkMask      = false(M,N); bulkMask(2:M-1, 2:N-1)    = true;

idxFreeSurf     = find(freeMask);
idxLeftFreeSurf = [1; idxFreeSurf(1:(end/2)); find(contactMask, 1, 'first')]; %idxLeftFreeSurf = idxLeftFreeSurf(1:(end-1));
nbLeft          = nnz(idxLeftFreeSurf);
idxRightFreeSurf= [find(contactMask, 1, 'last'); idxFreeSurf(((end/2)+1):end); (idxFreeSurf(end)+M)]; %idxRightFreeSurf = idxRightFreeSurf(2:(end));
idxContact      = find(contactMask);  

idxBulk       = find(bulkMask);
idxBottom     = find(bottomMask);
idxLeftEdge   = find(leftEdgeMask);
idxRightEdge  = find(rightEdgeMask);


% ------------------------------------------------------------------------
% 3.  Initialise 5×5 sparse cell container
% ------------------------------------------------------------------------
S2D = repmat({sparse(NM,NM)}, 2, 2);

% ------------------------------------------------------------------------
% 4.  E1 = Bernoulli on free surface  
% ------------------------------------------------------------------------
L = idxLeftFreeSurf; 
[DxxFree, ~] = getNonCompactFDmatrix(nbLeft,dx,2,args.ooa);
S2D{1,1}(L(2:end-1),L) =  C.C13 * I_NM(L(2:end-1), L) + C.C14*DxxFree(2:end-1, :);
S2D{1,2}(L(2:end-1),L) =  C.C11 * I_NM(L(2:end-1), L) + C.C12*DxxFree(2:end-1, :);

R = idxRightFreeSurf;
S2D{1,1}(R(2:end-1),R) =  C.C13 * I_NM(R(2:end-1), R) + C.C14*DxxFree(2:end-1, :);
S2D{1,2}(R(2:end-1),R) =  C.C11 * I_NM(R(2:end-1), R) + C.C12*DxxFree(2:end-1, :);

% ------------------------------------------------------------------------
% 4b.  Euler-beam equations inside raft  (rows = rowRaft)
%       overwrite the rows that were zeroed out above
% ------------------------------------------------------------------------
CC = idxContact;
[DxxRaft, ~] = getNonCompactFDmatrix(sum(x_contact),dx,2,args.ooa);
[Dx4Raft, ~] = getNonCompactFDmatrix(sum(x_contact),dx,4,args.ooa);
S2D{1, 1}(idxContact(1), :) = 0; S2D{1, 1}(idxContact(end), :) = 0;
S2D{1, 2}(idxContact(1), :) = 0; S2D{1, 2}(idxContact(end), :) = 0;
S2D{1, 1}(CC, CC) = C.C26 * DxxRaft + C.C24 * I_NM(CC, CC);
S2D{1, 2}(CC, CC) = (C.C22 + C.C25) * I_NM(CC, CC) + C.C21 * Dx4Raft;

% ------------------------------------------------------------------------
% 4c.  Surface-tension corner corrections ( +C27/dx ?²?/?x?z )
% ------------------------------------------------------------------------
%   right raft boundary, x > 0
rightBdry = find(contactMask, 1, 'last');      % right edge of raft

% right raft corner contribution
S2D{1,2}(rightBdry, idxRightFreeSurf) = ...
      S2D{1,2}(rightBdry, idxRightFreeSurf) ...
    + (C.C27/dx) * (I_NM(rightBdry+M, idxRightFreeSurf) - I_NM(rightBdry, idxRightFreeSurf))/dx;

% ------------------------------------------------------------------------
% 4c. Surface-tension corner correction ? LEFT raft boundary, x < 0
% ------------------------------------------------------------------------

% mask of free surface nodes (or their immediate left neighbour) that lie
% strictly to the *left* of the raft
leftBdry  = find(contactMask, 1, 'first');     % left  edge of raft
S2D{1,2}(leftBdry, idxLeftFreeSurf) = ...
    S2D{1,2}(leftBdry, idxLeftFreeSurf) ...
  + (C.C27/dx) * (I_NM(leftBdry, idxLeftFreeSurf) - I_NM(leftBdry-M, idxLeftFreeSurf))/dx;

if args.test == true % Dirichlet BC conditions for testing
    S2D{1, 1}(idxContact, :) = 0;
    S2D{1, 2}(idxContact, :) = I_NM(idxContact, :);
    %S4D{1, 2}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 3}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 4}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
    %S4D{1, 5}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
end
% ------------------------------------------------------------------------
% 5.  E2  ? Laplace in the bulk  (rows = rowBulk, unknown 1 : ?)
% ------------------------------------------------------------------------
S2D{1,1}(idxBulk,:)  = Dxx(idxBulk,:);
S2D{1,2}(idxBulk, :) = Dz(idxBulk, :);

% ------------------------------------------------------------------------
% 6.  E3  ? bottom impermeability  (rows = rowBottom, unknown 1 : ?)
% ------------------------------------------------------------------------
S2D{1,2}(idxBottom,:) = I_NM(idxBottom,:);

% ------------------------------------------------------------------------
% 7.  E4  ? radiative BC (rows = rowEdge)
% ------------------------------------------------------------------------
S2D{1,2}(idxLeftEdge,:)  = 0;
S2D{1,2}(idxRightEdge,:) = 0;
if startsWith(BCtype,'n','IgnoreCase',true)
    S2D{1,1}(idxLeftEdge,:)  = Dx(idxLeftEdge, :);
    S2D{1,1}(idxRightEdge,:) = Dx(idxRightEdge, :);
elseif startsWith(BCtype,'r','IgnoreCase',true)
    S2D{1,1}(idxLeftEdge,:)  = -C.C32*I_NM(idxLeftEdge,:)  + C.C31 * Dx(idxLeftEdge, :);
    S2D{1,1}(idxRightEdge,:) = +C.C32*I_NM(idxRightEdge,:) + C.C31 * Dx(idxRightEdge, :);
end
% ------------------------------------------------------------------------
% 8.  Derivative constraints
% ------------------------------------------------------------------------
% derivative-definition rows  (?x, ?xx, ?xxx, ?xxxx)
% row-2:  ?z   ? Dz ?     = 0
S2D{2,1} = Dz;      S2D{2,2} = -I_NM;


% ------------------------------------------------------------------------
% 9.  Assemble RHS
% ------------------------------------------------------------------------
b = zeros(2*NM, 1);
if args.test == true
    b(idxContact) = -0.001; 
else
    b(idxContact) = -C.C23 * motor_weights(:).';   % only on raft contact nodes
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

A = cell2mat(S2D);   % converts the 5×5 cell of NM×NM sparse blocks into one (5*NM)×(5*NM) sparse matrix
b_sparse = reshape(b.', [], 1);
xsol = full(A \ b_sparse);

end


