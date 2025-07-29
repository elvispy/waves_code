function [xsol, A, b] = build_system(N, M, dx, dz, C, x_free, ...
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
%lin  = @(i,j) reshape(i + (j-1)*N, [], 1);      % grid-to-linear
I_NM = speye(NM);

% ------------------------------------------------------------------------
% 1.  Finite-difference operators (1-D ? 2-D via Kronecker) 
% ------------------------------------------------------------------------
[Dx, Dz]   = getNonCompactFDmatrix2D(N,M,dx,dz,1,args.ooa); % ?/?x, ?/?z
[Dxx, Dzz] = getNonCompactFDmatrix2D(N,M,dx,dz,2,args.ooa);
Lapl = Dxx + Dzz;

% ------------------------------------------------------------------------
% 2.  Logical masks  ?  row-index vectors
% ------------------------------------------------------------------------
topMask       = false(M,N); topMask(1, 2:N-1)         = true;
freeMask      = repmat(x_free, M, 1)  & topMask;
contactMask   = repmat(x_contact,M,1) & topMask;
bottomMask    = false(M,N); bottomMask(end, 2:N-1)    = true;
edgeMask      = false(M,N); edgeMask(:, [1 end])      = true;
bulkMask      = false(M,N); bulkMask(2:M-1, 2:N-1)    = true;

idxFreeSurf   = find(freeMask);
idxContact    = find(contactMask);
idxBulk       = find(bulkMask);
idxBottom     = find(bottomMask);
idxEdge       = find(edgeMask);


% convenience: M×N grid of z-derivatives restricted to raft rows
DzRaft = Dz(idxContact ,:);

% ------------------------------------------------------------------------
% 3.  Initialise 5×5 sparse cell container
% ------------------------------------------------------------------------
S4D = repmat({sparse(NM,NM)}, 5, 5);

% ------------------------------------------------------------------------
% 4.  E1  ? Bernoulli on free surface   (rows = rowSurf) 
% ------------------------------------------------------------------------
R = idxFreeSurf;                       % for brevity

S4D{1,1}(R,:) =  C.C11*Dz(R,:) + C.C13*I_NM(R,:);
S4D{1,3}(R,:) =  C.C12*Dz(R,:) + C.C14*I_NM(R,:);

% ------------------------------------------------------------------------
% 4b.  Euler-beam equations inside raft  (rows = rowRaft)
%       overwrite the rows that were zeroed out above
% ------------------------------------------------------------------------
RR = idxContact;

S4D{1,1}(RR,:) = C.C22*DzRaft + C.C24*I_NM(RR,:) + C.C25*DzRaft;
S4D{1,3}(RR,:) = C.C26*I_NM(RR,:);
S4D{1,5}(RR,:) = C.C21*DzRaft;

% ------------------------------------------------------------------------
% 4c.  Surface-tension corner corrections ( +C27/dx ?²?/?x?z )
% ------------------------------------------------------------------------
%   right raft boundary, x > 0
rightBdry = find(contactMask, 1, 'last');      % right edge of raft


% Free nodes on the surface whose own column OR the next one to the right
% is free, and that lie strictly to the right of the raft:
idxRightFreeSurf = find(topMask &  repmat((x_free & (1:N > N/2)),M,1));


% right raft corner contribution
S4D{1,1}(rightBdry, idxRightFreeSurf) = ...
      S4D{1,1}(rightBdry, idxRightFreeSurf) ...
    + (C.C27/dx) * (Dz(rightBdry+M, idxRightFreeSurf) - Dz(rightBdry, idxRightFreeSurf))/dx;

% ------------------------------------------------------------------------
% 4c. Surface-tension corner correction ? LEFT raft boundary, x < 0
% ------------------------------------------------------------------------

% mask of free surface nodes (or their immediate left neighbour) that lie
% strictly to the *left* of the raft
leftBdry  = find(contactMask, 1, 'first');     % left  edge of raft
rowLeftFreeSurf = find(topMask &  repmat((x_free & (1:N < N/2)),M,1));

S4D{1,1}(leftBdry, rowLeftFreeSurf) = ...
    S4D{1,1}(leftBdry, rowLeftFreeSurf) ...
  + (C.C27/dx) * (Dz(leftBdry, rowLeftFreeSurf) - Dz(leftBdry-M, rowLeftFreeSurf))/dx;

%S4D{1, 1}(rowRaft, :) = Dz(rowRaft, :);
%S4D{1, 2}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
%S4D{1, 3}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
%S4D{1, 4}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
%S4D{1, 5}(rowRaft, :) = zeros(size(Dz(rowRaft, :)));
% ------------------------------------------------------------------------
% 5.  E2  ? Laplace in the bulk  (rows = rowBulk, unknown 1 : ?)
% ------------------------------------------------------------------------
S4D{1,1}(idxBulk,:)   = Lapl(idxBulk,:);
%S4D{1, 3}(rowBulk, :) = I_NM(rowBulk, :);

% ------------------------------------------------------------------------
% 6.  E3  ? bottom impermeability  (rows = rowBottom, unknown 1 : ?)
% ------------------------------------------------------------------------
S4D{1,1}(idxBottom,:) = Dz(idxBottom,:);

% ------------------------------------------------------------------------
% 7.  E4  ? radiative BC (rows = rowEdge)
% ------------------------------------------------------------------------
S4D{1,1}(idxEdge,:) = -C.C32*I_NM(idxEdge,:);
S4D{1,2}(idxEdge,:) =  C.C31*I_NM(idxEdge,:);

% ------------------------------------------------------------------------
% 8.  Derivative constraints
% ------------------------------------------------------------------------
% derivative-definition rows  (?x, ?xx, ?xxx, ?xxxx)
% row-2:  ?x   ? Dx ?     = 0
S4D{2,1} = Dx;      S4D{2,2} = -I_NM;
% row-3:  ?xx  ? Dx ?x    = 0
S4D{3,2} = Dx;      S4D{3,3} = -I_NM;
% row-4:  ?xxx ? Dx ?xx   = 0
S4D{4,3} = Dx;      S4D{4,4} = -I_NM;
% row-5:  ?xxxx ? Dx ?xxx = 0
S4D{5,4} = Dx;      S4D{5,5} = -I_NM;

% ------------------------------------------------------------------------
% 9.  Assemble RHS
% ------------------------------------------------------------------------
b = sparse(5, NM);
b(1, idxContact) = -C.C23 * motor_weights(:).';   % only on raft contact nodes

% ------------------------------------------------------------------------
% 10.  Optional boundary trimming  (Dirichlet vs Neumann)
% ------------------------------------------------------------------------
if startsWith(BCtype,'d','IgnoreCase',true)         % Dirichlet in x
    keep = ~edgeMask(:);
    S4D  = cellfun(@(A) A(keep,keep), S4D,'UniformOutput',false);
    b    = b(:,keep);
elseif startsWith(BCtype,'n','IgnoreCase',true)     % Neumann lumping
    % already incorporated in Dx via one-sided stencils; no action
end

A = cell2mat(S4D);   % converts the 5×5 cell of NM×NM sparse blocks into one (5*NM)×(5*NM) sparse matrix
b_sparse = reshape(b.', [], 1);
xsol = full(A \ b_sparse);

end
