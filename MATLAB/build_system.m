function [S4D, b] = build_system(N, M, dx, dz, C, x_free, x_contact, ...
                                 motor_weights, bcType)
%BUILD_SYSTEM  Assemble the 5×5 sparse block system and RHS.
%
%   Inputs
%   ------
%   N, M           : grid points in x and z
%   dx, dz         : grid spacings
%   C              : struct with all dimensional-less constants
%   x_free         : logical N×1, true on free-surface nodes outside raft
%   x_contact      : logical N×1, true on raft footprint
%   motor_weights  : Gaussian load weights (length(x_contact) vector)
%   bcType         : 'dirichlet' | 'neumann'  (only decides edge trimming)
%
%   Outputs
%   -------
%   S4D            : 5×5 cell; each cell is a NM-by-NM sparse matrix
%   b              : 5×NM sparse RHS (one row per equation)

% -------------------------------------------------------------------------
% 1. helper maps and primitives
% -------------------------------------------------------------------------
NM   = N*M;                       % total grid nodes
lin  = @(i,j) i + (j-1)*N;        % (i,j) ? linear index  (column-major!)

% sparse 2-D identity
I_NM = speye(NM);

% -------------------------------------------------------------------------
% 2. 1-D derivative matrices           <-- now using File-Exchange toolbox
% -------------------------------------------------------------------------
ooa = 2;                % order-of-accuracy you want (2,4,6,?)
% First derivatives
Dx1 = getNonCompactFDmatrix(N, dx, 1, ooa);     % d/dx
Dz1 = getNonCompactFDmatrix(M, dz, 1, ooa);     % d/dz
% Second derivatives (needed for Laplacian or viscous terms)
Dx2 = getNonCompactFDmatrix(N, dx, 2, ooa);     % d²/dx²
Dz2 = getNonCompactFDmatrix(M, dz, 2, ooa);     % d²/dz²

% Boundary rows (first & last) come back already one-sided and therefore
% *consistent* with the interior accuracy, so we no longer need the manual
% zeroing I did with `Dx1([1 end],:) = 0`, etc.  Remove those lines.

% 2-D versions via Kronecker products (unchanged)
D_dx = kron(speye(M), Dx1);             % NM × NM, acts along x
D_dz = kron(Dz1     , speye(N));        % NM × NM, acts along z
Lapl = kron(speye(M), Dx2) + ...
       kron(Dz2     , speye(N));        % ?² = ?²/?x² + ?²/?z²


% -------------------------------------------------------------------------
% 3. row masks for the four equation groups
% -------------------------------------------------------------------------
topMask       = false(N,M); topMask(2:N-1,1)      = true;          % z = 0
bottomMask    = false(N,M); bottomMask(2:N-1,end) = true;          % z = H
leftRightMask = false(N,M); leftRightMask([1 end],:) = true;       % x = ±
bulkMask      = false(N,M); bulkMask(2:N-1,2:M-1) = true;          % interior

surfRows   = find(topMask    & repmat(x_free ,1,1)); % only free nodes
bulkRows   = find(bulkMask);
botRows    = find(bottomMask);
edgeRows   = find(leftRightMask);

% -------------------------------------------------------------------------
% 4. initialise 5×5 sparse block container
% -------------------------------------------------------------------------
S4D = repmat({sparse(NM,NM)}, 5, 5);

% -------------------------------------------------------------------------
% 5. E1 ? Bernoulli on free surface  (rows = surfRows)
% -------------------------------------------------------------------------
R = surfRows;           % shorthand

% unknown #1 :  ?
S4D{1,1}(R,:) =  C.C11*D_dz(R,:) + C.C13*I_NM(R,:);

% unknown #3 :  ?_z
S4D{1,3}(R,:) =  C.C12*D_dz(R,:) + C.C14*I_NM(R,:);

% -------------------------------------------------------------------------
% 6. E2 ? Laplace in the bulk  (rows = bulkRows, unknown #1 : ?)
% -------------------------------------------------------------------------
S4D{2,1}(bulkRows,:) = Lapl(bulkRows,:);

% -------------------------------------------------------------------------
% 7. E3 ? bottom BC  (rows = botRows, unknown #1 : ?)
% -------------------------------------------------------------------------
S4D{3,1}(botRows,:) = D_dz(botRows,:);  % ??/?z = 0

% -------------------------------------------------------------------------
% 8. E4 ? radiative BC left/right  (rows = edgeRows)
%         ?C32 ?  +  C31 ?_x  = 0
% -------------------------------------------------------------------------
S4D{4,1}(edgeRows,:) = -C.C32*I_NM(edgeRows,:);
S4D{4,2}(edgeRows,:) =  C.C31*I_NM(edgeRows,:);

% -------------------------------------------------------------------------
% 9. build RHS   (5×NM, row #1 only)
% -------------------------------------------------------------------------
b = sparse(5, NM);
surfContactRows = surfRows( ismember(surfRows, find(repmat(x_contact,1,1))) );
b(1, surfContactRows) = -C.C23 * motor_weights(:);

% -------------------------------------------------------------------------
% 10. optional boundary trimming (kept minimal here)
% -------------------------------------------------------------------------
if startsWith(bcType,'d','IgnoreCase',true)
    % simply drop the first/last x rows (Dirichlet handled by padding later)
    keep     = ~leftRightMask(:);
    S4D = cellfun(@(A) A(keep,keep), S4D, 'UniformOutput',false);
    b   = b(:, keep);
elseif startsWith(bcType,'n','IgnoreCase',true)
    % Neumann: lump ghost rows into their neighbours (one-sided diff done in Dx)
    % => no action needed here
end
end
