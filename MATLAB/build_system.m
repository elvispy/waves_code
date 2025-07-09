function [A, b] = ...
          build_system(N, M, dx, dz, C, x_free, x_contact, ...
                           motor_weights, BCtype)
%BUILD_SYSTEM_V2   Assemble all five primary PDE blocks **plus**
%                  the Euler-beam equations inside the raft, the surface?
%                  tension corner terms, and the four horizontal-derivative
%                  definition blocks that you added in Python.
%
% Inputs  (same meaning as before, only two new ones):
%   rightBdry , leftBdry : scalar indices of the raft?s right/left limits
%
% Outputs (most users only need S4D and b)
%   S4D   ? 5×5 cell, each NM×NM sparse          (primary PDE + BC rows)
%   b     ? 5×NM sparse RHS                      (primary PDE + BC rows)
%   Dx    ? NM×NM sparse first-x-derivative      (returned because it is
%                                                 re-used when the caller
%                                                 stacks the four extra
%                                                 derivative-definition
%                                                 blocks as in Python)
%   rowSurf, rowBulk, ? ? handy index vectors you may want for post-process

% ------------------------------------------------------------------------
% 0.  House-keeping
% ------------------------------------------------------------------------
NM   = N*M;
lin  = @(i,j) reshape(i + (j-1)*N, [], 1);      % grid-to-linear
I_NM = speye(NM);

% ------------------------------------------------------------------------
% 1.  Finite-difference operators (1-D ? 2-D via Kronecker) 
% ------------------------------------------------------------------------
ooa  = 4;                                   % 2nd-order everywhere
Dx1  = getNonCompactFDmatrix(N, dx, 1, ooa);
Dz1  = getNonCompactFDmatrix(M, dz, 1, ooa);
Dx2  = getNonCompactFDmatrix(N, dx, 2, ooa);
Dz2  = getNonCompactFDmatrix(M, dz, 2, ooa);

Dx   = kron(speye(M), Dx1);                 % ?/?x
Dz   = kron(Dz1     , speye(N));            % ?/?z
Lapl = kron(speye(M), Dx2) + kron(Dz2, speye(N));

% ------------------------------------------------------------------------
% 2.  Logical masks  ?  row-index vectors
% ------------------------------------------------------------------------
topMask       = false(N,M); topMask(2:N-1,1)         = true;
freeMask      = false(N,M); freeMask(x_free,1)       = true;
contactMask   = false(N,M); contactMask(x_contact,1) = true;
bottomMask    = false(N,M); bottomMask(2:N-1,end)    = true;
edgeMask      = false(N,M); edgeMask([1 end],:)      = true;
bulkMask      = false(N,M); bulkMask(2:N-1,2:M-1)    = true;

rowSurf   = find(topMask   &  repmat(x_free' ,1,M));
rowBulk   = find(bulkMask);
rowBottom = find(bottomMask);
rowEdge   = find(edgeMask );

% raft rows (for the Euler-beam block E12)
rowRaft   = find(repmat(x_contact' ,1,M) & topMask);

% convenience: M×N grid of z-derivatives restricted to raft rows
DzRaft = Dz(rowRaft ,:);

% ------------------------------------------------------------------------
% 3.  Initialise 5×5 sparse cell container
% ------------------------------------------------------------------------
S4D = repmat({sparse(NM,NM)}, 5, 5);

% ------------------------------------------------------------------------
% 4.  E1  ? Bernoulli on free surface   (rows = rowSurf) 
% ------------------------------------------------------------------------
R = rowSurf;                       % for brevity

S4D{1,1}(R,:) =  C.C11*Dz(R,:) + C.C13*I_NM(R,:);
S4D{1,3}(R,:) =  C.C12*Dz(R,:) + C.C14*I_NM(R,:);

% ------------------------------------------------------------------------
% 4b.  Euler-beam equations inside raft  (rows = rowRaft)
%       overwrite the rows that were zeroed out above
% ------------------------------------------------------------------------
RR = rowRaft;

S4D{1,1}(RR,:) = C.C22*DzRaft + C.C24*I_NM(RR,:) + C.C25*DzRaft;
S4D{1,3}(RR,:) = C.C26*I_NM(RR,:);
S4D{1,5}(RR,:) = C.C21*DzRaft;

% ------------------------------------------------------------------------
% 4c.  Surface-tension corner corrections ( +C27/dx ?²?/?x?z )
% ------------------------------------------------------------------------
%   right raft boundary, x > 0
rightBdry = find(x_contact, 1, 'last');      % right edge of raft


% Free nodes on the surface whose own column OR the next one to the right
% is free, and that lie strictly to the right of the raft:
surfRightMask =  (x_free | circshift(x_free, -1))'  &  ((1:N).' > rightBdry);

% Linear-index mask for all (x,z) nodes whose *x* satisfies surfRightMask
rowMaskR = false(N*M,1);
rowMaskR(surfRightMask & true(1, M)) = true;

row_right = lin(rightBdry, 1);             % single node: (x = rightBdry, z = 1)

% build the mixed-derivative operator only once (still sparse!)
DxDz = Dx * Dz;               % ?²/?x?z   (same OO accuracy as Dx, Dz)

% right raft corner contribution
S4D{1,2}(row_right, rowMaskR) = ...
      S4D{1,2}(row_right, rowMaskR) ...
    + (C.C27/dx) * DxDz(row_right, rowMaskR);

% ------------------------------------------------------------------------
% 4c. Surface-tension corner correction ? LEFT raft boundary, x < 0
% ------------------------------------------------------------------------

% mask of free surface nodes (or their immediate left neighbour) that lie
% strictly to the *left* of the raft
leftBdry  = find(x_contact, 1, 'first');     % left  edge of raft
surfLeftMask =  (x_free | circshift(x_free, 1))'  &  ((1:N).' < leftBdry);

% logical row mask in the flattened NM-vector space
rowMaskL = false(N*M,1);
rowMaskL(surfLeftMask & true(1, M)) = true;     
row_left = lin(leftBdry, 1);              % single node: (x = leftBdry, z = 1)

S4D{1,1}(row_left, rowMaskL) = ...
    S4D{1,2}(row_left, rowMaskL) ...
  - (C.C27/dx) * DxDz(row_left, rowMaskL);   % note minus sign


% ------------------------------------------------------------------------
% 5.  E2  ? Laplace in the bulk  (rows = rowBulk, unknown 1 : ?)
% ------------------------------------------------------------------------
S4D{1,1}(rowBulk,:) = Lapl(rowBulk,:);

% ------------------------------------------------------------------------
% 6.  E3  ? bottom impermeability  (rows = rowBottom, unknown 1 : ?)
% ------------------------------------------------------------------------
S4D{1,1}(rowBottom,:) = Dz(rowBottom,:);

% ------------------------------------------------------------------------
% 7.  E4  ? radiative BC (rows = rowEdge)
% ------------------------------------------------------------------------
S4D{1,1}(rowEdge,:) = -C.C32*I_NM(rowEdge,:);
S4D{1,2}(rowEdge,:) =  C.C31*I_NM(rowEdge,:);

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
b(1, rowRaft) = -C.C23 * motor_weights(:).';   % only on raft contact nodes

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

end
