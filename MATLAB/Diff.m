function D = Diff(dx, orderx, dz, orderz, shape)
%DIFF   Finite-difference tensor for mixed derivatives on a uniform grid
%
%   D = Diff(dx, orderx, dz, orderz, [N M])
%
% returns a (N, M, N, M) array such that
%
%   (D(i,j,:,:)) * phi              ? ?^{orderx+orderz} ? / (?x^orderx ?z^orderz) at (i,j)
%
% All approximations are 2nd-order accurate in space, including boundaries.
%
% INPUTS
%   dx, dz     : grid spacings in x and z directions
%   orderx/z   : derivative order 0, 1 or 2
%   shape      : [N M] number of nodes in x and z
%
% OUTPUT
%   D          : (N,M,N,M) double  (use sparse Dx/Dz if memory is an issue)

% -------------------------------------------------------------------------
N = shape(1);
M = shape(2);

% 1-D differentiation matrices (N×N and M×M)
Dx = build_1D_fd_matrix(N, dx, orderx);
Dz = build_1D_fd_matrix(M, dz, orderz);

% Tensor (outer) product ? (N,M,N,M)
D = reshape(Dx, [N 1 N 1]) .* reshape(Dz, [1 M 1 M]);
end
% -------------------------------------------------------------------------
function A = build_1D_fd_matrix(N, h, order)
% Second-order accurate forward / centred / backward stencils

A = spalloc(N, N, 4*N);          % sparse; at most 4 non-zeros per row

for i = 1:N
    [offsets, coeffs] = one_sided_stencil(i, N, h, order);
    idx = i + offsets;           % global column indices
    A(i, idx) = coeffs;
end
A = full(A);                      % change to sparse(A) if desired
end
% -------------------------------------------------------------------------
function [offsets, w] = one_sided_stencil(i, N, h, order)
% Returns offsets (relative indices) and weights for a
% 2-nd-order accurate FD stencil at position i on length-N grid.

switch order
    % ------------------------------------------------------- 0th derivative
    case 0
        offsets = 0;             % ? itself
        w       = 1;
    % ------------------------------------------------------- 1st derivative
    case 1
        if i == 1                      % forward 2nd-order
            offsets = [0  1  2];
            w       = [-3  4 -1] / (2*h);
        elseif i == N                  % backward 2nd-order
            offsets = [-2 -1  0];
            w       = [ 1 -4  3] / (2*h);
        else                           % centred 2nd-order
            offsets = [-1 1];
            w       = [-1 1] / (2*h);
        end
    % ------------------------------------------------------- 2nd derivative
    case 2
        if i == 1                      % forward 4-point, 2nd-order
            offsets = [0 1 2 3];
            w       = [ 2 -5 4 -1] / h^2;
        elseif i == N                  % backward 4-point, 2nd-order
            offsets = [-3 -2 -1 0];
            w       = [-1 4 -5 2] / h^2;
        else                           % centred 3-point, 2nd-order
            offsets = [-1 0 1];
            w       = [1 -2 1] / h^2;
        end
    otherwise
        error('Derivative order must be 0, 1 or 2.');
end
end
