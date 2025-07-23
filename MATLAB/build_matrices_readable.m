% build_matrices_readable.m — assemble FD matrices for wave‑driven raft problem
% -------------------------------------------------------------------------
% Constructs two sparse matrices on an N×N Cartesian grid
%   A * phi = 0   (system matrix)
%   B .* A        (Hadamard) marks which unknowns each BC row touches.
% -------------------------------------------------------------------------
% INPUT
%   N          : grid points in x **and** z (square grid)
%   raftIdx    : [cStart cEnd] columns covered by the raft (no‑flux zone)
%   omega, k   : non‑dimensional frequency and wavenumber
%   dz, dx     : grid steps in vertical (z) and horizontal (x)
%   nu, sigma  : non‑dimensional viscosity and surface tension
% OUTPUT
%   A, B       : sparse(N^2) system & mask matrices
% -------------------------------------------------------------------------
function [A,B] = build_matrices_readable(N, raftStart, raftEnd, omega, k, dz, dx, nu, sigma)

raftIdx = [raftStart raftEnd];
% ------- 1. Pre‑allocate (≈ 20 non‑zeros / row is generous) ------------
Nz   = N^2;
A    = spalloc(Nz, Nz, 20*N);  % system matrix
B    = sparse (Nz, Nz);        % boundary‑mask (0/1)

% Short‑hands -----------------------------------------------------------
invDz   = 1/dz;                 %  Δz⁻¹
invDx2  = 1/dx^2;               %  Δx⁻²
colTop  = @(c) c*N;             %  linear index of surface node (column c)

% Derivative weights ----------------------------------------------------
% 3‑pt one‑sided d/dz  :  [ -3/2  2  -1/2 ] / dz
% 4‑pt one‑sided d²/dx²:  [ -15/2  6  -3/2 ] / dx²  (used at lateral edges)
% 3‑pt centred  d²/dx²:  [ 1  -2  1 ] / dx²         (used in interior columns)

dz_back      = [-3/2,  2, -1/2] * invDz;          % size‑3 row
d2x_4oneside = [-15/2, 6, -3/2] * invDx2;        % 4th‑order one‑sided
d2x_2centre  = [1, -2, 1]        * invDx2;        % 2nd‑order centred

% Helper to overwrite a matrix row and update the mask ------------------
function setRow(r, cols, vals)
    A(r, :) = 0;           % clear previous content (cheap for sparse)
    A(r, cols) = vals;
    B(r, cols) = 1;        % mark participation of these unknowns
end

% ---------------- 2. FREE‑SURFACE (z = 0) rows -------------------------
for c = 1:N
    i = colTop(c);   % linear index of surface node in column c

    % 2.1  Under‑raft region: ∂phi/∂z = 0  (no‑flux)
    if raftIdx(1) <= c && c <= raftIdx(2)
        setRow(i, [i, i-1, i-2], dz_back);
        continue;            % next column
    end

    % Edge flags determine stencil type for d²/dx²
    isLeftEdge  = (c == 1);
    isRightEdge = (c == N);

    % Diagonal coefficient common to **all** surface nodes (∂z term + ω²)
    diagSurf = (-3/2)*invDz ...                  % from ∂z
             - 3 * sigma * invDx2 * invDz ...    % from sigma∂x²∂z
             - 1i*8 * nu * invDx2 ...            % from –i8ν∂x²
             + omega^2;                          % + ω²φ

    % Base d/dz stencil (size 3)
    cols = [i, i-1, i-2];
    vals = diagSurf + [0, 0, 0];
    vals(2:3) = vals(2:3) + [2*invDz, -0.5*invDz];

    % 2.2  Lateral edges — 4‑pt one‑sided d²/dx² ------------------------
    if isLeftEdge || isRightEdge
        sgn  = 1 - 2*isRightEdge;          % +1 left, −1 right
        offs = sgn * (1:3) * N;            % ±N, ±2N, ±3N offsets

        % –i8ν∂x²φ term (surface nodes only)
        cols = [cols, i+offs];
        vals = [vals, -1i*8*nu * d2x_4oneside];

        % sigma∂x²∂z term needs nodes 1 & 2 layers below surface in those cols
        below1 = i + offs - 1;
        below2 = i + offs - 2;
        coeffZ = sigma * invDz;
        cols   = [cols, below1, below2];
        vals   = [vals, coeffZ*d2x_4oneside, coeffZ*d2x_4oneside];

        setRow(i, cols, vals);
        continue;
    end

    % 2.3  Interior surface columns — centred 2‑nd‑order d²/dx² ---------
    left  = i - N;   right = i + N;
    cols  = [cols, left, right];
    vals  = [vals, -1i*8*nu * d2x_2centre([1 3])];   % –i8ν∂x²φ

    % sigma∂x²∂z term (needs nodes below surface in left and right cols)
    below = [left, right] - 1;           % one layer below
    below2= below - 1;                   % two layers below
    coeffZ = sigma * invDz;
    cols   = [cols, below, below2];
    vals   = [vals, coeffZ*d2x_2centre([1 3]), coeffZ*d2x_2centre([1 3])];

    setRow(i, cols, vals);
end

% ---------------- 3. SOMMERFELD RADIATION (vertical sides) -------------
makeSommerfeld = @(row, dirSign) ...
    setRow(row, row + dirSign*(0:2)*N, ...
           [3/2, -2, 1/2]*dirSign + [-1i * k * dx * (dirSign>0), 0, 0]);

% 3a. Left boundary (x = x_min)   → forward difference (dirSign = +1)
for row = Nz - N + 1 : Nz  % last column in linear indexing
    makeSommerfeld(row, +1);
end

% 3b. Right boundary (x = x_max)  → backward difference (dirSign = -1)
for row = 1 : N
    makeSommerfeld(row, -1);
end

% ---------------- 4. BOTTOM WALL  ∂phi/∂z = 0  ------------------------
for col = 1 : N
    row = 1 + (col - 1) * N;           % lowest node in column
    setRow(row, [row, row+1, row+2],  [3/2, -2, 1/2] * invDz);
end

end

