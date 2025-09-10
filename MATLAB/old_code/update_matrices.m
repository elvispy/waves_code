%% update_matrices.m - Apply Boundary Conditions to the Finite Difference System
% This function updates the linear system by enforcing boundary conditions on the wave
% propagation problem. It modifies the system matrix and right-hand side vector.
%
% INPUTS:
%   LaplacianMatrix - Initial finite difference Laplacian matrix
%   xGridPoints     - Horizontal grid points
%   zGridPoints     - Vertical grid points
%   gridSize        - Number of grid points in each direction
%   boundaryIndices - Indices where boundary conditions are applied
%   baseMatrix      - Initial operator matrix (unmodified Laplacian)
%   raftAngle       - Rotation angle of the raft
%   raftDisplacement - Vertical displacement of the raft
%   verticalStep    - Grid spacing in the vertical (z) direction
%
% OUTPUTS:
%   updatedMatrix - Updated finite difference matrix with boundary conditions applied
%   rhsVector     - Right-hand side vector incorporating boundary conditions

function [updatedMatrix, rhsVector] = update_matrices(LaplacianMatrix, xGridPoints, zGridPoints, gridSize, boundaryIndices, baseMatrix, raftAngle, raftDisplacement, verticalStep)

% Initialize sparse right-hand side vector
rhsVector = sparse(gridSize^2, 1);

% Define boundary condition terms
leftBoundaryCondition = 0 * zGridPoints;  % Left boundary condition (Equation 2.18c from paper)
bottomBoundaryCondition = 0 * xGridPoints;  % Bottom boundary condition (Equation 2.18e from paper)
rightBoundaryCondition = 0 * zGridPoints;  % Right boundary condition (Equation 2.18d from paper)
topBoundaryCondition = -1i * verticalStep * (raftDisplacement + xGridPoints * raftAngle); % Top boundary condition (Equation 2.18b from paper, wave elevation adjustment)
topBoundaryCondition(abs(xGridPoints) > 1/2) = 0; % Apply raft boundary conditions

% Apply top boundary condition (Equation 2.18b from paper)
boundaryIndicesTop = gridSize:gridSize:gridSize^2;
rhsVector(boundaryIndicesTop) = topBoundaryCondition;
LaplacianMatrix(boundaryIndicesTop, :) = 0; % Zero out matrix rows for Dirichlet conditions

% Apply right boundary condition (Equation 2.18d from paper)
boundaryIndicesRight = gridSize^2 - gridSize + 1:gridSize^2;
rhsVector(boundaryIndicesRight) = rightBoundaryCondition;
LaplacianMatrix(boundaryIndicesRight, :) = 0;

% Apply bottom boundary condition (Equation 2.18e from paper)
boundaryIndicesBottom = 1:gridSize:gridSize^2;
rhsVector(boundaryIndicesBottom) = bottomBoundaryCondition;
LaplacianMatrix(boundaryIndicesBottom, :) = 0;

% Apply left boundary condition (Equation 2.18c from paper)
boundaryIndicesLeft = 1:gridSize;
rhsVector(boundaryIndicesLeft) = leftBoundaryCondition;
LaplacianMatrix(boundaryIndicesLeft, :) = 0;

% Restore the values at boundary indices from the initial matrix
LaplacianMatrix(boundaryIndices) = baseMatrix(boundaryIndices);
updatedMatrix = LaplacianMatrix;

end
