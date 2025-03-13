%% build_matrices.m - Constructs Finite Difference Matrices for Wave Propagation
% This function builds sparse matrices for the linear system used in solving
% the wave-driven propulsion problem. The matrices represent the finite difference
% discretization of the governing equations.
%
% INPUTS:
%   gridSize        - Number of grid points in each direction
%   raftStartIdx, raftEndIdx - Indices defining the location of the raft in the domain
%   oscillationFreq - Dimensionless oscillation frequency
%   waveNumber      - Computed wavenumber from the dispersion relation
%   verticalStep, horizontalStep - Grid spacings in the vertical (z) and horizontal (x) directions
%   viscosityCoeff  - Dimensionless viscosity
%   surfaceTension  - Dimensionless surface tension
%
% OUTPUTS:
%   systemMatrix    - Sparse matrix representing the discretized wave equation
%   boundaryMatrix  - Sparse matrix representing boundary conditions


function [systemMatrix,boundaryMatrix] = build_matrices(gridSize,raftStartIdx,raftEndIdx,oscillationFreq,waveNumber,verticalStep,horizontalStep,viscosityCoeff,surfaceTension)

% Initialize sparse matrices for finite difference discretization
systemMatrix=sparse(gridSize^2,gridSize^2);
boundaryMatrix=sparse(gridSize^2,gridSize^2);

% Define left and right boundary wavenumbers
waveNumberLeft=waveNumber;
waveNumberRight=-waveNumber;

% Apply boundary conditions and fill matrix entries
j=0;
for ii=gridSize:gridSize:gridSize^2
    j=j+1;
    id=find(systemMatrix(ii,:));
    systemMatrix(ii,id)=0;
    
    if  and(ii>=raftStartIdx*gridSize,ii<=raftEndIdx*gridSize)
        systemMatrix(ii,ii)=-3/2;
        systemMatrix(ii,ii-1)=2;
        systemMatrix(ii,ii-2)=-1/2;
        boundaryMatrix(ii,ii)=1;
        boundaryMatrix(ii,ii-1)=1;
        boundaryMatrix(ii,ii-2)=1;
    else  % Apply conditions at fluid domain edges
        if ii - gridSize <= 0  % Left boundary condition (Equation 2.18c from paper)
            systemMatrix(ii,ii)=(-3/2).*verticalStep.^(-1)+(-3).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*(-8)).*horizontalStep.^(-2).*viscosityCoeff+oscillationFreq.^2;
            systemMatrix(ii,ii-1)=2.*verticalStep.^(-1)+4.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-2)=(-1/2).*verticalStep.^(-1)+(-1).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+gridSize)=(-15/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*(-20)).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii+2*gridSize)=6.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*16).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii+3*gridSize)=(-3/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*(-4)).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii+gridSize-1)=10.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+2*gridSize-1)=(-8).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+3*gridSize-1)=2.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+gridSize-2)=(-5/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+2*gridSize-2)=2.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+3*gridSize-2)=(-1/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            boundaryMatrix(ii,ii)=1;
            boundaryMatrix(ii,ii-1)=1;
            boundaryMatrix(ii,ii-2)=1;
            boundaryMatrix(ii,ii+gridSize)=1;
            boundaryMatrix(ii,ii+2*gridSize)=1;
            boundaryMatrix(ii,ii+3*gridSize)=1;
            boundaryMatrix(ii,ii+gridSize-1)=1;
            boundaryMatrix(ii,ii+2*gridSize-1)=1;
            boundaryMatrix(ii,ii+3*gridSize-1)=1;
            boundaryMatrix(ii,ii+gridSize-2)=1;
            boundaryMatrix(ii,ii+2*gridSize-2)=1;
            boundaryMatrix(ii,ii+3*gridSize-2)=1;
        elseif ii + gridSize > gridSize^2  % Right boundary condition (Equation 2.18d from paper)
            systemMatrix(ii,ii)=(-3/2).*verticalStep.^(-1)+(-3).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*(-8)).*horizontalStep.^(-2).*viscosityCoeff+oscillationFreq.^2;
            systemMatrix(ii,ii-1)=2.*verticalStep.^(-1)+4.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-2)=(-1/2).*verticalStep.^(-1)+(-1).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-gridSize)=(-15/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*(-20)).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii-2*gridSize)=6.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*16).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii-3*gridSize)=(-3/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*(-4)).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii-gridSize-1)=10.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-2*gridSize-1)=(-8).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-3*gridSize-1)=2.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-gridSize-2)=(-5/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-2*gridSize-2)=2.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-3*gridSize-2)=(-1/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            boundaryMatrix(ii,ii)=1;
            boundaryMatrix(ii,ii-1)=1;
            boundaryMatrix(ii,ii-2)=1;
            boundaryMatrix(ii,ii-gridSize)=1;
            boundaryMatrix(ii,ii-2*gridSize)=1;
            boundaryMatrix(ii,ii-3*gridSize)=1;
            boundaryMatrix(ii,ii-gridSize-1)=1;
            boundaryMatrix(ii,ii-2*gridSize-1)=1;
            boundaryMatrix(ii,ii-3*gridSize-1)=1;
            boundaryMatrix(ii,ii-gridSize-2)=1;
            boundaryMatrix(ii,ii-2*gridSize-2)=1;
            boundaryMatrix(ii,ii-3*gridSize-2)=1;
        else
            systemMatrix(ii,ii)=(-3/2).*verticalStep.^(-1)+(-3).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*(-8)).*horizontalStep.^(-2).*viscosityCoeff+oscillationFreq.^2;
            systemMatrix(ii,ii-1)=2.*verticalStep.^(-1)+4.*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-2)=(-1/2).*verticalStep.^(-1)+(-1).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+gridSize)=(3/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*4).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii-gridSize)=(3/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension+(sqrt(-1)*4).*horizontalStep.^(-2).*viscosityCoeff;
            systemMatrix(ii,ii+gridSize-1)=(-2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-gridSize-1)=(-2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii+gridSize-2)=(1/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            systemMatrix(ii,ii-gridSize-2)=(1/2).*horizontalStep.^(-2).*verticalStep.^(-1).*surfaceTension;
            boundaryMatrix(ii,ii)=1;
            boundaryMatrix(ii,ii-1)=1;
            boundaryMatrix(ii,ii-2)=1;
            boundaryMatrix(ii,ii+gridSize)=1;
            boundaryMatrix(ii,ii-gridSize)=1;
            boundaryMatrix(ii,ii+gridSize-1)=1;
            boundaryMatrix(ii,ii-gridSize-1)=1;
            boundaryMatrix(ii,ii+gridSize-2)=1;
            boundaryMatrix(ii,ii-gridSize-2)=1;
        end
    end
    
end


% Apply Right Boundary Condition (Equation 2.18d from paper)

for ii=gridSize^2-gridSize+1:gridSize^2

    id=find(systemMatrix(ii,:));
    systemMatrix(ii,id)=0;
    
    systemMatrix(ii,ii)=-3/2-1i*waveNumberLeft*horizontalStep;
    systemMatrix(ii,ii-gridSize)=2;
    systemMatrix(ii,ii-2*gridSize)=-1/2;
    
    boundaryMatrix(ii,ii)=1;
    boundaryMatrix(ii,ii-gridSize)=1;
    boundaryMatrix(ii,ii-2*gridSize)=1;
end

% Apply Bottom Boundary Condition (Equation 2.18e from paper)
for ii=1:gridSize:gridSize^2
    id=find(systemMatrix(ii,:));
    systemMatrix(ii,id)=0;
    
    systemMatrix(ii,ii)=3/2;
    systemMatrix(ii,ii+1)=-2;
    systemMatrix(ii,ii+2)=1/2;
    
    boundaryMatrix(ii,ii)=1;
    boundaryMatrix(ii,ii+1)=1;
    boundaryMatrix(ii,ii+2)=1;
end

% Apply Left Boundary Condition (Equation 2.18c from paper)
for ii=1:gridSize
    id=find(systemMatrix(ii,:));
    systemMatrix(ii,id)=0;
    
    systemMatrix(ii,ii)=3/2-1i*waveNumberRight*horizontalStep;
    systemMatrix(ii,ii+gridSize)=-2;
    systemMatrix(ii,ii+2*gridSize)=1/2;
    
    boundaryMatrix(ii,ii)=1;
    boundaryMatrix(ii,ii+gridSize)=1;
    boundaryMatrix(ii,ii+2*gridSize)=1;
end

end