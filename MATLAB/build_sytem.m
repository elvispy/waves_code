function [A, b] = build_sytem(nx, nz, dx, dz, raftIndices, motorIndices)

    
    topIndices       = zeros(nx, nz); topIndices(2:(end-1), 1) = 1;
    leftRightIndices = zeros(nx, nz); leftRightIndices([1, end], :) = 1;
    bottomIndices    = zeros(nx, nz); bottomIndices(2:(end-1), end) = 1;
    bulkIndices      = zeros(nx, nz); bulkIndices(2:(end-1), 2:(end-1)) = 1;
    contactIndices   = zeros(nx, nz); contactIndices(:, 1) = raftIndices;
    
    
    S4D = cell(5, 5); % Phi and up to their fourth derivative
    
    


end