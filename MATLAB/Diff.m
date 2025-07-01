function D = Diff(dx, orderx, dz, orderz, shape)
% Returns a (N, M, N, M) finite difference operator approximating
% d^{orderx + orderz} / (dx^{orderx} dz^{orderz}) with 2nd-order accuracy.

    N = shape(1);
    M = shape(2);

    % Build 1D finite difference matrices
    Dx = build_1D_fd_matrix(N, dx, orderx);  % size (N, N)
    Dz = build_1D_fd_matrix(M, dz, orderz);  % size (M, M)

    % Vectorized outer product via broadcasting
    D = reshape(Dx, [N, 1, N, 1]) .* reshape(Dz, [1, M, 1, M]);  % (N, M, N, M)
end


function D1D = build_1D_fd_matrix(N, h, order)
% Returns an (N x N) matrix of finite difference coefficients for given order

    D1D = zeros(N, N);

    for i = 1:N
        stencil = get_fd_stencil(i, N, h, order);
        idx = stencil.indices;
        coeffs = stencil.coefficients;
        D1D(i, idx) = coeffs;
    end
end

function stencil = get_fd_stencil(i, N, h, order)
% Returns indices and coefficients for a second-order accurate FD stencil
% at index i in a grid of size N, for a derivative of given order (0, 1, or 2)

    % Handle 0th derivative (identity)
    if order == 0
        stencil.indices = i;
        stencil.coefficients = 1;
        return;
    end

    % Offsets and weights
    switch order
        case 1  % First derivative (second-order accurate)
            if i == 1
                % Forward 2nd-order
                offsets = [0, 1, 2];
                weights = [-3/2, 2, -1/2];
            elseif i == N
                % Backward 2nd-order
                offsets = [-2, -1, 0];
                weights = [1/2, -2, 3/2];
            else
                % Central difference
                offsets = [-1, 1];
                weights = [-1/2, 1/2];
            end
            weights = weights / h;

        case 2  % Second derivative (second-order accurate)
            if i == 1
                offsets = [0, 1, 2];
                weights = [2, -5, 4, -1]; % From Taylor expansion
                offsets = [0, 1, 2];
                weights = [2, -5, 4, -1] / h^2;
            elseif i == N
                offsets = [-2, -1, 0];
                weights = [-1, 4, -5, 2] / h^2;
            else
                offsets = [-1, 0, 1];
                weights = [1, -2, 1] / h^2;

        otherwise
            error('Only derivative orders 0, 1, 2 are supported.');
    end

    % Compute indices safely
    idx = i + offsets;
    valid = idx >= 1 & idx <= N;
    stencil.indices = idx(valid);
    stencil.coefficients = weights(valid);
end
