function x = solve_tensor_system(A, b, tol, maxiter)
    % Solve tensor system A · x = b
    if nargin < 3, tol = 1e-5; end
    if nargin < 4, maxiter = 1000; end

    A_dims = size(A);
    b_dims = size(b);

    m = numel(b);
    n = numel(A) / m;
    A_flat = reshape(A, [m, n]);
    b_vec = reshape(b, [m, 1]);

    if m == n
        x_flat = A_flat \ b_vec;
    else
        warning("Using least-squares solver for rectangular system.");
        x_flat = lsqminnorm(A_flat, b_vec);
    end

    x = reshape(x_flat, A_dims(b_dims + 1:end));
end
