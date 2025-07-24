function x = solve_tensor_system(A, b, tol, maxiter, restart)
%SOLVE_TENSOR_SYSTEM  Krylov + fallback solver for tensor linear systems
%
%   x = solve_tensor_system(A, b [, tol, maxiter, restart])
%
% * 1) Flattens tensor A so that size(b) are the leading dims.
% * 2) Converts to sparse.
% * 3) Square   : GMRES?BiCGSTAB?(final) A\b
%   Rectangular : least-squares (lsqminnorm) ? (final) A\b
%
% Returned x is reshaped back to the tensor trailing dims of A.

% ---------------- defaults ------------------------------------------------
if nargin < 3 || isempty(tol),      tol      = 1e-6;  end
if nargin < 4 || isempty(maxiter),  maxiter  = 400;   end
if nargin < 5 || isempty(restart),  restart  = 50;    end

% ---------------- flatten tensors -----------------------------------------
A_dims = size(A);
b_dims = size(b);
m      = numel(b);
n      = numel(A) / m;

A_flat = sparse( reshape(A, m, n) );
b_vec  = reshape(b, m, 1);

% ---------------- choose pathway -----------------------------------------
if m == n                               % ---------- square system ----------
    
    x_flat = A_flat \ b_vec;          % direct solve
%     
%     % preconditioner
%     try
%         ilu_opts.type    = 'ilutp';
%         ilu_opts.droptol = 1e-3;
%         [L,U] = ilu(A_flat, ilu_opts);                 %   1st try
%     catch ME1
%         warning('solve_tensor_system:ILU',...
%                 'ILUTP failed (%s). Retrying with ''udiag'' option.', ME1.message);
% 
%         try
%             ilu_opts.udiag = 'on';                     % force unit diagonals
%             [L,U] = ilu(A_flat, ilu_opts);             %   2nd try
%         catch ME2
%             warning('solve_tensor_system:ILU',...
%                     'ILU with udiag failed (%s). Using NO preconditioner.', ME2.message);
%             L = speye(m);                              % identity ? no precond.
%             U = speye(m);
%         end
%     end
%     % --- GMRES
%     [x_flat, flag] = gmres(A_flat, b_vec, restart, tol, maxiter, L, U);
% 
%     % --- BiCGSTAB fallback (if GMRES failed)
%     if flag ~= 0
%         warning('solve_tensor_system:GMRES', ...
%                 'GMRES did not converge (flag=%d). Trying BiCGSTAB.', flag);
% 
%         [x_flat, flag] = bicgstab(A_flat, b_vec, tol, maxiter, L, U);
%     end
% 
%     % --- Direct sparse LU/QR fallback
%     if flag ~= 0
%         warning('solve_tensor_system:BiCGSTAB', ...
%                 'BiCGSTAB did not converge (flag=%d). Using A\\b.', flag);
%         x_flat = A_flat \ b_vec;          % direct solve
%     end

else                                    % ---------- rectangular system -----
    warning('solve_tensor_system:Rectangular', ...
            'A is rectangular (%d×%d). Using least-squares solver.', m, n);

    x_flat = lsqminnorm(A_flat, b_vec);

    % If for any reason lsqminnorm failed to improve (rare), fall back:
    if any(isnan(x_flat)) || any(isinf(x_flat))
        warning('solve_tensor_system:LSfail', ...
                'lsqminnorm failed; using A\\b least-squares.');
        x_flat = A_flat \ b_vec;         % QR-based sparse least-squares
    end
end

% ---------------- reshape back to tensor ----------------------------------
x_shape = A_dims(numel(b_dims)+1:end);
x = reshape(x_flat, x_shape);
end
