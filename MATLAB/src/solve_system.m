function xsol = solve_system(A, b_sparse)

%% ---------- Fast path: sparse backslash + certified gate ----------

% silence warnings during solve
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:singularMatrix'); lastwarn('');

%% ---------- Diagonal equilibration + column reordering, then solve ----------
n = size(A,1);

% Row/col scalings (?-equilibration), with guards to avoid extremes
row_max = full(max(abs(A),[],2));  row_max(~isfinite(row_max) | row_max==0) = 1;
col_max = full(max(abs(A),[],1)).'; col_max(~isfinite(col_max) | col_max==0) = 1;

dr = 1 ./ row_max;  dr = min(max(dr, 1e-8), 1e+8);               % clamp
dc = 1 ./ col_max;  dc = min(max(dc, 1e-8), 1e+8);

Dr = spdiags(dr,0,n,n);
Dc = spdiags(dc,0,n,n);

As  = Dr * A * Dc;                      % scaled operator
beq = Dr * b_sparse;                    % scaled RHS

% Column permutation for stability/fill; MATLAB?s LU also reorders internally,
% but we apply p so ILU/Krylov later can reuse it consistently.
try
    p = colamd(As);
catch
    p = 1:n;
end
Ap  = As(:,p);
bp  = beq;

lastwarn('');              % reset warning state


% Solve scaled-permuted system, then unpermute and unscale
yp      = Ap \ bp;                      % fast path solve
y       = zeros(n,1);  y(p) = yp;       % undo column permutation
xtry    = Dc * y;                       % map back to original variables

[msg,~] = lastwarn;       % capture last warning

if isempty(msg)
    xsol = full(xtry);
    return;
end

% ---------- Certification on ORIGINAL A,b (componentwise + ??·?_c) ----------
% componentwise backward error
r      = b_sparse - A*xtry;
denvec = abs(A)*abs(xtry) + abs(b_sparse) + eps;
eta_c  = full(max(abs(r)./denvec));


khat_ref = Inf;
try
    % reuse scaled system to estimate refinement gain cheaply
    rp   = Dr * r;                      % residual in scaled coordinates
    dyp  = Ap \ rp;                     % one refinement step in scaled space
    dy   = zeros(n,1); dy(p) = dyp;
    dx   = Dc * dy;                     % map back
    gain = (norm(dx,2)/max(norm(xtry,2),realmin)) / max(eta_c,realmin);
    khat_ref = min(max(gain,1),1e20);
catch
end

khat       = min(max([1, khat_ref]), 1e20);
fwd_bound  = khat * eta_c;
reliable_digits = max(0, -log10(max(fwd_bound, realmin))); %#ok<NASGU>

tol_eta      = 1e-12;
tol_fwdbound = 1e-8;

quality_ok = all(isfinite([eta_c,fwd_bound])) && all(isfinite(xtry)) && ...
             (eta_c <= tol_eta) && (fwd_bound <= tol_fwdbound);

fallback_needed = ~quality_ok;

ternary = @(varargin)varargin{end-varargin{1}};

if ~fallback_needed
    xsol = full(xtry);
else
    % ---------- Robust, scalable fallback: scaled+permuted operator + strong preconditioning ----------

    n   = size(A,1);

    % (1) diagonal equilibration (?-equilibration with clamps)
    row_max = full(max(abs(A),[],2));  row_max(~isfinite(row_max) | row_max==0) = 1;
    col_max = full(max(abs(A),[],1)).'; col_max(~isfinite(col_max) | col_max==0) = 1;
    dr = 1 ./ row_max;  dr = min(max(dr,1e-8),1e8);
    dc = 1 ./ col_max;  dc = min(max(dc,1e-8),1e8);

    Dr = spdiags(dr,0,n,n);  Dc = spdiags(dc,0,n,n);
    As = Dr * A * Dc;                       % scaled operator
    bs = Dr * b_sparse;                     % scaled RHS

    % (1b) column permutation for stability/fill; reuse everywhere below
    try
        p = colamd(As);
    catch
        p = 1:n;
    end
    Ap = As(:,p);                           % scaled + permuted operator
    bp = bs;                                % RHS unchanged (left scaling only)

    % Helpers to apply Ap and its adjoint, and to undo the column permutation
    Aop   = @(x) Ap * x;
    ATop  = @(y) Ap' * y;
    %unperm = @(yp) accumarray(p(:), yp, [n 1], [], 0);


    % (2) preconditioner on Ap: try ILUTP with modest fill, else damped Jacobi
    usedILU = false;
    Mfun = [];  % left preconditioner (approximate inverse)
    try
        ilu_opts = struct('type','ilutp','droptol',1e-2,'milu','off','udiag',false);
        [L,U] = ilu(Ap, ilu_opts);
        piv = abs(diag(U)); okILU = all(isfinite(piv)) & min(piv) > 1e-15;
        if okILU
            usedILU = true;
            Mfun = @(y) U \ (L \ y);       % left preconditioner apply
        end
    catch
        % fall through
    end
    if isempty(Mfun)
        D = abs(diag(Ap)); D(~isfinite(D) | D==0) = 1;
        Mfun = @(y) y ./ D;                % damped Jacobi
    end

    % (3) Krylov stage 1: GMRES(restart) with left preconditioning on Ap
    tol_relres = 1e-8;
    maxit      = 300;
    restart    = 40;

    [xp,flag_gm,rel_gm,iter_gm] = gmres(@(x) Aop(Mfun(x)), bp, restart, tol_relres, ceil(maxit/restart));
    gm_ok = (flag_gm==0) && isfinite(rel_gm);
    if gm_ok
        xp = Mfun(xp);                     % retrieve solution of Ap*xp = bp from preconditioned variable
    else
        % (3b) BiCGSTAB fallback if GMRES stalls
        [xp,flag_bi,rel_bi,iter_bi] = bicgstab(@(x) Aop(Mfun(x)), bp, tol_relres, maxit);
        bi_ok = (flag_bi==0) && isfinite(rel_bi);
        if bi_ok, xp = Mfun(xp); end
    end
    krylov_ok = exist('xp','var') && (gm_ok || (exist('bi_ok','var') && bi_ok));

    % (3c) short iterative refinement on the scaled system (improves digits if M is decent)
    if krylov_ok
        for k = 1:2
            rp = bp - Aop(xp);
            if norm(rp,inf) <= 1e-12*(norm(Ap,inf)*max(norm(xp,inf),1)+norm(bp,inf)), break; end
            dxp = gmres(@(x) Aop(Mfun(x)), rp, restart, min(1e-12,tol_relres/100), ceil(100/restart));
            dxp = Mfun(dxp);
            xp  = xp + dxp;
        end
    end

    % (4) Regularized normal equations with PCG if Krylov failed
    reg_ok = false;
    if ~krylov_ok
        lam = 1e-8 * norm(A,1);                   % light Tikhonov in original scale
        % In scaled-permuted coords this is Ap'*Ap + ? I
        Nop = @(z) ATop(Aop(z)) + lam*z;          % SPD operator
        [xp,flag_reg,rel_reg,it_reg] = pcg(Nop, ATop(bp), tol_relres, 500);
        reg_ok = (flag_reg==0) && isfinite(rel_reg);
    end

    % (5) LSQR least-squares as last resort
    ls_ok = false;
    if ~krylov_ok && ~reg_ok
        Af_op = @(flag,x) strcmp(flag,'notransp')*Aop(x) + strcmp(flag,'transp')*ATop(x);
        [xp,flag_ls,rel_ls,it_ls] = lsqr(Af_op, bp, tol_relres, 2000);
        ls_ok = (flag_ls==0) || (flag_ls==1 && rel_ls<1e-6);
    end

    % (6) map back to original variables and report
    if exist('xp','var')
        y   = zeros(n,1); y(p) = xp;       % undo column permutation
        xsc = y;                            % solution in scaled coords
        xsol = full(dc .* xsc);             % map back: x = Dc * y
    else
        error('SolverFailed:AllFallbacks','All fallbacks failed to produce xp.');
    end

    if krylov_ok
        warning('FallbackIterative:Used', ...
            'Fallback: %s. relres_gm=%s, it_gm=%s, ILU=%d.', ...
            ternary(gm_ok,'GMRES','BiCGSTAB'), num2str(exist('rel_gm','var')*rel_gm + (~exist('rel_gm','var'))*rel_bi, '%.2e'), ...
            mat2str(exist('iter_gm','var')*iter_gm + (~exist('iter_gm','var'))*iter_bi), usedILU);
    elseif reg_ok
        warning('FallbackRegularized:Used', ...
            'Regularized normal equations used. relres=%.2e, it=%s.', rel_reg, mat2str(it_reg));
    elseif ls_ok
        res = A*xsol - b_sparse;
        relres_final = norm(res,2)/max(norm(b_sparse,2),1);
        warning('FallbackLSQR:Used', ...
            'LSQR used. relres2=%.2e, it=%s.', relres_final, mat2str(it_ls));
    else
        error('SolverFailed:AllFallbacks','All fallbacks failed: GMRES/BiCGSTAB/regularization/LSQR.');
    end

end % end if

end