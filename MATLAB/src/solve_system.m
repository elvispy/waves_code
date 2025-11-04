function xsol = solve_system(A, b_sparse)

% ---------- 0) Try A\b on the original system ----------
old1 = warning('query','MATLAB:nearlySingularMatrix');
old2 = warning('query','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:singularMatrix');
lastwarn('');
x_try0 = A \ b_sparse;
[msg0,~] = lastwarn;
warning(old1.state,'MATLAB:nearlySingularMatrix');
warning(old2.state,'MATLAB:singularMatrix');

if isempty(msg0)
    xsol = full(x_try0);
    return
end

error('Normal solver couldnt find a solution. Trying Iterative-based methods instead');
% ---------- 1) Diagonal equilibration + column reordering ----------
n = size(A,1);
row_max = full(max(abs(A),[],2));  row_max(~isfinite(row_max) | row_max==0) = 1;
col_max = full(max(abs(A),[],1)).'; col_max(~isfinite(col_max) | col_max==0) = 1;
dr = 1 ./ row_max;  dr = min(max(dr,1e-8),1e8);
dc = 1 ./ col_max;  dc = min(max(dc,1e-8),1e8);

Dr = spdiags(dr,0,n,n);
Dc = spdiags(dc,0,n,n);

As  = Dr * A * Dc;
bs  = Dr * b_sparse;

try
    p = colamd(As);
catch
    p = (1:n).';
end
Ap = As(:,p);   bp = bs;

% ---------- 2) Fast scaled solve + certification ----------
lastwarn('');
yp    = Ap \ bp;
y     = zeros(n,1);  y(p) = yp;
xtry  = Dc * y;

[msg,~] = lastwarn;  % if this warns, we still certify below

% Componentwise backward error on ORIGINAL A,b
r      = b_sparse - A*xtry;
denvec = abs(A)*abs(xtry) + abs(b_sparse) + eps;
eta_c  = full(max(abs(r)./denvec));

% Estimate refinement gain cheaply on scaled system
khat_ref = Inf;
try
    rp   = Dr * r;
    dyp  = Ap \ rp;
    dy   = zeros(n,1); dy(p) = dyp;
    dx   = Dc * dy;
    gain = (norm(dx,2)/max(norm(xtry,2),realmin)) / max(eta_c,realmin);
    khat_ref = min(max(gain,1),1e20);
catch
end
khat       = min(max([1, khat_ref]),1e20);
fwd_bound  = khat * eta_c;

tol_eta      = 1e-12;
tol_fwdbound = 1e-8;

quality_ok = all(isfinite([eta_c,fwd_bound])) && all(isfinite(xtry)) && ...
             (eta_c <= tol_eta) && (fwd_bound <= tol_fwdbound);

if isempty(msg) && quality_ok
    xsol = full(xtry);
    return
end

% ---------- 3) Fallback: preconditioned Krylov on Ap ----------
Aop  = @(x) Ap * x;    ATop = @(y) Ap' * y;

usedILU = false;  Mfun = [];
try
    ilu_opts = struct('type','ilutp','droptol',1e-2,'milu','off','udiag',false);
    [L,U] = ilu(Ap, ilu_opts);
    piv = abs(diag(U)); okILU = all(isfinite(piv)) & min(piv) > 1e-15;
    if okILU
        usedILU = true;
        Mfun = @(y) U \ (L \ y);
    end
catch
end
if isempty(Mfun)
    D = abs(diag(Ap)); D(~isfinite(D) | D==0) = 1;
    Mfun = @(y) y ./ D;    % damped Jacobi
end

tol_relres = 1e-8; maxit = 300; restart = 40;

[xp,flag_gm,rel_gm,iter_gm] = gmres(@(x) Aop(Mfun(x)), bp, restart, tol_relres, ceil(maxit/restart));
gm_ok = (flag_gm==0) && isfinite(rel_gm);
if gm_ok
    xp = Mfun(xp);
else
    [xp,flag_bi,rel_bi,iter_bi] = bicgstab(@(x) Aop(Mfun(x)), bp, tol_relres, maxit);
    bi_ok = (flag_bi==0) && isfinite(rel_bi);
    if bi_ok, xp = Mfun(xp); end
end
krylov_ok = exist('xp','var') && (gm_ok || (exist('bi_ok','var') && bi_ok));

% Short refinement on scaled system
if krylov_ok
    for k = 1:2
        rp = bp - Aop(xp);
        if norm(rp,inf) <= 1e-12*(norm(Ap,inf)*max(norm(xp,inf),1)+norm(bp,inf)), break; end
        dxp = gmres(@(x) Aop(Mfun(x)), rp, restart, min(1e-12,tol_relres/100), ceil(100/restart));
        dxp = Mfun(dxp);
        xp  = xp + dxp;
    end
end

% ---------- 4) Regularized normal equations if Krylov failed ----------
reg_ok = false;
if ~krylov_ok
    lam = 1e-8 * norm(A,1);
    Nop = @(z) ATop(Aop(z)) + lam*z;
    [xp,flag_reg,rel_reg,it_reg] = pcg(Nop, ATop(bp), tol_relres, 500);
    reg_ok = (flag_reg==0) && isfinite(rel_reg);
end

% ---------- 5) LSQR last resort ----------
ls_ok = false;
if ~krylov_ok && ~reg_ok
    Af_op = @(flag,x) strcmp(flag,'notransp')*Aop(x) + strcmp(flag,'transp')*ATop(x);
    [xp,flag_ls,rel_ls,it_ls] = lsqr(Af_op, bp, tol_relres, 2000);
    ls_ok = (flag_ls==0) || (flag_ls==1 && rel_ls<1e-6);
end

% ---------- 6) Map back ----------
if exist('xp','var')
    y = zeros(n,1); y(p) = xp;
    xsol = full(Dc * y);
else
    error('SolverFailed:AllFallbacks','All fallbacks failed to produce xp.');
end

% Optional: emit a compact warning about the path taken
if krylov_ok
    if gm_ok
        warning('FallbackIterative:GMRES','Fallback GMRES used. relres=%g it=%s ILU=%d.', rel_gm, mat2str(iter_gm), usedILU);
    else
        warning('FallbackIterative:BiCGSTAB','Fallback BiCGSTAB used. relres=%g it=%s ILU=%d.', rel_bi, mat2str(iter_bi), usedILU);
    end
elseif reg_ok
    warning('FallbackRegularized:Used','Regularized normal equations used.');
elseif ls_ok
    res = A*(Dc*y) - b_sparse;
    relres_final = norm(res,2)/max(norm(b_sparse,2),1);
    warning('FallbackLSQR:Used','LSQR used. relres2=%g it=%s.', relres_final, mat2str(it_ls));
end

end
 