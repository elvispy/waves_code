function [D, x, beta] = buildMappedFD(x1, xn, n, m, ooa, args)
% buildMappedFD  (production version for m=1,2)
%   Sparse mapped FD operator using analytic metric terms (no recurrence).
%
% Inputs:
%   x1, xn : domain endpoints (x1 < xn)
%   n      : number of grid points
%   m      : derivative order (1 or 2)
%   ooa    : even interior order (e.g., 2,4,6)
%   args   : optional struct
%            args.beta  (positive scalar)  -> fixed mapping strength
%            args.k     (decay rate)       -> only used to *choose* beta if no beta
%
% Outputs:
%   D      : n-by-n sparse operator approximating d^m/dx^m at nodes x
%   x      : 1-by-n node vector
%   beta   : mapping parameter actually used

    % -- checks
    if ~(isscalar(x1) && isscalar(xn) && x1 < xn), error('Require x1<xn'); end
    if ~(isscalar(n) && n >= max(3, m+1)), error('n too small'); end
    if ~(isscalar(m) && any(m==[1 2])), error('This build supports m=1 or m=2'); end
    if ~(isscalar(ooa) && ooa>=2), error('ooa must be even >=2'); end
    if mod(ooa,2)~=0, ooa = ooa+1; end

    % -- uniform computational grid
    xi  = linspace(0,1,n);
    dxi = 1/(n-1);
    L   = xn - x1;

    % -- choose mapping parameter beta (FIXED across refinements)
    if nargin>=6 && isstruct(args) && isfield(args,'beta') && args.beta>0
        beta = args.beta;
    else
        % sensible default if not provided: moderate clustering
        beta = 2.0;
        % (If you really want to estimate from args.k, do it once and re-use.)
    end

    % -- exponential map and metric terms
    % g(ξ) = x1 + L*(e^{βξ}-1)/(e^{β}-1)
    expb = exp(beta); den = max(expb - 1, eps);
    ee  = exp(beta*xi);
    x    = x1 + L*(ee - 1)/den;

    g1   = L*(beta*ee)/den;          % g'(ξ)
    g2   = L*(beta^2*ee)/den;        % g''(ξ)

    % -- FD matrices in ξ (your trusted builder)
    Dxi  = getNonCompactFDmatrix(n, dxi, 1, ooa);
    if m==2
        Dxi2 = getNonCompactFDmatrix(n, dxi, 2, ooa);
    end

    % -- assemble Dx using exact metric formulas (no discrete coeff derivatives)
    switch m
        case 1
            D = spdiags(1./g1(:), 0, n, n) * Dxi;

        case 2
            D = spdiags(1./(g1(:).^2), 0, n, n) * Dxi2 ...
              - spdiags((g2(:)./(g1(:).^3)), 0, n, n) * Dxi;
    end

    x = reshape(x,1,[]);
end
