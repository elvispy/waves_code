function [D, x] = buildMappedFD_anyOrder(x1, xn, n, m, ooa, args)
% buildMappedFD_anyOrder
% Finite-difference matrix for d^m/dx^m on a smoothly clustered grid
% via uniform->physical exponential mapping and a coefficient recurrence.
%
% Inputs:
%   x1, xn : domain endpoints (x1 < xn)
%   n      : number of points
%   m      : derivative order (integer >= 1)
%   ooa    : even order of accuracy for FD in xi (e.g., 2,4,6)
%   args   : optional struct; if present, args.k is expected decay rate near x1
%            (so decay length ell = 1/args.k). Used to pick clustering beta.
%
% Outputs:
%   D : n-by-n sparse matrix approximating d^m/dx^m at nodes x
%   x : 1-by-n row vector of physical nodes [x1 ... xn]
%
% Requires: getNonCompactFDmatrix (your uniform-grid FD builder)

% ---- checks
if ~(isscalar(x1) && isscalar(xn) && x1 < xn), error('Require x1 < xn.'); end
if ~(isscalar(n) && n >= max(3, m+1)), error('n too small for order m.'); end
if ~(isscalar(m) && m>=1 && m==floor(m)), error('m must be integer >= 1'); end
if ~(isscalar(ooa) && ooa>=2), error('ooa must be even >=2'); end
if mod(ooa,2)~=0
    warning('ooa should be even; bumping to next even.');
    ooa = ooa+1;
end

% ---- uniform computational grid
xi  = linspace(0,1,n);
dxi = 1/(n-1);
L   = xn - x1;

% ---- choose clustering beta from expected decay (optional)
if nargin < 6 || ~isstruct(args) || ~isfield(args,'k') || ~isscalar(args.k) || args.k<=0
    k = 10 / L;  % gentle default: ell ~ L/10
else
    k = args.k;
end
ell       = 1 / k;
Delta_x0  = min(ell/15, L/max(10,1.0*(n-1))); % ~15 pts per decay length & cap

% Solve for beta from Δx0 ≈ L * [beta/(exp(beta)-1)] / (n-1)
beta_fun  = @(b) L * (b ./ max(exp(b)-1, eps)) / (n-1) - Delta_x0;
try
    beta = fzero(beta_fun, [1e-8, 50]);
    if ~isfinite(beta) || beta<=0, beta = 2.0; end
catch
    beta = 2.0;
end
%fprintf("beta: %.2e \n", beta);

% ---- mapping and metrics
expb = exp(beta); den = expb - 1;
g    = @(t) x1 + L * (exp(beta*t) - 1) / den;
gp   = @(t) L * (beta*exp(beta*t)) / den;  % g'(xi)
x    = g(xi);
a    = (1 ./ gp(xi)).';                    % a = 1/g'(xi), column vector (n×1)

% ---- FD matrices in xi: Dxi^(k), k=1..m (uniform grid, order ooa)
%Dxi1 = getNonCompactFDmatrix(n, dxi, 1, ooa);
DxiK = cell(1, m);
%DxiK{1} = Dxi1;
for kOrd = 1:m
    DxiK{kOrd} = getNonCompactFDmatrix(n, dxi, kOrd, ooa);
end

% ---- Recurrence for coefficients c_{m,k} at nodes (each is n×1)
% c(1,1) = a ; c(1,k>1) = 0
C = cell(m,1);  % each C{t} is an n-by-t matrix holding c_{t,1..t}
C{1} = a;       % n×1

for t = 1:(m-1)
    ct   = C{t};                  % n×t
    ctp1 = zeros(n, t+1);         % will hold c_{t+1,1..t+1}
    % first derivative in xi of each column of ct
    dct  = DxiK{1} * ct;             % n×t
    % recurrence: for k=1..t
    %   c_{t+1,k}   = a .* (d/dxi c_{t,k})
    %   c_{t+1,k+1} = a .* c_{t,k}
    ctp1(:,1:t)   = a .* dct;     % n×t
    ctp1(:,2:t+1) = ctp1(:,2:t+1) + a .* ct;
    C{t+1} = ctp1;
end

% ---- Assemble Dx^(m) = sum_{k=1..m} diag(c_{m,k}) * Dxi^(k)
cm = C{m};   % n×m
D  = spalloc(n, n, (m*(ooa+m))*n);  % rough sparsity guess
for kOrd = 1:m
    D = D + spdiags(cm(:,kOrd), 0, n, n) * DxiK{kOrd};
end

% row vector for x
x = reshape(x, 1, []);

end
