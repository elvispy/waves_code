function M = DtN_generator(N, h)
% DtN_GENERATOR  Generate the DtN matrix of size N×N
%   M = DtN_generator(N) uses h = 1.
%   M = DtN_generator(N, h) scales the result by h.
%
%   M ? (h/?) * [ D + I + T ],
%   where D is the 5?point banded part, I is the identity,
%   and T is the long?range Toeplitz from the integral.

    if nargin<2, h = 1.0; end

    %% 1) Finite?difference 5?point stencil
    DtN = diag(66*ones(N,1));
    if N>1
        DtN = DtN + diag(-32*ones(N-1,1),1) + diag(-32*ones(N-1,1),-1);
    end
    if N>2
        DtN = DtN + diag(-1*ones(N-2,1),2) + diag(-1*ones(N-2,1),-2);
    end
    DtN = DtN/18 + eye(N);

    %% 2) Build the long?range coefficients exactly as in Python
    coeffs = zeros(N+1,1);
    coef = @(n,d) -n./(n + d) + (2*n - d)/2 .* log((n+1)./(n-1)) - 1;

    % Python does: for jj in 1:(N/2-1), n=2*jj+1
    max_jj = floor(N/2) - 1;
    for jj = 1:max_jj
        n = 2*jj + 1;
        coeffs(n)   = coeffs(n)   + coef(n, -1);
        coeffs(n+2) = coeffs(n+2) + coef(n, +1);
        coeffs(n+1) = coeffs(n+1) - 2*coef(n,  0);
    end

    %% 3) Toeplitz from the first N entries of `coeffs`
    % T(i,j) = coeffs(|i-j|+1)
    c = coeffs(1:N);
    T = toeplitz(c);

    %% 4) Combine and scale
    M = h * (DtN + T) / pi;
end
