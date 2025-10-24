
function [k,phi,a,b,A,B] = fit_cosine(x,z,k0)
% Fits z ~ a*cos(k*x + phi) + b with unknown amplitude a and phase phi.

x = x(:); z = z(:);

function [rss,A,B,b] = solveABb(kval)
    H = [cos(kval*x) sin(kval*x) ones(size(x))];
    beta = H \ z;                 % [A; B; b]
    A = beta(1); B = beta(2); b = beta(3);
    r = H*beta - z;
    rss = r.'*r;
end

% initial k guess
if nargin<3 || isempty(k0)
    dx = mean(diff(x));
    if numel(x)>3 && max(abs(diff(diff(x)))) < 1e-6*abs(dx)
        z0 = z - mean(z);
        N = numel(z0);
        Z = fft(z0);
        f = (0:N-1)'/(N*dx);
        [~,idx] = max(abs(Z(2:floor(N/2))).^2);
        k0 = 2*pi*f(idx+1);
    else
        k0 = 2*pi/(max(x)-min(x));
    end
end

% optimize k
k = fminsearch(@(kv) solveABb(kv), k0);

% recover final params
[~,A,B,b] = solveABb(k);
a   = hypot(A,B);
phi = atan2(-B,A);
end


