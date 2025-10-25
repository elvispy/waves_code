function delta = gaussian_load(x0, sigma, x)
    % Smooth point load on an arbitrary 1D grid.
    % Normalized so that ?_{-1/2}^{1/2} f(x) dx = 1
    %
    % Inputs:
    %   x0    : center of Gaussian
    %   sigma : standard deviation
    %   x     : query points (vector)
    % Output:
    %   delta : normalized Gaussian evaluated at x
    % Test: integral(@(xx) gaussian_load(x0,sigma,xx), -0.5, 0.5)  % ? 1

    a = -0.5; b = 0.5;

    % preserve caller's shape for compatibility with integral()
    sz   = length(x);
    xcol = x(:);

    % unnormalized Gaussian
    phi  = exp(-0.5 * ((xcol - x0)./sigma).^2);

    % continuous normalization on [a,b]
    Z = sigma*sqrt(pi/2) * ( erf((b-x0)/(sqrt(2)*sigma)) - erf((a-x0)/(sqrt(2)*sigma)) );
    C = 1./Z;

    % return with original shape
    delta = reshape(C*phi, sz, 1);
end
