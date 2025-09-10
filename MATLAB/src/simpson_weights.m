function w = simpson_weights(N, h)
% Simpson's 1/3 rule weights for uniform or non-uniform grids.
%
% Parameters:
%   N : Number of points (must be odd and >= 3). Ignored if `h` is a vector.
%   h : Either:
%       - scalar → uniform spacing
%       - vector of grid coordinates (must be sorted and of odd length)
%
% Returns:
%   w : Weights vector of length N

    if isnumeric(h) && isscalar(h)  % Uniform grid
        if N < 3 || mod(N, 2) == 0
            error("N must be an odd integer >= 3 for Simpson’s rule.");
        end
        w = ones(1, N);
        w(2:2:N-1) = 4;
        w(3:2:N-2) = 2;
        w = w * (h / 3);

    elseif isnumeric(h) && isvector(h)  % Non-uniform grid
        x = h(:);
        N = length(x);
        if N < 3 || mod(N, 2) == 0
            error("Grid length must be odd and >= 3 for Simpson’s rule.");
        end
        w = zeros(N, 1);

        % Composite Simpson rule over each interval triplet
        for i = 1:2:N-2
            x0 = x(i);   x1 = x(i+1);   x2 = x(i+2);
            h0 = x1 - x0;
            h1 = x2 - x1;

            % Exact quadratic weights
            w0 =   h0/3 +  h1/6   - (h1^2)/(6*h0);
            w1 =  (h0^2)/(6*h1) + h0/2 + h1/2 + (h1^2)/(6*h0);
            w2 = -(h0^2)/(6*h1) + h0/6 + h1/3;

            w(i)   = w(i)   + w0;
            w(i+1) = w(i+1) + w1;
            w(i+2) = w(i+2) + w2;
        end

    else
        error("Invalid type for `h`. Must be scalar or 1D vector.");
    end
end
