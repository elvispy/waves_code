function k_final = dispersion_k(omega, g, H, nu, sigma, rho, k0, num_steps)
    % Newton iteration for complex dispersion relation
    if nargin < 7 || isempty(k0)
        k0 = 1.0 + 0.0i;
    end
    if nargin < 8
        num_steps = 500;
    end

    k = k0;
    for i = 1:num_steps
        tanh_kH = tanh(k * H);
        lhs = k * tanh_kH * g;
        rhs = (-sigma / rho) * k^3 * tanh_kH + omega^2 - 4i * nu * omega * k^2;
        f = lhs - rhs;

        % Numerical derivative: central difference
        dk = 1e-8;
        f_plus = dispersion_residual(k + dk, omega, g, H, nu, sigma, rho);
        f_minus = dispersion_residual(k - dk, omega, g, H, nu, sigma, rho);
        df = (f_plus - f_minus) / (2 * dk);

        k = k - f / df;
    end

    k_final = k;
end

function f = dispersion_residual(k, omega, g, H, nu, sigma, rho)
    tanh_kH = tanh(k * H);
    lhs = k * tanh_kH * g;
    rhs = (-sigma / rho) * k^3 * tanh_kH + omega^2 - 4i * nu * omega * k^2;
    f = lhs - rhs;
end
