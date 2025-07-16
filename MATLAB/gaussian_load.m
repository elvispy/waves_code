function delta = gaussian_load(x0, sigma, x)
    % Smooth point load on an arbitrary 1D grid.
    % Normalized so sum(delta .* w)‰ˆ 1

    x = x(:);
    dx = diff(x);
    w_mid = 0.5 * (dx(1:end-1) + dx(2:end));
    w = [0.5 * dx(1); w_mid; 0.5 * dx(end)];

    phi = exp(-0.5 * ((x - x0) / sigma).^2);
    delta = phi ./ sum(phi .* w);
end
