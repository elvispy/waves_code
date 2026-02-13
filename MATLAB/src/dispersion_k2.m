function k = dispersion_k2(omega, EI, g, H, rho, rho1, h, T)
% Solve for real positive k from:
%   omega^2 = (D k^5/rho - T k^3/rho + g k) / (k h' + coth(kH))
%   h' = rho1*h/rho
%
% Here we use D = EI (per unit width). Adjust if needed.

    if ~isfinite(omega) || omega <= 0
        k = NaN; return;
    end

    D = EI;                       % <-- mapping assumption
    hprime = (rho1*h)/rho;

    cothf = @(x) cosh(x)./sinh(x);

    % Residual in k
    f = @(kk) ( (D*kk.^5 - T*kk.^3 + rho*g*kk) ./ (rho*(kk*hprime + cothf(kk*H))) ) - omega.^2;

    % Good default guess: gravity-wave-like scaling
    k0 = max(omega^2/g, 1e-6);

    % Try fzero with a single guess; if that fails, bracket by scanning
    try
        k_try = fzero(f, k0);
        if isfinite(k_try) && k_try > 0
            k = k_try; return;
        end
    catch
    end

    % Bracket scan around k0
    ks = logspace(log10(k0/1e3), log10(k0*1e3), 80);
    vals = arrayfun(f, ks);
    good = isfinite(vals);

    idx = find(good(1:end-1) & good(2:end) & (vals(1:end-1).*vals(2:end) <= 0), 1, 'first');
    if isempty(idx)
        k = NaN; return;
    end

    try
        k = fzero(f, [ks(idx) ks(idx+1)]);
        if ~(isfinite(k) && k > 0), k = NaN; end
    catch
        k = NaN;
    end
end
