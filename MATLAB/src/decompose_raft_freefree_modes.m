function modal = decompose_raft_freefree_modes(S, varargin)
%DECOMPOSE_RAFT_FREEFREE_MODES Modal projection of raft response onto free-free modes.
%
% modal = DECOMPOSE_RAFT_FREEFREE_MODES(S)
% modal = DECOMPOSE_RAFT_FREEFREE_MODES(S, 'num_modes', 8, ...
%     'include_rigid', true, 'verbose', true)
%
% Inputs:
%   S: struct from flexible_surferbot_v2 containing fields:
%      - S.x, S.eta, S.args
%      - S.args.x_contact, S.args.pressure, S.args.loads
%
% Optional parameters:
%   'num_modes'     : total number of modes to use (default 8)
%   'include_rigid' : include rigid translation/rotation modes (default true)
%   'verbose'       : print modal table and diagnostics (default true)
%
% Output:
%   modal struct with fields:
%      n, mode_type, betaL, beta, q, Q, F, balance_residual,
%      energy_frac, eta_recon, recon_rel_err, x_raft, Psi, gram_cond
%
% Notes on notation used below:
%   - eta is the complex displacement on raft nodes.
%   - Q and F are modal projections of d*p and f, respectively.
%   - balance_residual checks:
%       (EI*beta_n^4 - rho_raft*omega^2) q_n  ?  (Q_n - F_n)
%     for each retained mode.

    p = inputParser;
    addParameter(p, 'num_modes', 8, @(v) isnumeric(v) && isscalar(v) && (v >= 1) && (round(v) == v));
    addParameter(p, 'include_rigid', true, @(v) islogical(v) && isscalar(v));
    addParameter(p, 'verbose', true, @(v) islogical(v) && isscalar(v));
    parse(p, varargin{:});
    opts = p.Results;

    % --- Validate required fields and dimensions ---
    assert(isstruct(S) && isfield(S, 'args'), 'Input S must be a struct with field S.args.');
    args = S.args;
    requiredArgs = {'x_contact', 'L_raft', 'd', 'EI', 'rho_raft', 'omega', 'pressure', 'loads'};
    for i = 1:numel(requiredArgs)
        assert(isfield(args, requiredArgs{i}), 'Missing required field: S.args.%s', requiredArgs{i});
    end
    assert(isfield(S, 'x') && isfield(S, 'eta'), 'Input S must contain fields S.x and S.eta.');

    contact = logical(args.x_contact(:));
    x_all = S.x(:);
    eta_all = S.eta(:);
    assert(numel(contact) == numel(x_all), 'Length mismatch: S.args.x_contact vs S.x.');
    assert(numel(eta_all) == numel(x_all), 'Length mismatch: S.eta vs S.x.');

    x_raft = x_all(contact);
    eta = eta_all(contact);
    Nr = numel(x_raft);
    assert(Nr >= 3, 'Need at least 3 raft points for modal decomposition.');

    p_raft = args.pressure(:);
    f_raft = args.loads(:);
    assert(numel(p_raft) == Nr, 'S.args.pressure must have %d entries (raft points).', Nr);
    assert(numel(f_raft) == Nr, 'S.args.loads must have %d entries (raft points).', Nr);

    % --- Quadrature weights (trapz on raft grid) ---
    % We store weights as a vector so weighted inner products can be written
    % as <a,b>_W = a' * (b .* w), without explicitly building diag(w).
    w = trapz_weights(x_raft);
    Weta = eta .* w;
    Wdp = (args.d * p_raft) .* w;
    Wf = f_raft .* w;

    % --- Build requested mode bank ---
    n_requested = opts.num_modes;
    n_max = Nr;
    if n_requested > n_max
        % With more modes than points the basis matrix becomes rank-limited.
        warning('Requested %d modes but only %d raft points available. Capping modes.', n_requested, n_max);
        n_requested = n_max;
    end

    n_rigid = 0;
    if opts.include_rigid
        n_rigid = min(2, n_requested);
    end
    n_elastic = n_requested - n_rigid;

    % Analytic free-free expressions are written on xi in [0, L].
    xi = x_raft + args.L_raft / 2; % map from [-L/2, L/2] to [0, L]
    Phi_raw = zeros(Nr, n_requested);
    n_list = (0:(n_requested - 1)).'; % external mode labels printed in logs
    mode_type = repmat({''}, n_requested, 1);
    betaL = zeros(n_requested, 1);
    beta = zeros(n_requested, 1);

    col = 0;
    if n_rigid >= 1
        col = col + 1;
        % Free-free rigid translation mode.
        Phi_raw(:, col) = ones(Nr, 1);
        mode_type{col} = 'rigid';
    end
    if n_rigid >= 2
        col = col + 1;
        % Free-free rigid rotation mode around raft center.
        Phi_raw(:, col) = xi - args.L_raft / 2;
        mode_type{col} = 'rigid';
    end

    if n_elastic > 0
        % Elastic roots satisfy cosh(betaL)*cos(betaL) = 1.
        betaL_el = freefree_betaL_roots(n_elastic);
        for j = 1:n_elastic
            col = col + 1;
            betaL(col) = betaL_el(j);
            beta(col) = betaL(col) / args.L_raft;
            % Mode shape family matches utils/simple_mode_video.m.
            Phi_raw(:, col) = freefree_mode_shape(xi, args.L_raft, betaL(col));
            mode_type{col} = 'elastic';
        end
    end

    % --- Weighted orthonormalization (modified Gram-Schmidt) ---
    % This stabilizes projections when raw shapes are nearly collinear on
    % the discrete grid. keep marks columns that survived deflation.
    [Psi, keep] = weighted_mgs(Phi_raw, w);
    if ~all(keep)
        warning('Dropped %d nearly dependent mode(s) during weighted orthonormalization.', sum(~keep));
    end

    if isempty(Psi)
        error('No valid modes remained after orthonormalization.');
    end

    n_list = n_list(keep);
    mode_type = mode_type(keep);
    betaL = betaL(keep);
    beta = beta(keep);

    % --- Projection using weighted least squares expression ---
    % We keep the explicit M and RHS form:
    %   M = Psi^H W Psi, rhs = Psi^H W y, coeff = M^{-1} rhs
    % even though Psi is approximately W-orthonormal after MGS.
    M = Psi' * (Psi .* w);
    rhs_q = Psi' * Weta;
    rhs_Q = Psi' * Wdp;
    rhs_F = Psi' * Wf;

    gram_cond = cond(M);
    % Fallback protects against loss of rank from discrete sampling.
    use_pinv = ~isfinite(gram_cond) || (gram_cond > 1e10);
    if use_pinv
        warning('Ill-conditioned Gram matrix (cond=%.3e). Using pinv fallback.', gram_cond);
        Minv = pinv(M);
        q = Minv * rhs_q;
        Q = Minv * rhs_Q;
        F = Minv * rhs_F;
    else
        q = M \ rhs_q;
        Q = M \ rhs_Q;
        F = M \ rhs_F;
    end

    beta4 = beta.^4;
    % Modal beam-balance residual. Small values indicate the projected
    % numerical solution is consistent with the modal equation.
    balance_residual = (args.EI * beta4 - args.rho_raft * args.omega^2) .* q - (Q - F);

    eta_recon = Psi * q;
    % Weighted relative reconstruction error on raft interval.
    recon_num = sqrt(real((eta - eta_recon)' * ((eta - eta_recon) .* w)));
    recon_den = sqrt(real(eta' * (eta .* w)));
    if recon_den > 0
        recon_rel_err = recon_num / recon_den;
    else
        recon_rel_err = NaN;
    end

    % Modal energy fraction based on |q_n|^2 (diagnostic only).
    q_energy = abs(q).^2;
    q_energy_sum = sum(q_energy);
    if q_energy_sum > 0
        energy_frac = q_energy / q_energy_sum;
    else
        energy_frac = zeros(size(q_energy));
    end

    modal = struct();
    modal.n = n_list;                          % external mode indices used in logs
    modal.mode_type = mode_type;               % 'rigid' or 'elastic' per retained mode
    modal.betaL = betaL;                       % nondimensional wavenumber beta_n * L
    modal.beta = beta;                         % dimensional modal wavenumber beta_n [1/m]
    modal.q = q;                               % modal coefficients of eta
    modal.Q = Q;                               % modal coefficients of d * pressure
    modal.F = F;                               % modal coefficients of forcing load
    modal.balance_residual = balance_residual; % per-mode beam-balance mismatch
    modal.energy_frac = energy_frac;           % |q_n|^2 normalized across retained modes
    modal.eta_recon = eta_recon;               % reconstructed raft displacement Psi * q
    modal.recon_rel_err = recon_rel_err;       % weighted relative reconstruction error
    modal.x_raft = x_raft;                     % physical raft-node coordinates [m]
    modal.Psi = Psi;                           % weighted-orthonormal basis on raft grid
    modal.gram_cond = gram_cond;               % condition number of Psi' * W * Psi

    if opts.verbose
        print_modal_log(modal);
    end
end

function w = trapz_weights(x)
    % Return vector weights so that trapz(x, f) == sum(w .* f) on the same grid.
    x = x(:);
    n = numel(x);
    if n == 1
        w = 1;
        return;
    end
    dx = diff(x);
    if any(dx <= 0)
        error('x_raft must be strictly increasing for trapz weights.');
    end
    w = zeros(n, 1);
    w(1) = dx(1) / 2;
    w(end) = dx(end) / 2;
    if n > 2
        w(2:(end - 1)) = (x(3:end) - x(1:(end - 2))) / 2;
    end
end

function betaL = freefree_betaL_roots(n)
    % Positive elastic roots of cosh(betaL)*cos(betaL) = 1.
    % Each root is bracketed in [k*pi, (k+1)*pi], k >= 1.
    betaL = zeros(n, 1);
    f = @(y) cosh(y) .* cos(y) - 1;
    for k = 1:n
        a = k * pi;
        b = (k + 1) * pi;
        betaL(k) = fzero(f, [a, b]);
    end
end

function psi = freefree_mode_shape(xi, L, betaL)
    % Free-free elastic mode shape in xi in [0, L].
    % The max-abs normalization is temporary; final normalization is done
    % by weighted MGS using the actual discrete quadrature.
    beta = betaL / L;
    bx = beta * xi;
    alpha = (sin(betaL) - sinh(betaL)) / (cosh(betaL) - cos(betaL));
    psi = (sin(bx) + sinh(bx)) + alpha * (cos(bx) + cosh(bx));

    scale = max(abs(psi));
    if isfinite(scale) && scale > 0
        psi = psi / scale;
    end
end

function [Psi, keep] = weighted_mgs(Phi, w)
    % Weighted modified Gram-Schmidt with inner product:
    %   <u,v>_W = u' * (v .* w)
    % keep(j) flags whether Phi(:,j) was linearly independent numerically.
    [nrow, ncol] = size(Phi);
    Psi = zeros(nrow, ncol);
    keep = false(ncol, 1);
    count = 0;

    col_norms = sqrt(max(real(sum(conj(Phi) .* (Phi .* w), 1)), 0));
    % Relative tolerance tied to largest weighted raw-column norm.
    tol = 1e-10 * max([1, col_norms]);

    for j = 1:ncol
        v = Phi(:, j);
        for k = 1:count
            proj = Psi(:, k)' * (v .* w);
            v = v - Psi(:, k) * proj;
        end

        nv = sqrt(max(real(v' * (v .* w)), 0));
        if isfinite(nv) && (nv > tol)
            count = count + 1;
            Psi(:, count) = v / nv;
            keep(j) = true;
        end
    end

    Psi = Psi(:, 1:count);
end

function print_modal_log(modal)
    % Console-first log intended for sweep/debug traceability.
    fprintf('\nModal decomposition (free-free beam basis)\n');
    fprintf('n | type    | betaL      | |q_n|      | arg(q_n)   | |Q_n|      | |F_n|      | |r_n|\n');
    fprintf('----------------------------------------------------------------------------------------\n');
    for i = 1:numel(modal.n)
        fprintf('%1d | %-7s | %10.5f | %10.3e | %10.3f | %10.3e | %10.3e | %10.3e\n', ...
            modal.n(i), modal.mode_type{i}, modal.betaL(i), abs(modal.q(i)), angle(modal.q(i)), ...
            abs(modal.Q(i)), abs(modal.F(i)), abs(modal.balance_residual(i)));
    end
    fprintf('Diagnostics: sum(energy_frac)=%.6f, recon_rel_err=%.3e, cond(Psi''W Psi)=%.3e\n', ...
        sum(modal.energy_frac), modal.recon_rel_err, modal.gram_cond);
end
