

Gamma_vals        = 50; %linspace(0.1, 10, 5);
Fr_vals           = [1, 10, 100];%logspace(-2, 2, 11);
Re_vals           = Inf; %logspace(3, 6, 3);
Kappa_vals        = logspace(-3, 0, 201);
We_vals           = 1; %logspace(-1, 3, 5);
Lambda_vals       = 0.6; %linspace(0.1, 0.5, 3);
MotorPosRatio_vals = linspace(-.9, .9, 7);  % example range

T = sweep_nd_groups(Gamma_vals, Fr_vals, Re_vals, Kappa_vals, We_vals, Lambda_vals, MotorPosRatio_vals);


function T = sweep_nd_groups(Gamma_vals, Fr_vals, Re_vals, Kappa_vals, We_vals, Lambda_vals, MotorPosRatio_vals)

    addpath('../../src');

    % --- fixed dimensional constants (edit as needed)
    base = struct( ...
        'sigma',       72.2e-3, ...
        'rho',         1000, ...
        'g',           9.81, ...
        'motor_force', 50e-6, ... %0.13e-3*2.5e-3, ...
        'BC',          'radiative', ...
        'rho_raft_base', 0.052, ...
        'L_raft_base', 0.05 );

    % --- build full ND grid (now 7D including motor_pos_ratio)
    [GAM, FR, RE, KAP, WE, LAM, MPR] = ndgrid( ...
        Gamma_vals(:), Fr_vals(:), Re_vals(:), ...
        Kappa_vals(:), We_vals(:), Lambda_vals(:), MotorPosRatio_vals(:));

    nCases = numel(GAM);

    % --- preallocate storage for outputs
    Gamma_col        = GAM(:);
    Fr_col           = FR(:);
    Re_col           = RE(:);
    Kappa_col        = KAP(:);
    We_col           = WE(:);
    Lambda_col       = LAM(:);
    MotorPosRatio_col = MPR(:);

    eta_1_col    = NaN(nCases,1);
    eta_end_col  = NaN(nCases,1);
    thrust_col   = NaN(nCases,1);
    success_col  = false(nCases,1);
    n_used_col   = NaN(nCases,1);
    M_used_col   = NaN(nCases,1);

    % --- start / reuse parallel pool with exactly 3 workers
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', 3);   % 3 workers as requested
    elseif pool.NumWorkers ~= 3
        delete(pool);
        pool = parpool('local', 3);
    end

    % --- progress tracking (every 5%)
    pctStep  = 0.05;   % 5%
    nextFrac = pctStep;
    tStart   = tic;
    completed = 0;

    dq = parallel.pool.DataQueue;
    afterEach(dq, @updateProgress);

    fprintf('Starting ND sweep with %d cases on %d workers...\n', nCases, pool.NumWorkers);

    % --- main parallel loop
    parfor i = 1:nCases
        Gamma          = Gamma_col(i);
        Fr             = Fr_col(i);
        Re             = Re_col(i);
        Kappa          = Kappa_col(i);
        We             = We_col(i);
        Lambda         = Lambda_col(i);
        motor_pos_ratio = MotorPosRatio_col(i);

        % ND -> dimensional
        p_i = nd_to_dimensional(Gamma, Fr, Re, Kappa, We, Lambda, motor_pos_ratio, base);

        % run solver once
        [eta1, etaEnd, thrust, success, n_used, M_used] = run_once_nd(p_i);

        % store results
        eta_1_col(i)   = eta1;
        eta_end_col(i) = etaEnd;
        thrust_col(i)  = thrust;
        success_col(i) = success;
        n_used_col(i)  = n_used;
        M_used_col(i)  = M_used;

        % notify progress
        send(dq, i);
    end

    fprintf('ND sweep complete. Elapsed time: %s\n', fmtTime(toc(tStart)));

    % --- assemble table (only output)
    T = table(Gamma_col, Fr_col, Re_col, Kappa_col, We_col, Lambda_col, MotorPosRatio_col, ...
              eta_1_col, eta_end_col, thrust_col, ...
              n_used_col, M_used_col, success_col, ...
        'VariableNames', {'Gamma','Fr','Re','Kappa','We','Lambda','motor_pos_ratio', ...
                          'eta_1','eta_end','thrust','n_used','M_used','success'});

    % ------------ nested helper for progress / ETA ------------
    function updateProgress(~)
        % This nested function shares completed, nextFrac, tStart, nCases, pctStep
        completed = completed + 1;
        frac = completed / nCases;
        if frac >= nextFrac || completed == nCases
            elapsed = toc(tStart);
            estTotal = elapsed / max(frac, eps);
            estRemain = estTotal - elapsed;
            fprintf('Progress: %3.0f%%  |  elapsed: %s  |  ETA: %s\n', ...
                100*frac, fmtTime(elapsed), fmtTime(estRemain));
            nextFrac = nextFrac + pctStep;
        end
    end
end


function s = fmtTime(tsec)
% crude hh:mm:ss formatter
    if tsec < 60
        s = sprintf('%.1fs', tsec);
    elseif tsec < 3600
        m = floor(tsec/60);
        s = sprintf('%dm %02ds', m, round(tsec - 60*m));
    else
        h = floor(tsec/3600);
        rem = tsec - 3600*h;
        m  = floor(rem/60);
        s = sprintf('%dh %02dm', h, m);
    end
end


function [eta1, etaEnd, thrust, success, n_used, M_used] = run_once_nd(p)
% Run flexible_surferbot_v2 once and detect warnings/errors.
% On any warning or error, success = false and outputs are left NaN.

    eta1    = NaN;
    etaEnd  = NaN;
    thrust  = NaN;
    success = false;
    n_used  = NaN;
    M_used  = NaN;

    % reset last warning
    lastwarn('');

    try
        % call the solver
        [~, ~, ~, ~, eta, args] = flexible_surferbot_v2( ...
            'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
            'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
            'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
            'L_domain', p.L_domain, 'n',p.n,'M',p.M, ...
            'motor_force',p.motor_force,'BC',p.BC);

        % check for warnings
        [warnMsg, ~] = lastwarn;
        if ~isempty(warnMsg)
            % treat any warning as failure; leave NaNs
            success = false;
            return;
        end

        % if we get here, no warnings and no errors
        success = true;

        eta1   = eta(1);
        etaEnd = eta(end);

        % thrust per span (as in your original code)
        thrust = args.thrust / args.d;

        % record resolution actually used
        n_used = args.n;
        M_used = args.M;

    catch
        % any error => treat as failure, leave NaNs
        success = false;
    end
end


function p = nd_to_dimensional(Gamma, Fr, Re, Kappa, We, Lambda, motor_pos_ratio, base)
% Convert ND groups -> dimensional parameters for flexible_surferbot_v2

    rho   = base.rho;
    g     = base.g;
    sigma = base.sigma;

    % --- length
    L = sqrt(Gamma * We * sigma / (rho * g * Fr^2));

    % --- width/thickness-like geometric scale
    d = Lambda * L;

    % --- frequency
    omega = (Fr^(3/2)) * (g^(3/4)) * (rho^(1/4)) / ...
            (Gamma^(1/4) * We^(1/4) * sigma^(1/4));

    % --- bending stiffness EI
    EI = Kappa * (Gamma^(3/2)) * (We^(5/2)) * (sigma^(5/2)) / ...
         (Fr^3 * g^(3/2) * rho^(3/2));

    % --- linear mass density of raft
    rho_raft = rho * L^2 / Gamma;  % this is rho_R

    % --- kinematic viscosity from Re
    nu = L^2 * omega / Re;

    % --- map to parameter struct used by solver
    p = struct();
    p.sigma         = sigma;
    p.rho           = rho;
    p.g             = g;
    p.nu            = nu;
    p.L_raft        = L;
    p.motor_position = motor_pos_ratio * L/2;   % swept ND motor position
    p.d             = d;
    p.EI            = EI;
    p.rho_raft      = rho_raft;
    p.domainDepth   = NaN;   % let solver choose defaults
    p.L_domain      = NaN;
    p.n             = NaN;
    p.M             = NaN;
    p.motor_force   = base.motor_force;
    p.BC            = base.BC;
    p.omega         = omega;
    p.ooa           = 4;     % as before
end
