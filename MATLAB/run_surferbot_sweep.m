function run_surferbot_sweep
% ------------------------------------------------------------------------
%  Surferbot parameter sweep with progress messages
% ------------------------------------------------------------------------

% ---------- Base variables ----------------------------------------------
base = struct( ...
    'sigma'         , 72.2e-3 , ...
    'rho'           , 1000.0  , ...
    'omega'         , 2*pi*5  , ...
    'nu'            , 1.0e-6  , ...
    'g'             , 9.81    , ...
    'L_raft'        , 0.05    , ...
    'motor_position', 0.015   , ...
    'd'             , 0.03    , ...
    'EI'            , 3.0e9*3e-2*(1e-4)^3/12 , ...
    'rho_raft'      , 0.018*3 , ...
    'L_domain'      , 0.2     , ...
    'domainDepth'   , 0.4     , ...
    'n'             , 101     , ...
    'M'             , 200     , ...
    'motor_inertia' , 0.13e-3*2.5e-3 , ...
    'BC'            , 'radiative' );

% ---------- Parameters to sweep -----------------------------------------
sweep.omega         = 2*pi * [10 30 50];
sweep.EI            = base.EI * [0.01 .1 1 10];
sweep.motor_inertia = base.motor_inertia * [0.01 .1 1 10];

% ---------- Build the Cartesian product ---------------------------------
vars   = fieldnames(sweep);
grid   = cellfun(@(v) sweep.(v), vars,'uni',0);
comb   = cell(1,numel(vars));
[comb{:}] = ndgrid(grid{:});
nRuns  = numel(comb{1});

outdir = 'surferbot_results';

% ---------- Progress monitor --------------------------------------------
dq      = parallel.pool.DataQueue;
progCnt = 0;                              % shared with nested function
afterEach(dq, @updateProgress);

% ---------- Parallel sweep ----------------------------------------------
parfor idx = 1:nRuns
    args = base;
    for k = 1:numel(vars)
        args.(vars{k}) = comb{k}(idx);
    end
    nv = reshape([fieldnames(args)'; struct2cell(args)'] ,1,[]);
    save_surferbot_run(outdir, nv{:});
    send(dq, 1);                           % tick
end

fprintf('All %d runs completed.\n', nRuns);

% ---------- Nested callback ---------------------------------------------
    function updateProgress(~)
        progCnt = progCnt + 1;
        fprintf('Completed %3d / %3d  (%.1f %%)\n', ...
                progCnt, nRuns, 100*progCnt/nRuns);
    end
end
