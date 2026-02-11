addpath '../src'

% Names of outputs you want
names = {'U', 'x', 'z', 'phi', 'eta', 'args'};

L_raft = 0.05;
%E      = 330e3;       % Pa
%h      = 0.0015;      % m, thickness
%d      = 0.0021065;        % m, depth
EI0    = 1.55e-3; %E * d * h^3/12; 1.55e-3, 1.5e-4, 2.95e-5
omega0 = 2*pi*80;     % rad/s (reference)
rho_oomoo = 1.34e3;   % kg/m3

motor_position = 0.222 * L_raft; %0.3 * L_raft/2;

[out{1:numel(names)}] = flexible_surferbot_v2( ...
    'sigma'         , 72.2e-3      , ...   % [N/m] surface tension
    'rho'           , 1000.0        , ...   % [kg/m3] water density
    'omega'         , omega0     , ...   % [rad 1/s] drive frequency
    'nu'            , 0e-6      , ...   % [m2/s] kinematic viscosity
    'g'             , 9.81          , ...   % [m/s2] gravity
    'L_raft'        , L_raft        , ...   % [m] raft length   'L_domain'      , 10*L_raft      , ...
    'motor_position', motor_position  , ...   % [m] motor x-position (from -L/2 to L/2) %'d'             , d             , ...   % [m] raft depth (spanwise)
    'EI'            , EI0, ... Inf* 3.0e9 * 3e-2 * (9.9e-4)^3 / 12 , ...  % [N m�] bending stiffness
    'rho_raft'      , 0.052 , ...   % [kg/m] linear mass %'loads'         , linspace(-1, 1, 101), ... %'domainDepth'   , 10*L_raft     , ...   % [m] water depth % %'n'             , 101          , ...   % grid points in the raft %'M'             , 200          , ...   % gird points in the z direction
    'motor_force'   , 50e-6, ...  % [kg m�] motor inertia
    'forcing_width' , 1e-3, ...
    'BC'            , 'radiative'        );% boundary-condition type


% Build struct from names + values
S = cell2struct(out(:), names(:), 1); 

plot_surferbot_run(S);
%plot_surferbot_profile(S);
%plot_moving_surferbot(S);

% Modal decomposition log (non-fatal for plotting workflow)
try
    S.modal = decompose_raft_freefree_modes(S, ...
        'num_modes', 8, ...
        'include_rigid', true, ...
        'verbose', true);

    % Simple histogram-style plot: normalized modal amplitude |q_n|
    q_abs = abs(S.modal.q);
    q_max = max(q_abs);
    if q_max > 0
        q_norm = q_abs / q_max;
    else
        q_norm = zeros(size(q_abs));
    end

    figure('Color', 'w');
    bar(S.modal.n, q_norm, 0.75, 'FaceColor', [0.25 0.45 0.85], 'EdgeColor', 'none');
    xlabel('Mode number');
    ylabel('Normalized |q_n|');
    title('Modal amplitude spectrum');
    grid on;
catch ME
    warning('Modal decomposition log skipped: %s', ME.message);
end


