addpath '../src'

% Names of outputs you want
names = {'U', 'x', 'z', 'phi', 'eta', 'args'};

L_raft = 0.05;
[out{1:numel(names)}] = flexible_surferbot_v2( ...
    'sigma'         , 72.2e-3      , ...   % [N/m] surface tension
    'rho'           , 1000.0        , ...   % [kg/m3] water density
    'omega'         , 2*pi*80        , ...   % [rad 1/s] drive frequency
    'nu'            , 0*1.0e-6      , ...   % [m2/s] kinematic viscosity
    'g'             , 9.81          , ...   % [m/s2] gravity
    'L_raft'        , L_raft        , ...   % [m] raft length  
    'L_domain'      , 5*L_raft      , ...
    'motor_position', 0.24*L_raft/2  , ...   % [m] motor x-position (from -L/2 to L/2)
    'd'             , 0.03         , ...   % [m] raft depth (spanwise)
    'EI'            , 3.0e9 * 3e-2 * (9.9e-4)^3 / 12 , ...  % [N m�] bending stiffness
    'rho_raft'      , 0.052        , ...   % [kg/m] linear mass %'loads'         , linspace(-1, 1, 101), ... %'domainDepth'   , 10*L_raft     , ...   % [m] water depth % %'n'             , 101          , ...   % grid points in the raft %'M'             , 200          , ...   % gird points in the z direction
    'motor_inertia' , 0.13e-3*2.5e-3, ...  % [kg m�] motor inertia
    'BC'            , 'radiative'        );% boundary-condition type


% Build struct from names + values
S = cell2struct(out(:), names(:), 1);

plot_surferbot_run(S);
%% 