addpath './src'

% Names of outputs you want
names = {'U', 'x', 'z', 'phi', 'eta', 'args'};

[out{1:numel(names)}] = flexible_surferbot_v2( ...
    'sigma'         , 0*72.2e-3      , ...   % [N m?¹] surface tension
    'rho'           , 1000.0       , ...   % [kg m?³] water density
    'omega'         , 2*pi*10      , ...   % [rad s?¹] drive frequency
    'nu'            , 0*1.0e-6       , ...   % [m² s?¹] kinematic viscosity
    'g'             , 10*9.81         , ...   % [m s?²] gravity
    'L_raft'        , 0.05         , ...   % [m] raft length
    'motor_position', 1.9e-2/2   , ...   % [m] motor x-position (from -L/2 to L/2)
    'd'             , 0.03         , ...   % [m] raft depth (spanwise)
    'EI'            , 100*3.0e9 * 3e-2 * (9.9e-4)^3 / 12 , ...  % [N m²] bending stiffness
    'rho_raft'      , 0.018*3.0    , ...   % [kg m?¹] linear mass
    'L_domain'      , 0.2          , ...   % [m] domain length
    'domainDepth'   , 0.2          , ...   % [m] water depth
    'n'             , 201          , ...   % grid points in the raft
    'M'             , 100          , ...   % gird points in the z direction
    'motor_inertia' , 0.13e-3*2.5e-3, ...  % [kg m²] motor inertia
    'BC'            , 'radiative'        );% boundary-condition type


% Build struct from names + values
S = cell2struct(out(:), names(:), 1);

plot_surferbot_run(S);