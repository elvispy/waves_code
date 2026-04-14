function inviscid_test_sweep(saveDir)
%INVISCID_TEST_SWEEP Run a paired coarse 3D sweep with and without viscosity.
%
% The sweep reuses the trusted parameter ranges from the existing 2D sweep
% scripts, but reduces each dimension to 5 values:
%   omega  : same range as sweep_omega_motorPosition
%   x_M    : same range as sweep_motorPosition_EI
%   EI     : same range as sweep_motorPosition_EI
%
% Two matched datasets are produced:
%   1. viscous  : nu = 1e-6
%   2. inviscid : nu = 0

if nargin < 1, saveDir = 'data'; end
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'src'));
addpath(script_dir);

L_raft = 0.05;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',1e-6, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.24*L_raft/2, 'd',0.03, ...
    'EI',3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.052, ...
    'domainDepth',nan, 'L_domain',nan, 'n',nan, 'M',nan, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*80, 'ooa',4);

omega_list = 2*pi*linspace(4, 100, 5);
motor_position_list = linspace(0.00, 0.48, 5) * L_raft;
EI_list = base.EI * 10.^linspace(-3, 1, 5);
nu_values = [1e-6, 0];
nu_names = {'viscous', 'inviscid'};

n_omega = numel(omega_list);
n_mp = numel(motor_position_list);
n_EI = numel(EI_list);
results = struct();
results.omega_list = omega_list;
results.motor_position_list = motor_position_list;
results.EI_list = EI_list;
results.base = base;
results.generated_on = datestr(now, 30);

for inu = 1:numel(nu_values)
    nu = nu_values(inu);
    label = nu_names{inu};
    fprintf('Running %s sweep (nu = %.3g)\n', label, nu);

    proto = struct( ...
        'Sxx', NaN, ...
        'alpha_beam', NaN, ...
        'eta_left_beam', NaN, ...
        'eta_right_beam', NaN, ...
        'thrust_N', NaN, ...
        'tail_flat_ratio', NaN, ...
        'success', false);
    data(n_omega, n_mp, n_EI) = proto; %#ok<AGROW>

    for iw = 1:n_omega
        fprintf('%s: %d/%d omega slices\n', label, iw, n_omega);
        parfor idx = 1:(n_mp * n_EI)
            [ip, ie] = ind2sub([n_mp, n_EI], idx);
            p = base;
            p.nu = nu;
            p.omega = omega_list(iw);
            p.motor_position = motor_position_list(ip);
            p.EI = EI_list(ie);

            data(iw, ip, ie) = run_case(p);
        end
    end

    results.(label) = data;
end

save(fullfile(saveDir, 'inviscid_test_sweep.mat'), 'results');
fprintf('Saved %s\n', fullfile(saveDir, 'inviscid_test_sweep.mat'));
end

function out = run_case(p)
[~, ~, ~, ~, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'L_domain',p.L_domain,'n',p.n,'M',p.M, ...
    'motor_inertia',p.motor_inertia,'BC',p.BC,'ooa',p.ooa);

    metrics = extract_eta_edge_metrics(eta, args);
    idx0 = max(1, ceil(0.05 * numel(eta)));
    tail = abs(eta(1:idx0));
    tail_ratio = std(tail) / max(eps, mean(tail));

    Sxx = (args.rho * args.g / 4 + 3 / 4 * args.sigma * args.k^2) * ...
          (abs(metrics.eta_left_domain)^2 - abs(metrics.eta_right_domain)^2);
    alpha_beam = -(abs(metrics.eta_left_beam)^2 - abs(metrics.eta_right_beam)^2) / ...
                 (abs(metrics.eta_left_beam)^2 + abs(metrics.eta_right_beam)^2);

    out = struct( ...
        'Sxx', Sxx, ...
        'alpha_beam', alpha_beam, ...
        'eta_left_beam', metrics.eta_left_beam, ...
        'eta_right_beam', metrics.eta_right_beam, ...
        'thrust_N', args.thrust / args.d, ...
        'tail_flat_ratio', tail_ratio, ...
        'success', true);
end
