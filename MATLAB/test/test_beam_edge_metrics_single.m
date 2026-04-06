function test_beam_edge_metrics_single
%TEST_BEAM_EDGE_METRICS_SINGLE Validate beam-end eta extraction.

addpath('../src');
addpath('.');

[~, x, ~, ~, eta, args] = flexible_surferbot_v2( ...
    'sigma',72.2e-3, 'rho',1000, 'omega',2*pi*80, 'nu',0, 'g',9.81, ...
    'L_raft',0.05, 'motor_position',0.24*0.05/2, 'd',0.03, ...
    'EI',3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.052, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative');

metrics = extract_eta_edge_metrics(eta, args);
left_idx = find(args.x_contact, 1, 'first');
right_idx = find(args.x_contact, 1, 'last');

assert(metrics.eta_left_domain == eta(1));
assert(metrics.eta_right_domain == eta(end));
assert(metrics.left_beam_idx == left_idx);
assert(metrics.right_beam_idx == right_idx);
assert(metrics.eta_left_beam == eta(left_idx));
assert(metrics.eta_right_beam == eta(right_idx));
assert(left_idx > 1);
assert(right_idx < numel(eta));
assert(x(left_idx) >= -args.L_raft / 2 - 10 * eps(args.L_raft));
assert(x(right_idx) <= args.L_raft / 2 + 10 * eps(args.L_raft));

fprintf('beam-edge metrics test passed: beam indices [%d, %d], domain size %d\\n', ...
    left_idx, right_idx, numel(eta));
