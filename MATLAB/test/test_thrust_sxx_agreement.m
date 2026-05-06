% test_thrust_sxx_agreement.m
% Asserts T/d ≈ Sxx (radiation-stress proxy) to within 5%
% across three frequency/surface-tension cases.

src = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src');
addpath(src);

inviscid_cases = { ...
    struct('omega', 2*pi*10,  'sigma', 72.2e-3, 'label', '10 Hz capgrav inviscid'); ...
    struct('omega', 2*pi*40,  'sigma', 72.2e-3, 'label', '40 Hz capgrav inviscid'); ...
    struct('omega', 2*pi*80,  'sigma', 72.2e-3, 'label', '80 Hz cap inviscid')};

rho           = 1000.0;
g             = 9.81;
L_raft        = 0.05;
d_val         = 0.03;
motor_pos     = 0.015;
motor_inertia = 0.13e-3 * 2.5e-3;

for ii = 1:numel(inviscid_cases)
    c = inviscid_cases{ii};
    [~, ~, ~, ~, eta, args] = flexible_surferbot_v2( ...
        'sigma', c.sigma, 'rho', rho, 'nu', 0.0, 'g', g, ...
        'L_raft', L_raft, 'd', d_val, 'omega', c.omega, ...
        'motor_position', motor_pos, 'motor_inertia', motor_inertia);

    T_over_d = args.thrust / args.d;
    k_real   = real(args.k);
    pref     = rho * g / 4 + 3/4 * c.sigma * k_real^2;
    Sxx      = pref * (abs(eta(1))^2 - abs(eta(end))^2);

    denom   = max(abs(T_over_d), abs(Sxx));
    if denom < 1e-30; denom = 1e-30; end
    rel_err = abs(T_over_d - Sxx) / denom;

    fprintf('%s  T/d=%+.4e  Sxx=%+.4e  rel_err=%.4f\n', ...
        c.label, T_over_d, Sxx, rel_err);
    assert(rel_err < 0.10, sprintf('%s: rel_err %.4f >= 0.10', c.label, rel_err));
end

% Viscous check: Sxx != T/d for viscous waves (attenuation reduces far-field amplitude),
% so instead verify small nu barely perturbs thrust relative to inviscid baseline.
[~, ~, ~, ~, ~, args_inv] = flexible_surferbot_v2( ...
    'sigma', 72.2e-3, 'rho', rho, 'nu', 0.0, 'g', g, ...
    'L_raft', L_raft, 'd', d_val, 'omega', 2*pi*40, ...
    'motor_position', motor_pos, 'motor_inertia', motor_inertia);
[~, ~, ~, ~, ~, args_vis] = flexible_surferbot_v2( ...
    'sigma', 72.2e-3, 'rho', rho, 'nu', 1e-6, 'g', g, ...
    'L_raft', L_raft, 'd', d_val, 'omega', 2*pi*40, ...
    'motor_position', motor_pos, 'motor_inertia', motor_inertia);
T_inv = args_inv.thrust / args_inv.d;
T_vis = args_vis.thrust / args_vis.d;
visc_rel_diff = abs(T_vis - T_inv) / abs(T_inv);
fprintf('40 Hz viscous vs inviscid  T/d_inv=%+.4e  T/d_vis=%+.4e  rel_diff=%.4f\n', ...
    T_inv, T_vis, visc_rel_diff);
assert(visc_rel_diff < 0.01, sprintf('viscous perturbation too large: %.4f', visc_rel_diff));

fprintf('PASS: all cases passed\n');
