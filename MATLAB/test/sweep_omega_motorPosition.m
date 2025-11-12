function S = omega_motorPosition_sweep
% Sweep omega and motor_position. Retry runs with higher resolution if tail_flat_ratio > 0.05.

addpath('../src');

% --- base parameters
L_raft = 0.05;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0*1e-6, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.24*L_raft/2, 'd',0.03, ...
    'EI',3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.052, ...
    'domainDepth',nan, 'L_domain', nan, 'n',nan, 'M',nan, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*80, 'ooa', 4);

% --- sweep lists (edit as needed)
omega_list           = 2*pi*(4:4:100);                 % rad/s
motor_position_list  = (0.02:0.02:0.48) * (L_raft/2);  % meters

% --- storage (2D: omega x motor_position)
proto = struct('thrust_N',NaN,'N_x',NaN,'M_z',NaN, ...
               'tail_flat_ratio',NaN,'args',struct(), ...
               'EI',NaN,'omega',NaN,'motor_position',NaN,'Sxx',NaN, ...
               'eta_edge_ratio',NaN,'n_used',NaN,'M_used',NaN, ...
               'eta_1',NaN,'eta_end',NaN, ...
               'success',false,'retries',0);
S( numel(omega_list), numel(motor_position_list) ) = proto; %#ok<AGROW>

% --- control
tail_thresh = 0.05;
max_retries = 3;
growth      = 1.10;

for iw = 1:numel(omega_list)
    for ip = 1:numel(motor_position_list)
        p = base;
        p.omega           = omega_list(iw);
        p.motor_position  = motor_position_list(ip);
        % EI fixed to base.EI

        [R, meta] = run_case_with_retry(p, tail_thresh, max_retries, growth);

        S(iw,ip).thrust_N        = R.thrust_N;
        S(iw,ip).N_x             = R.N_x;
        S(iw,ip).M_z             = R.M_z;
        S(iw,ip).tail_flat_ratio = R.tail_flat_ratio;
        S(iw,ip).args            = R.args;
        S(iw,ip).eta_1           = R.eta_1;
        S(iw,ip).eta_end         = R.eta_end;
        S(iw,ip).EI              = p.EI;
        S(iw,ip).omega           = p.omega;
        S(iw,ip).motor_position  = p.motor_position;
        S(iw,ip).Sxx             = R.Sxx;
        S(iw,ip).eta_edge_ratio  = R.eta_edge_ratio;
        S(iw,ip).n_used          = meta.n_used;
        S(iw,ip).M_used          = meta.M_used;
        S(iw,ip).success         = meta.success;
        S(iw,ip).retries         = meta.retries;
    end
end

% --- console table summary
omega_col = arrayfun(@(e) e.omega, S(:));
mp_col    = arrayfun(@(e) e.motor_position, S(:));
ok_col    = arrayfun(@(e) e.success, S(:));
T = table(omega_col, mp_col, [S(:).N_x].', [S(:).M_z].', ...
          [S(:).thrust_N].', [S(:).Sxx].', [S(:).eta_edge_ratio].', ...
          [S(:).tail_flat_ratio].', [S(:).n_used].', [S(:).M_used].', ok_col, ...
    'VariableNames', {'omega_rad_s','motor_position_m','N','M','thrust_N','Sxx','eta_edge_ratio','tail_flat_ratio','n_used','M_used','success'});
disp('=== omega x motor_position sweep results ==='); disp(T);

% --- plot 1: log10 eta-edge ratio
figure;
surf(reshape([S.omega], [numel(omega_list) numel(motor_position_list)])/(2*pi), ...
     reshape([S.motor_position], [numel(omega_list) numel(motor_position_list)]), ...
     reshape(log10([S.eta_edge_ratio]), [numel(omega_list) numel(motor_position_list)]), ...
     'DisplayName', 'Log of ratio of tails');
xlabel('Omega [Hz]'); ylabel('motor\_position [m]'); zlabel('eta(1)/eta(end)');
set(gca, 'FontSize', 16);
caxis([-1 1]); colorbar; hold on;

idxSurferbot = find([S.omega] == base.omega & [S.motor_position] == base.motor_position, 1);
if ~isempty(idxSurferbot)
    scatter3(S(idxSurferbot).omega/(2*pi), S(idxSurferbot).motor_position, ...
        log10(S(idxSurferbot).eta_edge_ratio), 100, ...
        'r', 'filled', 'DisplayName', 'Surferbot');
    legend('show', 'Location', 'south');
end

% --- plot 2: asymmetry factor
figure;
asymmetry_factor = (abs([S.eta_1]).^2 - abs([S.eta_end]).^2) ./ ...
                   (abs([S.eta_1]).^2 + abs([S.eta_end]).^2);
surf(reshape([S.omega], [numel(omega_list) numel(motor_position_list)])/(2*pi), ...
     reshape([S.motor_position], [numel(omega_list) numel(motor_position_list)]), ...
     reshape(asymmetry_factor, [numel(omega_list) numel(motor_position_list)]), ...
     'DisplayName', 'Asymmetry factor');
xlabel('Omega [Hz]'); ylabel('motor\_position [m]'); zlabel('Asymmetry factor');
set(gca, 'FontSize', 16);
caxis([-1 1]); colorbar; hold on;

if ~isempty(idxSurferbot)
    scatter3(S(idxSurferbot).omega/(2*pi), S(idxSurferbot).motor_position, ...
        asymmetry_factor(idxSurferbot), 100, ...
        'r', 'filled', 'DisplayName', 'Surferbot');
    legend('show', 'Location', 'south');
end

S = struct2table(S(:));

end

% ================= helper =================
function [R, meta] = run_case_with_retry(p, tail_thresh, max_retries, growth)
    retries = 0;
    success = false;

    while true
        R = run_once(p);

        if R.tail_flat_ratio <= tail_thresh
            success = true;
            break
        end

        if retries >= max_retries
            break
        end

        % increase resolution by 10% and retry
        n = max(R.args.n+1, ceil(growth * R.args.n));
        M = max(R.args.M+1, ceil(growth * R.args.M));
        p.n = n; p.M = M;
        retries = retries + 1;
    end

    meta = struct('n_used', R.args.n, 'M_used', R.args.M, ...
                  'success', success, 'retries', retries);
end

function S = run_once(p)
    [~, ~, ~, ~, eta, args] = flexible_surferbot_v2( ...
        'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
        'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
        'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
        'L_domain', p.L_domain, 'n',p.n,'M',p.M, ...
        'motor_inertia',p.motor_inertia,'BC',p.BC);

    % tail flatness on first 5% of points near the left boundary
    idx0 = max(1, ceil(0.05*numel(eta)));
    tail = abs(eta(1:idx0));
    tail_ratio = std(tail) / max(eps, mean(tail));

    % LH proxy and edge ratio
    Sxx = (args.rho*args.g/4 + 3/4*args.sigma*args.k^2) * ...
          (abs(eta(1))^2 - abs(eta(end))^2);
    eta_edge_ratio = abs(eta(1)) / max(eps, abs(eta(end)));

    args = rmfield(args, 'phi_z');
    S = struct( ...
        'N_x',args.N,'M_z',args.M, ...
        'thrust_N',args.thrust/args.d, ...
        'tail_flat_ratio',tail_ratio, ...
        'eta_1', eta(1), 'eta_end', eta(end), ...
        'args', args, ...
        'Sxx', Sxx, ...
        'eta_edge_ratio', eta_edge_ratio);
end