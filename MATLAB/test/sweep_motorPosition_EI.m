function S = sweep_motorPosition_EI
% Sweep motor_position and EI. Retry runs with higher resolution if tail_flat_ratio > 0.05.

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
% positions as a fraction of half-raft length (meters)
motor_position_list = (0.02:0.02:0.48) * L_raft;
EI_list             = base.EI * 10.^linspace(-3, 1, 57);   % same EI grid

% --- storage (2D: motor_position x EI)
proto = struct('thrust_N',NaN,'N_x',NaN,'M_z',NaN, ...
               'tail_flat_ratio',NaN,'args',struct(), ...
               'EI',NaN,'motor_position',NaN,'Sxx',NaN, ...
               'eta_edge_ratio',NaN,'n_used',NaN,'M_used',NaN, ...
               'eta_1',NaN,'eta_end',NaN, ...
               'success',false,'retries',0);
S( numel(motor_position_list), numel(EI_list) ) = proto; %#ok<AGROW>

% --- control
tail_thresh = 0.05;
max_retries = 3;
growth      = 1.10;

for ip = 1:numel(motor_position_list)
    for ie = 1:numel(EI_list)
        p = base;
        p.motor_position = motor_position_list(ip);
        p.EI             = EI_list(ie);

        [R, meta] = run_case_with_retry(p, tail_thresh, max_retries, growth);

        S(ip,ie).thrust_N        = R.thrust_N;
        S(ip,ie).N_x             = R.N_x;
        S(ip,ie).M_z             = R.M_z;
        S(ip,ie).tail_flat_ratio = R.tail_flat_ratio;
        S(ip,ie).args            = R.args;
        S(ip,ie).eta_1           = R.eta_1;
        S(ip,ie).eta_end         = R.eta_end;
        S(ip,ie).EI              = p.EI;
        S(ip,ie).motor_position  = p.motor_position;
        S(ip,ie).Sxx             = R.Sxx;
        S(ip,ie).eta_edge_ratio  = R.eta_edge_ratio;
        S(ip,ie).n_used          = meta.n_used;
        S(ip,ie).M_used          = meta.M_used;
        S(ip,ie).success         = meta.success;
        S(ip,ie).retries         = meta.retries;
    end
end

% --- console table summary
EI_col     = arrayfun(@(e) e.EI, S(:));
mp_col     = arrayfun(@(e) e.motor_position, S(:));
ok_col     = arrayfun(@(e) e.success, S(:));
T = table(EI_col, mp_col, [S(:).N_x].', [S(:).M_z].', ...
          [S(:).thrust_N].', [S(:).Sxx].', [S(:).eta_edge_ratio].', ...
          [S(:).tail_flat_ratio].', [S(:).n_used].', [S(:).M_used].', ok_col, ...
    'VariableNames', {'EI','motor_position','N','M','thrust_N','Sxx','eta_edge_ratio','tail_flat_ratio','n_used','M_used','success'});
disp('=== motor_position x EI sweep results ==='); disp(T);

% --- plot 1: log10 eta-edge ratio
figure;
surf(reshape([S.EI], [numel(motor_position_list) numel(EI_list)]), ...
     reshape([S.motor_position], [numel(motor_position_list) numel(EI_list)]), ...
     reshape(log10([S.eta_edge_ratio]), [numel(motor_position_list) numel(EI_list)]), ...
     'DisplayName', 'Log of ratio of tails');
xlabel('EI'); ylabel('motor\_position [m]'); zlabel('eta(1)/eta(end)');
set(gca, 'XScale', 'log'); set(gca, 'FontSize', 16);
caxis([-1 1]); colorbar; hold on;

idxSurferbot = find([S.EI] >= base.EI & [S.motor_position] == base.motor_position, 1);
if ~isempty(idxSurferbot)
    scatter3(S(idxSurferbot).EI, S(idxSurferbot).motor_position, ...
        log10(S(idxSurferbot).eta_edge_ratio), 100, ...
        'r', 'filled', 'DisplayName', 'Surferbot');
    legend('show', 'Location', 'south');
end

% --- plot 2: asymmetry factor
figure;
allargs = [S.args]; allnd_groups = [allargs.nd_groups]; %#ok<NASGU>
asymmetry_factor = (abs([S.eta_1]).^2 - abs([S.eta_end]).^2) ./ ...
                   (abs([S.eta_1]).^2 + abs([S.eta_end]).^2);
surf(reshape([S.EI], [numel(motor_position_list) numel(EI_list)]), ...
     reshape([S.motor_position], [numel(motor_position_list) numel(EI_list)]), ...
     reshape(asymmetry_factor, [numel(motor_position_list) numel(EI_list)]), ...
     'DisplayName', 'Asymmetry factor');
xlabel('EI'); ylabel('motor\_position [m]'); zlabel('Asymmetry factor');
set(gca, 'XScale', 'log'); set(gca, 'FontSize', 16);
caxis([-1 1]); colorbar; hold on;

if ~isempty(idxSurferbot)
    scatter3(S(idxSurferbot).EI, S(idxSurferbot).motor_position, ...
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