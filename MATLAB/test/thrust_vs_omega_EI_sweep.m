function S = thrust_vs_omega_EI_sweep
% Sweep omega and EI. Retry runs with higher resolution if tail_flat_ratio > 0.05.

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
omega_list = 2*pi*(4:4:100);        % rad/s
EI_list    = base.EI * 10.^linspace(-3, 1, 51);   % simple multiples

% --- storage (2D: omega x EI)
proto = struct('thrust_N',NaN,'N_x',NaN,'M_z',NaN, ...
               'tail_flat_ratio',NaN,'args',struct(), ...
               'EI',NaN,'omega',NaN,'Sxx',NaN, ...
               'eta_edge_ratio',NaN,'n_used',NaN,'M_used',NaN, ...
               'success',false,'retries',0);
S( numel(omega_list), numel(EI_list) ) = proto; %#ok<AGROW>

% --- control
tail_thresh = 0.05;
max_retries = 3;
growth      = 1.10;

for iw = 1:numel(omega_list)
    for ie = 1:numel(EI_list)
        p = base;
        p.omega = omega_list(iw);
        p.EI    = EI_list(ie);

        [R, meta] = run_case_with_retry(p, tail_thresh, max_retries, growth);

        S(iw,ie).thrust_N        = R.thrust_N;
        S(iw,ie).N_x             = R.N_x;
        S(iw,ie).M_z             = R.M_z;
        S(iw,ie).tail_flat_ratio = R.tail_flat_ratio;
        S(iw,ie).args            = R.args;
        S(iw,ie).EI              = p.EI;
        S(iw,ie).omega           = p.omega;
        S(iw,ie).Sxx             = R.Sxx;
        S(iw,ie).eta_edge_ratio  = R.eta_edge_ratio;
        S(iw,ie).n_used          = meta.n_used;
        S(iw,ie).M_used          = meta.M_used;
        S(iw,ie).success         = meta.success;
        S(iw,ie).retries         = meta.retries;
    end
end

% --- console table summary
EI_col     = arrayfun(@(e) e.EI,    S(:));
omega_col  = arrayfun(@(e) e.omega, S(:));
ok_col     = arrayfun(@(e) e.success, S(:));
T = table(EI_col, omega_col, [S(:).N_x].', [S(:).M_z].', ...
          [S(:).thrust_N].', [S(:).Sxx].', [S(:).eta_edge_ratio].', ...
          [S(:).tail_flat_ratio].', [S(:).n_used].', [S(:).M_used].', ok_col, ...
    'VariableNames', {'EI','omega','N','M','thrust_N','Sxx','eta_edge_ratio','tail_flat_ratio','n_used','M_used','success'});
disp('=== omega x EI sweep results ==='); disp(T);


surf(reshape([S.EI], [numel(omega_list) numel(EI_list)]), ...
    reshape([S.omega], [numel(omega_list) numel(EI_list)])/(2*pi), ...
    reshape(log10([S.eta_edge_ratio]), [numel(omega_list) numel(EI_list)]), 'DisplayName', 'Log of ratio of tails'); 
xlabel('EI'); ylabel('Omega'); zlabel('eta(1)/eta(end)'); 
set(gca, 'XScale', 'log'); set(gca, 'FontSize', 16); 
caxis([-1 1]); colorbar; hold on; 

idxSurferbot = find([S.EI] == base.EI & [S.omega] == base.omega);

scatter3(S(idxSurferbot).EI, S(idxSurferbot).omega/(2*pi), ...
    log10(S(idxSurferbot).eta_edge_ratio), 100, ...
    'r', 'filled', 'DisplayName', 'Surferbot'); 
legend('show', 'Location', 'south')


figure;
allargs = [S.args]; allndgroups = [allargs.ndgroups]; allkappa = [allndgroups.kappa];
asymmetry_factor = ([S.eta_1].^2 - [S.eta_end].^2) ./ ([S.eta_1].^2 + [S.eta_end].^2);
surf(reshape(allkappa, [numel(omega_list) numel(EI_list)]), ...
    reshape([S.omega], [numel(omega_list) numel(EI_list)])/(2*pi), ...
    reshape(asymmetry_factor, [numel(omega_list) numel(EI_list)]), 'DisplayName', 'Asymmetry factor'); 
xlabel('kappa'); ylabel('Omega'); zlabel('eta(1)^2 - eta(end)^2 / eta(1)^2 + eta(end)^2'); 
set(gca, 'XScale', 'log'); set(gca, 'FontSize', 16); 
caxis([-1 1]); colorbar; hold on; 

idxSurferbot = find([S.EI] == base.EI & [S.omega] == base.omega);

scatter3(allkappa(idxSurferbot), S(idxSurferbot).omega/(2*pi), ...
    asymmetry_factor(idxSurferbot), 100, ...
    'r', 'filled', 'DisplayName', 'Surferbot'); 
legend('show', 'Location', 'south')

S = struct2table(S(:));

end

% ================= helper =================
function [R, meta] = run_case_with_retry(p, tail_thresh, max_retries, growth)
    %n = p.n; M = p.M;
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
    
    meta = struct('n_used', R.args.n, 'M_used', R.args.M, 'success', success, 'retries', retries);
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
