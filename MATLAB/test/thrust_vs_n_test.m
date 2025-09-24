function thrust_vs_n_test
% n-resolution study: vary N, plot |eta(x)| per case and thrust vs N

addpath('../src');

L_raft = 0.05;
base = struct( ...
    'sigma',0, 'rho',1000, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.3*L_raft/2, 'd',L_raft/2, ...
    'EI',100*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*3.0, ...
    'domainDepth',0.2, 'n',401, 'M',50, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*10, 'ooa', 2);

% Ns to test (ensure odd, increasing)
n_list = ensure_odd(ceil(base.n * [1, 2, 4]));
n_list = unique(n_list,'stable');

% Preallocate with matching fields
proto = struct('x',[],'z',[],'phi',[],'eta',[], ...
               'N_x',0,'M_z',0, ...
               'thrust_N',NaN, 'tail_flat_ratio',NaN, 'disp_res',NaN);
S = repmat(proto, 1, numel(n_list));

% Run sweep
for i = 1:numel(n_list)
    p = base; p.n = n_list(i);
    S(i) = run_case(p);
end

% Plot |eta(x)| per case
figure(1); clf; hold on;
for i = 1:numel(S)
    semilogy(S(i).x, abs(S(i).eta), 'DisplayName', sprintf('N=%d', S(i).N_x));
end
xlabel('x (m)'); ylabel('|{\eta}|'); legend('show');
set(gca,'FontSize',14); set(gcf, 'Position',[100 100 800 400]);
title('Free-surface amplitude vs x for varying N');

% Plot thrust vs N
figure(2); clf;
plot([S.N_x], [S.thrust_N], 'o-','MarkerSize',6,'LineWidth',1.2);
xlabel('N (streamwise points)'); ylabel('Thrust (N)');
set(gca,'FontSize',14);
title('Thrust vs N');

% Console table
T = table([S.N_x].', [S.thrust_N].', [S.tail_flat_ratio].', [S.disp_res].', ...
    'VariableNames', {'N_x','thrust_N','tail_flat_ratio','dispersion_resid'});
disp('=== N-sweep results ==='); disp(T);

end

% ---- helpers ----
function n = ensure_odd(n)
n = n(:).';
for k = 1:numel(n), if mod(n(k),2)==0, n(k)=n(k)+1; end, end
end

function S = run_case(p)
[~, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'n',p.n,'M',p.M,'motor_inertia',p.motor_inertia,'BC',p.BC);

k    = real(args.k);
res  = abs(args.omega^2 - args.g*k);

dx = abs(x(2)-x(1)); dz = abs(z(2)-z(1));
[Dx, ~] = getNonCompactFDmatrix2D(args.M,args.N,dx,dz,1,args.ooa);
u  = reshape(Dx * reshape(phi, args.M*args.N,1), args.M, args.N); %#ok<NASGU>

rho = args.rho; omega = args.omega; %#ok<NASGU>
LH  = 0.25 * rho * omega^2 / k * (abs(eta(2))^2 - abs(eta(end-1))^2); %#ok<NASGU>
mom = rho * trapz(z, abs(u(:,2)).^2 - abs(u(:,end-1)).^2);            %#ok<NASGU>

idx0 = max(1, ceil(0.75*numel(eta)));
tail = abs(eta(idx0:end));
tail_ratio = std(tail) / max(eps, mean(tail));

S = struct('x',x,'z',z,'phi',phi,'eta',eta, ...
           'N_x',args.N,'M_z',args.M, ...
           'thrust_N',args.thrust, ...
           'tail_flat_ratio',tail_ratio, ...
           'disp_res',res);
end
