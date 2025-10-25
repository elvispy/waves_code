function thrust_vs_n_test
% n-resolution study: vary N, plot |eta(x)| per case and thrust vs N

addpath('../src');
close all;
L_raft = 0.5;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0*1e-6, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.4*L_raft/2, 'd',L_raft/2, ...
    'EI', 100*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*10.0, ...
    'domainDepth',0.5,'L_domain', 3*L_raft, 'n',500, 'M',200, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*7, 'ooa', 4);

% Ns to test (ensure odd, increasing)
n_list = ensure_odd(ceil(base.n * [1, 2, 4]));
n_list = unique(n_list,'stable');

% Preallocate with matching fields
% Preallocate with matching fields
proto = struct('x',[],'z',[],'phi',[],'eta',[], ...
               'N_x',0,'M_z',0, ...
               'thrust_N',NaN,'tail_flat_ratio',NaN,'disp_res',NaN, ...
               'args', struct());   % <-- add this
S = repmat(proto, 1, numel(n_list));

% Run sweep
for ii = 1:numel(n_list)
    p = base; p.n = n_list(ii); %p.motor_inertia = p.motor_inertia * 2^(ii-1);
    S(ii) = run_case(p);
end

% Plot |eta(x)| per case
% --- after S is filled ---
A = S(1).args;  % constants

figure(1); clf;
set(gcf,'Units','pixels','Position',[1 1 1200 500]);  % wide

% left plot (~75% width)
ax = axes('Position',[0.07 0.12 0.68 0.80]); hold(ax,'on');
for i = 1:numel(S)
    
    semilogy(ax, S(i).x, abs(S(i).eta), 'o-', 'MarkerSize', 4, ...
        'DisplayName', sprintf('N=%d | k*dx=%.2g', S(i).N_x, S(i).args.k*S(i).args.dx));
end


%xline(-L_raft/2); xline(L_raft/2);
xlabel(ax,'x (m)'); ylabel(ax,'|{\eta}(x)|');
legend(ax,'show','Location','best'); grid(ax,'on'); set(ax,'FontSize',14);
title(ax,'Free-surface amplitude vs x for varying N');

% right log panel (single textbox)
info = sprintf([ ...
    '(k*dx)^-1 = %.3g\n' ...
    'H/dz      = %.3g\n' ...
    'omega     = %.3g 1/s\n' ...
    'L         = %.3g m\n' ...
    'L_{domain}= %.3g m\n' ...
    'd         = %.3g m\n' ...
    'EI        = %.3g\n' ...
    'Gamma     = %.3g\n' ...
    'Fr        = %.3g\n' ...
    'kappa     = %.3g\n' ...
    'Lambda    = %.3g\n' ...
    'Weber     = %.3g \n' ...
    'Reynolds  = %.3g'], ...
    2*pi/(A.dx*A.k), A.domainDepth/A.dz, ...
    A.omega, A.L_raft, A.L_domain, A.d, A.EI, ...
    A.nd_groups.Gamma, A.nd_groups.Fr, A.nd_groups.kappa, ...
    A.nd_groups.Lambda, A.nd_groups.We, A.nd_groups.Re);


annotation('textbox',[0.78 0.12 0.20 0.80], ...
    'String', sprintf('Run constants\n\n%s', info), ...
    'Interpreter','none','EdgeColor','none', ...
    'VerticalAlignment','top','FontSize',16,'FontName','FixedWidth');

% Plot thrust vs N
figure(2); clf; set(gcf,'Position',[1284 52 560 420]);
plot([S.N_x], [S.thrust_N], 'o-','MarkerSize',6,'LineWidth',1.2);
xlabel('N (streamwise points)'); ylabel('Thrust (N)');
set(gca,'FontSize',14);
title('Thrust vs N');

% Plot successive ratio |eta^{(i+1)}| / |eta^{(i)}|
figure(3); clf; set(gcf,'Position',[1278 557 560 420]);
ax = axes('Position',[0.1 0.12 0.8 0.8]); hold(ax,'on');




for i = 1:numel(S)-1
    x0 = S(i).x;
    x1 = S(i+1).x;
    eta0 = abs(S(i).eta(:));
    eta1 = abs(S(i+1).eta(:));

    % Interpolate to coarser grid
    if ~isequal(x0, x1)
        eta1_interp = interp1(x1(:), eta1, x0(:), 'linear', NaN);
    else
        eta1_interp = eta1;
    end

    % Ratio
    ratio = eta1_interp ./ eta0;

    % Plot, using 'line' to avoid NaN-based segmentation and legend explosion
    h = plot(x0, ratio, 'LineWidth', 1.5);
    h.DisplayName = sprintf('# %d / # %d', S(i+1).N_x, S(i).N_x);
end

xlabel('x (m)'); ylabel('|?^{(i+1)}(x)| / |?^{(i)}(x)|');
title('Convergence check: ratio of successive free-surface amplitudes');
legend('show','Location','best'); grid on; set(gca,'FontSize',14);



% --- Pressure profiles along raft for all N ---
figure(4); clf;
set(gcf,'Units','pixels','Position',[1 571 1000 400]);

subplot(1,2,1); hold on;
title('Real(p) along raft');
xlabel('x (m)'); ylabel('Re(p) (Pa)');
for i = 1:numel(S)
    plot(S(i).x(S(i).args.x_contact), real(S(i).args.pressure), 'DisplayName', ...
        sprintf('N=%d', S(i).N_x));
end
legend('show','Location','best'); grid on; set(gca,'FontSize',14);

subplot(1,2,2); hold on;
title('Imag(p) along raft');
xlabel('x (m)'); ylabel('Im(p) (Pa)');
for i = 1:numel(S)
    plot(S(i).x(S(i).args.x_contact), imag(S(i).args.pressure), 'DisplayName', ...
        sprintf('N=%d', S(i).N_x));
end
legend('show','Location','best'); grid on; set(gca,'FontSize',14);

sgtitle('Pressure along raft vs resolution (N)');


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
           'args', args, ...
           'disp_res',res);
end
