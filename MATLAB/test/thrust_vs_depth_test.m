function S = thrust_vs_depth_test
% Domain-depth study: vary H=domainDepth, plot |eta(x)| and thrust vs H

addpath('../src');

L_raft = 0.05;
base = struct( ...
    'sigma',0, 'rho',1000, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.3*L_raft/2, 'd',L_raft/2, ...
    'EI',10, 'rho_raft',0.018*3.0, ...
    'domainDepth',0.5, 'n',101, 'M',100, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*5, 'ooa', 4);

H_list = [1 2 4];

proto = struct('x',[],'z',[],'phi',[],'eta',[], ...
               'N_x',0,'M_z',0, 'args', struct(), ...
               'thrust_N',NaN,'tail_flat_ratio',NaN,'disp_res',NaN, ...
               'H',NaN);
S = repmat(proto, 1, numel(H_list));

for i = 1:numel(H_list)
    p = base; 
    p.domainDepth = base.domainDepth * H_list(i); 
    p.M = base.M * H_list(i); 
    S(i) = run_case(p);  % now includes H
end

% --- after you compute S ---
figure(1); clf;
set(gcf,'Units','pixels','Position',[100 100 1200 500]);  % wide window

% left axes (~75% width), aligned left
ax = axes('Position',[0.08 0.12 0.67 0.80]); hold(ax,'on');
for i = 1:numel(S)
    semilogy(ax, S(i).x, abs(S(i).eta), 'DisplayName', sprintf('H=%.3g m', S(i).H));
end
xlabel(ax,'x (m)'); ylabel(ax,'|{\eta}|'); legend(ax,'show','Location','best'); set(ax,'FontSize',14);
title(ax,'Free-surface amplitude vs x for varying depth');

% right text panel (~25% width)
axlog = axes('Position',[0.78 0.12 0.20 0.80], 'Visible','off');

% ONE info string with params common to all runs (pick what you want)
A = S(1).args;
info = sprintf(['k*dx=%.3g, dz/H=%.3g\n' ...
                'omega=%.3g, L=%.3g, \n d=%.3g, EI=%.3g, \n', ...
                'Gamma=%.3g, Fr=%.3g, \n kappa=%.3g, ', ...
                'Lambda=%.3g \n L_domain = %.3g'], ...
               A.dx * A.k, A.dz/ A.domainDepth, ...
               A.omega, A.L_raft, A.d, A.EI, ...
               A.nd_groups.Gamma, A.nd_groups.Fr, A.nd_groups.kappa, ...
               A.nd_groups.Lambda, A.L_domain);

text(axlog,0,1,'Run constants','FontWeight','bold','FontSize',16, ...
     'Units','normalized','VerticalAlignment','top');
text(axlog,0,0.9,info,'Units','normalized','FontSize',18, ...
     'Interpreter','none','VerticalAlignment','top');


figure(2); clf;
plot([S.H], [S.thrust_N], 'o-','MarkerSize',6,'LineWidth',1.2);
xlabel('Depth H (m)'); ylabel('Thrust (N)'); set(gca,'FontSize',14);
title('Thrust vs depth');

T = table([S.H].', [S.thrust_N].', [S.tail_flat_ratio].', [S.disp_res].', ...
    'VariableNames', {'H_m','thrust_N','tail_flat_ratio','dispersion_resid'});
disp('=== Depth-sweep results ==='); disp(T);
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
           'disp_res',res, ...
           'args', args, ...
           'H', p.domainDepth);   % <-- add H here
end
