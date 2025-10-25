function thrust_vs_motor_inertia_test
% Goal: η should scale linearly with motor inertia J.
% Checks proportionality η(Ji) ≈ αi·η(Jref) and αi ≈ Ji/Jref.

addpath('../src');

% --- base setup ---
L_raft = 0.05;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.4*L_raft/2, 'd',L_raft/2, ...
    'EI',100*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*10.0, ...
    'domainDepth',0.5, 'L_domain', 3*L_raft, 'n',101, 'M',200, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*2, 'ooa', 4);


% --- sweep in J ---
J_list = base.motor_inertia * [1/4 1/2 1 2 4 8 16];
proto = struct('x',[],'eta',[],'J',NaN,'thrust_N',NaN, 'args', struct(), 'LH', nan);
S = repmat(proto,1,numel(J_list));
for i = 1:numel(J_list)
    p = base; p.motor_inertia = J_list(i);
    S(i) = run_case_eta_only(p);
end

% --- proportionality check against reference (Jref = baseline 1.0x) ---
[~, iref] = min(abs(J_list - base.motor_inertia)); % pick closest to 1.0x
eta_ref = S(iref).eta;
Jref    = S(iref).J;

alpha   = zeros(1,numel(S));        % best-fit scale factors
propErr = zeros(1,numel(S));        % ||ηi - αi ηref|| / ||ηi||
scaleErr= zeros(1,numel(S));        % |αi - Ji/Jref| / |Ji/Jref|
for i = 1:numel(S)
    % complex LS scale (conjugate inner product)
    num   = eta_ref' * S(i).eta;
    denom = eta_ref' * eta_ref;
    alpha(i) = num / denom;
    propErr(i)= norm(S(i).eta - alpha(i)*eta_ref) / max(eps, norm(S(i).eta));
    scaleExp  = S(i).J / Jref;
    scaleErr(i)= abs(alpha(i) - scaleExp) / max(eps, abs(scaleExp));
end

% --- plots ---
% |η(x)| for each J
A = S(1).args;

figure(1); clf;
set(gcf,'Units','pixels','Position',[80 80 1200 500]);  % wide window

% left plot (~75% width)
ax = axes('Position',[0.07 0.12 0.68 0.80]); hold(ax,'on');
for i = 1:numel(S)
    semilogy(ax, S(i).x, abs(S(i).eta), 'DisplayName', sprintf('J=%.2e', S(i).J));
end
xlabel(ax,'x (m)'); ylabel(ax,'|{\eta}(x)|');
legend(ax,'show','Location','best'); grid(ax,'on');
set(ax,'FontSize',13);
title(ax,'Free-surface amplitude vs x for varying motor inertia');

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

% α(J) vs expected Ji/Jref
figure(2); clf; hold on;
plot(J_list/Jref, real(alpha), 'o-', 'DisplayName','Re(\alpha)'); 
plot(J_list/Jref, imag(alpha), 's-', 'DisplayName','Im(\alpha)');
plot(J_list/Jref, J_list/Jref, 'k--', 'DisplayName','expected = Ji/Jref');
xlabel('Ji/Jref'); ylabel('\alpha_i'); legend('show','Location','best'); grid on; set(gca,'FontSize',13);
title('\alpha vs expected scaling');

% Thrust vs J (diagnostic)
figure(3); clf;
plot(J_list, [S.thrust_N], 'o-','LineWidth',1.2, 'DisplayName', 'FT'); hold on;
plot(J_list, [S.LH],       'x-','LineWidth',1.8, 'DisplayName', 'LH'); 
xlabel('Motor inertia J (kg * m^2)'); ylabel('Thrust (N)'); set(gca,'FontSize',13);
title('Thrust vs motor inertia'); legend('show','Location','best');

% --- console summary ---
T = table(J_list.', alpha.', propErr.', scaleErr.', [S.thrust_N].', ...
    'VariableNames', {'J','alpha','propErr','scaleErr','thrust_N'});
disp('=== η ~ J proportionality check ==='); disp(T);
fprintf('Max proportionality error: %.3e\n', max(propErr));
fprintf('Max scaling error vs Ji/Jref: %.3e\n', max(scaleErr));
fprintf('Suggested tolerances: propErr < 1e-2, scaleErr < 5e-2\n');

end

% ---- helpers ----
function S = run_case_eta_only(p)
[~, x, ~, ~, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'n',p.n,'M',p.M,'motor_inertia',p.motor_inertia,'BC',p.BC);

S = struct('x',x,'eta',eta,'J',p.motor_inertia, ...
           'thrust_N',args.thrust,'args',args, ...
           'LH', 1/4 * p.rho * p.omega^2 / args.k * (abs(eta(1))^2 - abs(eta(end))^2));
end

