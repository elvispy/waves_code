function thrust_vs_motor_inertia_test
% Goal: η should scale linearly with motor inertia J.
% Checks proportionality η(Ji) ≈ αi·η(Jref) and αi ≈ Ji/Jref.

addpath('../src');

% --- base setup ---
L_raft = 0.05;
base = struct( ...
    'sigma',0, 'rho',100, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.5*L_raft/2, 'd',L_raft/2, ...
    'EI',100*3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*3.0, ...
    'domainDepth',0.2, 'n',201, 'M',100, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*10);

% --- sweep in J ---
J_list = base.motor_inertia * [1 2 4 8];
proto = struct('x',[],'eta',[],'J',NaN,'thrust_N',NaN);
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
figure(1); clf; hold on;
for i = 1:numel(S)
    semilogy(S(i).x, abs(S(i).eta), 'DisplayName', sprintf('J=%.2e', S(i).J));
end
xlabel('x (m)'); ylabel('|η(x)|'); legend('show','Location','best'); set(gca,'FontSize',13);
title('Free-surface amplitude vs x for varying motor inertia'); set(gcf, 'Position',[100 100 800 400]);

% α(J) vs expected Ji/Jref
figure(2); clf; hold on;
plot(J_list/Jref, real(alpha), 'o-', 'DisplayName','Re(\alpha)'); 
plot(J_list/Jref, imag(alpha), 's-', 'DisplayName','Im(\alpha)');
plot(J_list/Jref, J_list/Jref, 'k--', 'DisplayName','expected = Ji/Jref');
xlabel('Ji/Jref'); ylabel('\alpha_i'); legend('show','Location','best'); grid on; set(gca,'FontSize',13);
title('\alpha vs expected scaling');

% Thrust vs J (diagnostic)
figure(3); clf;
plot(J_list, [S.thrust_N], 'o-','LineWidth',1.2);
xlabel('Motor inertia J (kg·m^2)'); ylabel('Thrust (N)'); set(gca,'FontSize',13);
title('Thrust vs motor inertia');

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
S = struct('x',x,'eta',eta,'J',p.motor_inertia,'thrust_N',args.thrust);
end
