function analyze_predict_second_family_v2(saveDir)
% Strategy: extract EVERYTHING from the solver at points along the curve.
% For each mode n at each (EI, x_M) point:
%   q_n = -F_n / D_n   =>   D_n = -F_n / q_n
% This gives the actual complex, mode-dependent D_n.
% Then fit D_n(EI) for each symmetric mode and use it to predict the curve.

if nargin < 1, saveDir = 'data'; end
addpath('../src');

Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L_raft = St(1).args.L_raft;
base   = St(1).args;

mp_list = unique([St.motor_position]);
EI_list = unique([St.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);
MP_norm_list = mp_list / L_raft;

eta_1   = reshape([St.eta_1],   n_mp, n_EI);
eta_end = reshape([St.eta_end], n_mp, n_EI);
eta_1_sq   = abs(eta_1).^2;
eta_end_sq = abs(eta_end).^2;
asymmetry  = -(eta_1_sq - eta_end_sq) ./ (eta_1_sq + eta_end_sq);
S_data = (eta_end + eta_1) / 2;
A_data = (eta_end - eta_1) / 2;
SA_ratio = log10(abs(S_data) ./ (abs(A_data) + eps));

% ---- Extract lowest S~0 curve ----
curve_EI = [];
curve_mp = [];
for ie = 1:n_EI
    col = asymmetry(:, ie);
    cr_mp = []; cr_SA = [];
    for im = 1:(n_mp-1)
        if col(im) * col(im+1) < 0
            t = col(im) / (col(im) - col(im+1));
            cr_mp(end+1) = MP_norm_list(im) + t*(MP_norm_list(im+1)-MP_norm_list(im));
            sa_col = SA_ratio(:,ie);
            cr_SA(end+1) = sa_col(im) + t*(sa_col(im+1)-sa_col(im));
        end
    end
    if ~isempty(cr_mp)
        s0 = cr_SA < 0;
        if any(s0)
            curve_EI(end+1) = EI_list(ie);
            curve_mp(end+1) = min(cr_mp(s0));
        end
    end
end
bad = abs(curve_mp - 0.3634) < 0.01;
curve_EI(bad) = []; curve_mp(bad) = [];

% ---- Run solver at ~20 points spread across the full EI range ----
n_sample = 20;
idx = round(linspace(1, numel(curve_EI), n_sample));
sEI = curve_EI(idx);
sMP = curve_mp(idx);

n_modes = 8;
all_q = zeros(n_sample, n_modes);
all_F = zeros(n_sample, n_modes);
all_Sfar = zeros(n_sample, 1);
all_beta = zeros(n_modes, 1);

fprintf('Running %d solver cases...\n', n_sample);
for ip = 1:n_sample
    [~, x, ~, ~, eta, args] = flexible_surferbot_v2( ...
        'sigma',base.sigma,'rho',base.rho,'nu',base.nu,'g',base.g, ...
        'L_raft',base.L_raft,'motor_position',sMP(ip)*L_raft, ...
        'd',base.d,'EI',sEI(ip),'rho_raft',base.rho_raft, ...
        'domainDepth',base.domainDepth,'L_domain',base.L_domain, ...
        'motor_inertia',base.motor_inertia,'BC',base.BC,'omega',base.omega);

    Srun = struct('x',x,'eta',eta,'args',args);
    modal = decompose_raft_freefree_modes(Srun,'num_modes',n_modes,'verbose',false);

    all_q(ip,:) = modal.q.';
    all_F(ip,:) = modal.F.';
    all_Sfar(ip) = (eta(end)+eta(1))/2;

    if ip == 1
        all_beta = modal.beta;
        mode_types = modal.mode_type;
        Psi_left  = modal.Psi(1,:);
        Psi_right = modal.Psi(end,:);
        is_sym = abs(Psi_left+Psi_right) > abs(Psi_left-Psi_right);
        x_raft = modal.x_raft;
    end
end

% ---- Extract D_n = -F_n / q_n at each point ----
all_D = -all_F ./ all_q;  % n_sample x n_modes

fprintf('\n=== Extracted D_n (mode-dependent, complex) ===\n');
sym_idx = find(is_sym);
for j = sym_idx
    fprintf('\nMode %d (%s), beta=%.2f:\n', j-1, mode_types{j}, all_beta(j));
    fprintf('  %-11s | %-7s | %-22s | %-12s\n', 'EI', 'x_M/L', 'D_n', '|D_n|');
    for ip = 1:n_sample
        fprintf('  %10.3e | %6.4f | %+10.3e %+10.3ei | %10.3e\n', ...
            sEI(ip), sMP(ip), real(all_D(ip,j)), imag(all_D(ip,j)), abs(all_D(ip,j)));
    end
end

% ---- Fit D_n(EI) for each symmetric mode ----
% Model: D_n(EI) = EI * beta_n^4 + C_n  (complex C_n absorbs added mass + damping)
% Linear fit: D_n = beta_n^4 * EI + C_n

fprintf('\n=== Fitting D_n(EI) = beta_n^4 * EI + C_n ===\n');
C_fit = zeros(numel(sym_idx), 1);  % complex intercepts
slope_check = zeros(numel(sym_idx), 1);

for jj = 1:numel(sym_idx)
    j = sym_idx(jj);
    D_vec = all_D(:, j);
    EI_vec = sEI(:);
    beta4 = all_beta(j)^4;

    % Linear regression: D = beta4 * EI + C
    % C = D - beta4 * EI (should be constant if model is correct)
    C_vals = D_vec - beta4 * EI_vec;

    C_fit(jj) = mean(C_vals);
    C_std = std(C_vals);

    fprintf('  Mode %d (%s): beta^4=%.2e\n', j-1, mode_types{j}, beta4);
    fprintf('    C = %+.4e %+.4ei  (std: %.3e, CoV: %.1f%%)\n', ...
        real(C_fit(jj)), imag(C_fit(jj)), abs(C_std), 100*abs(C_std)/abs(C_fit(jj)));

    slope_check(jj) = beta4;
end

% ---- Fit radiation coefficients a_n (same as v1) ----
Q_sym = all_q(:, sym_idx);
a_fit = Q_sym \ all_Sfar;

S_check = Q_sym * a_fit;
resid = mean(abs(all_Sfar - S_check) ./ abs(all_Sfar));
fprintf('\nRadiation coefficients (fit residual: %.2e):\n', resid);
for jj = 1:numel(sym_idx)
    fprintf('  a_%d = %+.4e %+.4ei  (|a|=%.4e)\n', ...
        sym_idx(jj)-1, real(a_fit(jj)), imag(a_fit(jj)), abs(a_fit(jj)));
end

% ---- Predict curve: f(x_M, EI) = sum a_n * W_n(x_M) / D_n(EI) = 0 ----
% D_n(EI) = beta_n^4 * EI + C_n  (complex, mode-dependent)

xM_scan = linspace(0.01, max(x_raft)*0.95, 500);
W_scan = zeros(numel(sym_idx), numel(xM_scan));
for jj = 1:numel(sym_idx)
    j = sym_idx(jj);
    Psi_ref_run = [];
    % Re-run one reference to get mode shapes on this grid
    if jj == 1
        [~,x0,~,~,eta0,args0] = flexible_surferbot_v2( ...
            'sigma',base.sigma,'rho',base.rho,'nu',base.nu,'g',base.g, ...
            'L_raft',base.L_raft,'motor_position',0.15*base.L_raft, ...
            'd',base.d,'EI',1e-3,'rho_raft',base.rho_raft, ...
            'domainDepth',base.domainDepth,'L_domain',base.L_domain, ...
            'motor_inertia',base.motor_inertia,'BC',base.BC,'omega',base.omega);
        Sref = struct('x',x0,'eta',eta0,'args',args0);
        mref = decompose_raft_freefree_modes(Sref,'num_modes',n_modes,'verbose',false);
        Psi_ref = mref.Psi;
        x_raft_ref = mref.x_raft;
    end
    W_scan(jj,:) = interp1(x_raft_ref, Psi_ref(:,j), xM_scan, 'linear', 0);
end

EI_pred = logspace(log10(min(EI_list)), log10(max(EI_list)), 500);
xM_pred = NaN(size(EI_pred));

for ie = 1:numel(EI_pred)
    EI_val = EI_pred(ie);

    % Mode-dependent complex D_n
    D_sym = zeros(numel(sym_idx), 1);
    for jj = 1:numel(sym_idx)
        D_sym(jj) = slope_check(jj) * EI_val + C_fit(jj);
    end

    if any(abs(D_sym) < 1e-3 * max(abs(D_sym))), continue; end

    coeffs = a_fit ./ D_sym;
    f_scan = coeffs.' * W_scan;  % complex

    % Factor out common phase and find zero of the rotated real part
    phase = angle(sum(a_fit));
    f_rot = real(f_scan * exp(-1i*phase));

    for ix = 1:(numel(xM_scan)-1)
        if f_rot(ix) * f_rot(ix+1) < 0
            t = f_rot(ix) / (f_rot(ix) - f_rot(ix+1));
            xM_pred(ie) = xM_scan(ix) + t*(xM_scan(ix+1)-xM_scan(ix));
            break;
        end
    end
end

% ---- Plot ----
fig = figure('Color','w','Units','centimeters','Position',[2 2 24 16]);
ax = gca; hold(ax,'on');

contourf(ax, reshape([St.EI],n_mp,n_EI), ...
    reshape([St.motor_position],n_mp,n_EI)/L_raft, ...
    asymmetry, 50, 'LineStyle','none');
contour(ax, reshape([St.EI],n_mp,n_EI), ...
    reshape([St.motor_position],n_mp,n_EI)/L_raft, ...
    asymmetry, [0 0], 'LineColor','k','LineWidth',1.5);
caxis([-1 1]); colormap(ax, bwr_colormap()); colorbar;

plot(ax, curve_EI, curve_mp, 'ko', 'MarkerSize', 5, 'LineWidth', 1, ...
    'DisplayName', 'Data');

valid = ~isnan(xM_pred);
plot(ax, EI_pred(valid), xM_pred(valid), 'r-', 'LineWidth', 2.5, ...
    'DisplayName', 'f(x_M,EI)=0 (mode-dep D_n)');

set(ax,'XScale','log');
xlabel('EI (N m^4)'); ylabel('x_M / L');
title('Prediction with mode-dependent D_n(EI)');
legend('show','Location','northeast','FontSize',11);
set(ax,'FontSize',12);
xlim([min(EI_list) max(EI_list)]); ylim([0 0.5]);

outfile = fullfile(saveDir, 'analyze_predict_second_family_v2.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[24 16],'PaperPosition',[0 0 24 16]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);

end

function cmap = bwr_colormap(n_colors, gamma)
    if nargin < 1, n_colors = 256; end
    if nargin < 2, gamma = 1.3; end
    rgb_anchors = [0.99 0.35 0.00; 1 1 1; 0 0.35 0.80];
    lab_anchors = rgb2lab(rgb_anchors);
    t_lin = linspace(-1,1,n_colors).';
    t_nl = sign(t_lin) .* (abs(t_lin).^gamma);
    lab_interp = interp1([-1 0 1].', lab_anchors, t_nl, 'linear');
    cmap = max(min(lab2rgb(lab_interp),1),0);
end
