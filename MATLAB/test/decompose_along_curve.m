function decompose_along_curve(saveDir)
% Re-run solver at points along the lowest S~0 curve,
% decompose into free-free modes, and look for patterns.
%
% Only ~12 points to keep it fast.

if nargin < 1, saveDir = 'data'; end
addpath('../src');

Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

L_raft = St(1).args.L_raft;
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
S_grid = (eta_end + eta_1) / 2;
A_grid = (eta_end - eta_1) / 2;
SA_ratio = log10(abs(S_grid) ./ (abs(A_grid) + eps));

% Extract lowest S~0 curve
curve_EI = [];
curve_mp = [];
for ie = 1:n_EI
    col = asymmetry(:, ie);
    crossings_mp = [];
    crossings_SA = [];
    for im = 1:(n_mp-1)
        if col(im) * col(im+1) < 0
            t = col(im) / (col(im) - col(im+1));
            mp_zero = MP_norm_list(im) + t * (MP_norm_list(im+1) - MP_norm_list(im));
            sa_col = SA_ratio(:, ie);
            sa_zero = sa_col(im) + t * (sa_col(im+1) - sa_col(im));
            crossings_mp(end+1) = mp_zero;
            crossings_SA(end+1) = sa_zero;
        end
    end
    if ~isempty(crossings_mp)
        s0_mask = crossings_SA < 0;
        if any(s0_mask)
            candidates = crossings_mp(s0_mask);
            curve_EI(end+1) = EI_list(ie);
            curve_mp(end+1) = min(candidates);
        end
    end
end

% Sample ~12 points spread across the curve
n_sample = 12;
sample_idx = round(linspace(1, numel(curve_EI), n_sample));
sample_EI = curve_EI(sample_idx);
sample_mp = curve_mp(sample_idx);

fprintf('Sampling %d points along lowest S~0 curve\n\n', n_sample);

% Base parameters
base = St(1).args;
n_modes = 8;

% Storage
all_q      = zeros(n_sample, n_modes);
all_energy = zeros(n_sample, n_modes);
all_phase  = zeros(n_sample, n_modes);
all_beta   = zeros(n_sample, n_modes);
mode_types = cell(n_modes, 1);

for ip = 1:n_sample
    EI_val = sample_EI(ip);
    mp_val = sample_mp(ip) * L_raft;

    [~, x, ~, ~, eta, args] = flexible_surferbot_v2( ...
        'sigma', base.sigma, 'rho', base.rho, 'nu', base.nu, ...
        'g', base.g, 'L_raft', base.L_raft, ...
        'motor_position', mp_val, 'd', base.d, ...
        'EI', EI_val, 'rho_raft', base.rho_raft, ...
        'domainDepth', base.domainDepth, 'L_domain', base.L_domain, ...
        'motor_inertia', base.motor_inertia, 'BC', base.BC, ...
        'omega', base.omega);

    Srun = struct('x', x, 'eta', eta, 'args', args);
    modal = decompose_raft_freefree_modes(Srun, 'num_modes', n_modes, 'verbose', false);

    all_q(ip, 1:numel(modal.q)) = modal.q.';
    all_energy(ip, 1:numel(modal.energy_frac)) = modal.energy_frac.';
    all_phase(ip, 1:numel(modal.q)) = rad2deg(angle(modal.q.'));
    all_beta(ip, 1:numel(modal.beta)) = modal.beta.';

    if ip == 1
        mode_types = modal.mode_type;
    end
end

% ---- Print modal amplitudes ----
fprintf('%-11s | %-7s', 'EI', 'x_M/L');
for j = 1:n_modes
    fprintf(' | |q%d| %-4s', j-1, mode_types{j}(1:min(3,end)));
end
fprintf('\n%s\n', repmat('-', 1, 12+10+n_modes*14));

for ip = 1:n_sample
    fprintf('%10.3e | %6.4f', sample_EI(ip), sample_mp(ip));
    for j = 1:n_modes
        fprintf(' | %10.3e', abs(all_q(ip,j)));
    end
    fprintf('\n');
end

% ---- Print energy fractions ----
fprintf('\n%-11s | %-7s', 'EI', 'x_M/L');
for j = 1:n_modes
    fprintf(' | E%d %-5s', j-1, mode_types{j}(1:min(3,end)));
end
fprintf('\n%s\n', repmat('-', 1, 12+10+n_modes*14));

for ip = 1:n_sample
    fprintf('%10.3e | %6.4f', sample_EI(ip), sample_mp(ip));
    for j = 1:n_modes
        fprintf(' | %9.1f%%', 100*all_energy(ip,j));
    end
    fprintf('\n');
end

% ---- Print phases ----
fprintf('\n%-11s | %-7s', 'EI', 'x_M/L');
for j = 1:n_modes
    fprintf(' | ph%d %-4s', j-1, mode_types{j}(1:min(3,end)));
end
fprintf('\n%s\n', repmat('-', 1, 12+10+n_modes*14));

for ip = 1:n_sample
    fprintf('%10.3e | %6.4f', sample_EI(ip), sample_mp(ip));
    for j = 1:n_modes
        fprintf(' | %+9.1f°', all_phase(ip,j));
    end
    fprintf('\n');
end

% ---- Plot ----
fig = figure('Color','w','Units','centimeters','Position',[1 1 30 22]);

% Panel 1: |q_n| along the curve
subplot(2,2,1); ax = gca; hold(ax,'on');
for j = 1:min(6, n_modes)
    plot(ax, log10(sample_EI), abs(all_q(:,j)), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('mode %d (%s)', j-1, mode_types{j}));
end
set(ax,'YScale','log');
xlabel('log_{10}(EI)'); ylabel('|q_n|');
title('Modal amplitudes'); legend('show','Location','best','FontSize',8);
grid on; set(ax,'FontSize',11);

% Panel 2: energy fractions
subplot(2,2,2); ax = gca; hold(ax,'on');
for j = 1:min(6, n_modes)
    plot(ax, log10(sample_EI), 100*all_energy(:,j), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('mode %d (%s)', j-1, mode_types{j}));
end
xlabel('log_{10}(EI)'); ylabel('Energy fraction (%)');
title('Modal energy distribution'); legend('show','Location','best','FontSize',8);
grid on; set(ax,'FontSize',11);

% Panel 3: phases
subplot(2,2,3); ax = gca; hold(ax,'on');
for j = 1:min(6, n_modes)
    plot(ax, log10(sample_EI), all_phase(:,j), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('mode %d (%s)', j-1, mode_types{j}));
end
xlabel('log_{10}(EI)'); ylabel('arg(q_n) [deg]');
title('Modal phases'); legend('show','Location','best','FontSize',8);
grid on; set(ax,'FontSize',11); ylim([-180 180]);

% Panel 4: q_n in complex plane for one mid-curve point
subplot(2,2,4); ax = gca; hold(ax,'on');
mid = round(n_sample/2);
for j = 1:min(6, n_modes)
    plot(ax, real(all_q(mid,j)), imag(all_q(mid,j)), 'o', 'MarkerSize', 10, ...
        'LineWidth', 2, 'DisplayName', sprintf('q_%d', j-1));
    text(real(all_q(mid,j))*1.1, imag(all_q(mid,j))*1.1, sprintf('q_%d',j-1), 'FontSize',9);
end
xlabel('Re(q_n)'); ylabel('Im(q_n)');
title(sprintf('Complex q_n at EI=%.2e', sample_EI(mid)));
axis equal; grid on; set(ax,'FontSize',11);

sgtitle('Modal decomposition along lowest S\approx0 curve', 'FontSize',13,'FontWeight','bold');

outfile = fullfile(saveDir, 'decompose_along_curve.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[30 22],'PaperPosition',[0 0 30 22]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);

end
