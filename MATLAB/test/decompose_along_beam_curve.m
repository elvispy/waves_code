function decompose_along_beam_curve(saveDir)
%DECOMPOSE_ALONG_BEAM_CURVE Re-run the solver along the lowest beam-end
%second-family curve, then decompose the raft response into free-free modes.
%
% This is the beam-end analogue of decompose_along_curve.m. The white curve
% is extracted from the coupled x_M-EI sweep using beam-end amplitudes:
%   alpha_beam = -( |eta_left_beam|^2 - |eta_right_beam|^2 ) / ...
%                 ( |eta_left_beam|^2 + |eta_right_beam|^2 )

if nargin < 1, saveDir = 'data'; end
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'src'));

Dat = load(fullfile(saveDir, 'sweepMotorPositionEI.mat'));
St = Dat.S;
if istable(St), St = table2struct(St); end

required = {'eta_left_beam','eta_right_beam'};
for i = 1:numel(required)
    if ~isfield(St, required{i})
        error('decompose_along_beam_curve:MissingField', ...
            'Field %s is missing. Regenerate the coupled sweep with the updated sweep script.', ...
            required{i});
    end
end

L_raft = St(1).args.L_raft;
mp_list = unique([St.motor_position]);
EI_list = unique([St.EI]);
n_mp = numel(mp_list);
n_EI = numel(EI_list);
MP_norm_list = mp_list / L_raft;

eta_left = reshape([St.eta_left_beam], n_mp, n_EI);
eta_right = reshape([St.eta_right_beam], n_mp, n_EI);
eta_left_sq = abs(eta_left).^2;
eta_right_sq = abs(eta_right).^2;
asymmetry = -(eta_left_sq - eta_right_sq) ./ (eta_left_sq + eta_right_sq);
S_grid = (eta_right + eta_left) / 2;
A_grid = (eta_right - eta_left) / 2;
SA_ratio = log10(abs(S_grid) ./ (abs(A_grid) + eps));

curve_EI = [];
curve_mp = [];
for ie = 1:n_EI
    col = asymmetry(:, ie);
    crossings_mp = [];
    crossings_SA = [];
    for im = 1:(n_mp - 1)
        if col(im) * col(im + 1) < 0
            t = col(im) / (col(im) - col(im + 1));
            mp_zero = MP_norm_list(im) + t * (MP_norm_list(im + 1) - MP_norm_list(im));
            sa_col = SA_ratio(:, ie);
            sa_zero = sa_col(im) + t * (sa_col(im + 1) - sa_col(im));
            crossings_mp(end + 1) = mp_zero; %#ok<AGROW>
            crossings_SA(end + 1) = sa_zero; %#ok<AGROW>
        end
    end
    if ~isempty(crossings_mp)
        s0_mask = crossings_SA < 0;
        if any(s0_mask)
            candidates = crossings_mp(s0_mask);
            curve_EI(end + 1) = EI_list(ie); %#ok<AGROW>
            curve_mp(end + 1) = min(candidates); %#ok<AGROW>
        end
    end
end

n_sample = 12;
sample_idx = round(linspace(1, numel(curve_EI), n_sample));
sample_EI = curve_EI(sample_idx);
sample_mp = curve_mp(sample_idx);

fprintf('Sampling %d points along lowest beam-end S~0 curve\n\n', n_sample);

base = St(1).args;
n_modes = 8;

all_q = zeros(n_sample, n_modes);
all_energy = zeros(n_sample, n_modes);
all_phase = zeros(n_sample, n_modes);
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

    if ip == 1
        mode_types = modal.mode_type;
    end
end

fig = figure('Color','w','Units','centimeters','Position',[1 1 30 22]);

subplot(2,2,1); ax = gca; hold(ax,'on');
for j = 1:min(6, n_modes)
    plot(ax, log10(sample_EI), abs(all_q(:,j)), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('mode %d (%s)', j - 1, mode_types{j}));
end
set(ax,'YScale','log');
xlabel('log_{10}(EI)'); ylabel('|q_n|');
title('Modal amplitudes'); legend('show','Location','best','FontSize',8);
grid on; set(ax,'FontSize',11);

subplot(2,2,2); ax = gca; hold(ax,'on');
for j = 1:min(6, n_modes)
    plot(ax, log10(sample_EI), 100 * all_energy(:,j), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('mode %d (%s)', j - 1, mode_types{j}));
end
xlabel('log_{10}(EI)'); ylabel('Energy fraction (%)');
title('Modal energy distribution'); legend('show','Location','best','FontSize',8);
grid on; set(ax,'FontSize',11);

subplot(2,2,3); ax = gca; hold(ax,'on');
for j = 1:min(6, n_modes)
    plot(ax, log10(sample_EI), all_phase(:,j), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('mode %d (%s)', j - 1, mode_types{j}));
end
xlabel('log_{10}(EI)'); ylabel('arg(q_n) [deg]');
title('Modal phases'); legend('show','Location','best','FontSize',8);
grid on; set(ax,'FontSize',11); ylim([-180 180]);

subplot(2,2,4); ax = gca; hold(ax,'on');
mid = round(n_sample / 2);
for j = 1:min(6, n_modes)
    plot(ax, real(all_q(mid,j)), imag(all_q(mid,j)), 'o', 'MarkerSize', 10, ...
        'LineWidth', 2, 'DisplayName', sprintf('q_%d', j - 1));
    text(real(all_q(mid,j)) * 1.1, imag(all_q(mid,j)) * 1.1, sprintf('q_%d', j - 1), 'FontSize', 9);
end
xlabel('Re(q_n)'); ylabel('Im(q_n)');
title(sprintf('Complex q_n at EI=%.2e', sample_EI(mid)));
axis equal; grid on; set(ax,'FontSize',11);

sgtitle('Modal decomposition along lowest beam-end S\approx0 curve', 'FontSize',13,'FontWeight','bold');

outfile = fullfile(saveDir, 'decompose_along_beam_curve.pdf');
set(fig,'PaperUnits','centimeters','PaperSize',[30 22],'PaperPosition',[0 0 30 22]);
print(fig, outfile, '-dpdf','-painters','-r300');
fprintf('\nSaved to %s\n', outfile);
end
