function fit_second_family_curve(saveDir)
% Extract the lowest S~0 curve from sweep data and fit empirical formulas.
% Try: x_M/L = a + b*log10(EI) + c*log10(EI)^2

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
% Remove outlier
bad = abs(curve_mp - 0.3634) < 0.01;
curve_EI(bad) = []; curve_mp(bad) = [];

logEI = log10(curve_EI(:));
mp = curve_mp(:);
n = numel(logEI);

fprintf('Curve: %d points, EI range [%.2e, %.2e], x_M/L range [%.3f, %.3f]\n', ...
    n, min(curve_EI), max(curve_EI), min(mp), max(mp));

% ---- Fit 1: linear in log10(EI) ----
X1 = [ones(n,1), logEI];
c1 = X1 \ mp;
mp_fit1 = X1 * c1;
r1 = sqrt(mean((mp - mp_fit1).^2));
fprintf('\nFit 1: x_M/L = %.4f + %.4f * log10(EI)\n', c1(1), c1(2));
fprintf('  RMSE = %.4f, R^2 = %.4f\n', r1, 1 - sum((mp-mp_fit1).^2)/sum((mp-mean(mp)).^2));

% ---- Fit 2: quadratic in log10(EI) ----
X2 = [ones(n,1), logEI, logEI.^2];
c2 = X2 \ mp;
mp_fit2 = X2 * c2;
r2 = sqrt(mean((mp - mp_fit2).^2));
fprintf('\nFit 2: x_M/L = %.4f + %.4f * log10(EI) + %.6f * log10(EI)^2\n', c2(1), c2(2), c2(3));
fprintf('  RMSE = %.4f, R^2 = %.4f\n', r2, 1 - sum((mp-mp_fit2).^2)/sum((mp-mean(mp)).^2));

% ---- Fit 3: cubic in log10(EI) ----
X3 = [ones(n,1), logEI, logEI.^2, logEI.^3];
c3 = X3 \ mp;
mp_fit3 = X3 * c3;
r3 = sqrt(mean((mp - mp_fit3).^2));
fprintf('\nFit 3: x_M/L = %.4f + %.4f*log10(EI) + %.6f*log10(EI)^2 + %.8f*log10(EI)^3\n', ...
    c3(1), c3(2), c3(3), c3(4));
fprintf('  RMSE = %.4f, R^2 = %.4f\n', r3, 1 - sum((mp-mp_fit3).^2)/sum((mp-mean(mp)).^2));

% ---- Fit 4: power law  x_M/L = a * EI^b ----
logmp = log10(mp);
X4 = [ones(n,1), logEI];
c4 = X4 \ logmp;
mp_fit4 = 10.^(X4 * c4);
r4 = sqrt(mean((mp - mp_fit4).^2));
fprintf('\nFit 4: x_M/L = %.4f * EI^%.4f\n', 10^c4(1), c4(2));
fprintf('  RMSE = %.4f, R^2 = %.4f\n', r4, 1 - sum((mp-mp_fit4).^2)/sum((mp-mean(mp)).^2));

% ---- Write CSV for comparison ----
T = table(curve_EI(:), mp, mp_fit1, mp_fit2, mp_fit3, mp_fit4, ...
    mp-mp_fit1, mp-mp_fit2, mp-mp_fit3, mp-mp_fit4, ...
    'VariableNames', {'EI','xM_data','linear','quadratic','cubic','power', ...
    'res_lin','res_quad','res_cub','res_pow'});
csvfile = fullfile(saveDir, 'second_family_fits.csv');
writetable(T, csvfile);
fprintf('\nSaved CSV to %s\n', csvfile);

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
