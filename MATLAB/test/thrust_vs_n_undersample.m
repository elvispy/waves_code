function thrust_vs_n_undersample(save_path)
% Runs two raft simulations with different spatial resolutions (n_pair)
% Checks that the coarse grid (x,z) points exist within the fine grid
% Extracts fine data on coarse grid (no interpolation), saves results and L2 error

addpath('../src');

% ---- base physical and numerical parameters ----
L_raft = 0.05;
base = struct( ...
    'sigma',72.2e-3, 'rho',1000, 'nu',0, 'g',9.81, ...
    'L_raft',L_raft, 'motor_position',0.35*L_raft/2, 'd',L_raft/2, ...
    'EI',3.0e9*3e-2*(9.9e-4)^3/12, 'rho_raft',0.018*10.0, ...
    'domainDepth',0.5, 'n',115, 'M',200, ...
    'motor_inertia',0.13e-3*2.5e-3, 'BC','radiative', ...
    'omega',2*pi*10, 'ooa', 2);

% ---- choose exactly two N values ----
n_pair = ensure_odd([115, 229]);      % example pair
n_pair = unique(n_pair,'stable');
if numel(n_pair) ~= 2, error('Need exactly two distinct N values.'); end

% ---- run both cases ----
S = repmat(empty_proto(),1,2);
for i = 1:2
    p = base; p.n = n_pair(i);
    S(i) = run_case(p);
end

% ---- pick coarse vs fine by N_x ----
[~, iC] = min([S.N_x]);  [~, iF] = max([S.N_x]);
C = S(iC);  F = S(iF);

% ---- subset check with tolerance ----
tolx = max(eps(max(1,abs(F.x))))*10;      % conservative ulp-based tol
tolz = max(eps(max(1,abs(F.z))))*10;

[isSubX, idxX] = ismembertol(C.x(:), F.x(:), tolx, 'DataScale', 1);
[isSubZ, idxZ] = ismembertol(C.z(:), F.z(:), tolz, 'DataScale', 1);

if ~all(isSubX) || ~all(isSubZ)
    missingX = find(~isSubX, 1);
    missingZ = find(~isSubZ, 1);
    if ~all(isSubX)
        warning('Coarse x not subset of fine x. Example missing at x=%.16g.', C.x(missingX));
    end
    if ~all(isSubZ)
        warning('Coarse z not subset of fine z. Example missing at z=%.16g.', C.z(missingZ));
    end
    error('Aborting: subset requirement failed. No file saved.');
end

% ---- extract fine-on-coarse by indexing (no interpolation) ----
etaF_on_C = F.eta(idxX);                         % 1D along x
phiF_on_C = F.phi(idxZ, idxX);                   % 2D (rows=z, cols=x)

% ---- package and save ----
meta.N_coarse = C.N_x;  meta.N_fine = F.N_x;
meta.M        = C.M_z;
meta.args_coarse = C.args;
meta.args_fine   = F.args;

if nargin < 1 || isempty(save_path)
    ts = datestr(now,'yyyymmdd_HHMMSS');
    save_path = sprintf('undersampled_Nc%d_Nf%d_%s.mat', meta.N_coarse, meta.N_fine, ts);
end

xC = C.x;  zC = C.z;  etaC = C.eta;  phiC = C.phi; %#ok<NASGU>
save(save_path, 'xC','zC','etaC','phiC','etaF_on_C','phiF_on_C','meta');

% ---- minimal console output ----
l2_eta = norm(etaF_on_C - etaC) / sqrt(numel(etaC));
fprintf('Saved %s | N_coarse=%d N_fine=%d | L2(eta diff)=%.3e\n', ...
        save_path, meta.N_coarse, meta.N_fine, l2_eta);
end

% ================= helpers =================
function S = empty_proto()
S = struct('x',[],'z',[],'phi',[],'eta',[], 'N_x',0,'M_z',0,'args',struct());
end

function n = ensure_odd(n)
n = n(:).'; for k = 1:numel(n), if mod(n(k),2)==0, n(k)=n(k)+1; end, end
end

function S = run_case(p)
[~, x, z, phi, eta, args] = flexible_surferbot_v2( ...
    'sigma',p.sigma,'rho',p.rho,'omega',p.omega,'nu',p.nu,'g',p.g, ...
    'L_raft',p.L_raft,'motor_position',p.motor_position,'d',p.d, ...
    'EI',p.EI,'rho_raft',p.rho_raft,'domainDepth',p.domainDepth, ...
    'n',p.n,'M',p.M,'motor_inertia',p.motor_inertia,'BC',p.BC);

S = struct('x',x,'z',z,'phi',phi,'eta',eta, 'N_x',args.N,'M_z',args.M,'args',args);
end
