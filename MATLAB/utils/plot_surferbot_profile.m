function plot_surferbot_profile(run_dir)
% PLOT_SURFERBOT_PROFILE
% Purpose:
%   Reproduce a 1D snapshot of the free-surface elevation h(x) with the raft
%   region highlighted as a flat segment, similar to the reference figure.
%
% Usage:
%   plot_surferbot_profile()              % choose run from manifest
%   plot_surferbot_profile(run_dir)       % give path to run directory
%   plot_surferbot_profile(S)             % pass loaded struct S directly
%
% Inputs:
%   run_dir  String/char path to a run directory, OR a struct S with fields:
%            U, x, z, phi, eta, args (with x_contact, omega, etc.).
%
% Output:
%   Saves a figure 'eta_profile_1D.svg' and '.fig' in run_dir (or pwd/figures
%   if a struct S is passed).

%% --- 0.  Select folder / struct if not provided --------------------
S = [];  % will hold data if provided as struct

if nargin < 1 || isempty(run_dir)
    % Same manifest logic as plot_surferbot_run
    outdir = 'surferbot_results';
    manifestFile = fullfile(outdir, 'manifest.csv');
    if ~isfile(manifestFile)
        error('Manifest file not found: %s', manifestFile);
    end

    T = readtable(manifestFile);
    numbered_T = addvars(T, (1:height(T))', 'Before', 1, ...
                         'NewVariableNames', 'Index');
    fprintf('Available simulations:\n');
    disp(numbered_T);

    selection = NaN;
    while isnan(selection) || floor(selection) ~= selection || ...
          selection < 1 || selection > height(T)
        try
            selection = input(sprintf('Enter the row number to plot [1-%d]: ', height(T)));
            if isempty(selection) || ~(isnumeric(selection) && isscalar(selection))
                selection = NaN;
            end
        catch
            selection = NaN;
        end
        if isnan(selection) || floor(selection) ~= selection || ...
           selection < 1 || selection > height(T)
            fprintf('Invalid input. Please enter a single integer from the "Index" column.\n');
        end
    end

    run_id = T.run_id{selection};
    run_dir = fullfile(outdir, run_id);
    fprintf('\nPlotting 1D profile for run: %s\n', run_id);

elseif isstruct(run_dir)
    % User passed the loaded data directly
    S = run_dir;
    run_dir = fullfile(pwd, 'figures');
    fprintf('Using provided struct S; saving outputs in: %s\n', run_dir);
end

if ~isempty(run_dir) && ~(exist('S','var') && ~isempty(S))
    assert(isfolder(run_dir), 'Folder not found: %s', run_dir);
end
if ~isfolder(run_dir)
    mkdir(run_dir);
end

%% --- 1.  Load data -------------------------------------------------
if isempty(S)
    dataFile = fullfile(run_dir,'results.mat');
    S = load(dataFile);
end

U    = S.U;          %#ok<NASGU>  % not used, but kept for consistency
x    = flip(S.x);
eta  = S.eta;
args = S.args;

assert(isfield(args,'x_contact') && ~isempty(args.x_contact), ...
       'args.x_contact must exist and be non-empty.');

%% --- 2.  Build h(x) snapshot (t = 0) ------------------------------
% Use the stored complex amplitude at t=0: h(x,0) = Re(eta)
scaleX = 1e2;  % m -> cm
scaleY = 1e6;  % m -> µm

x_cm   = x * scaleX;
h_um   = real(eta) * scaleY;

% Indices where the raft is in contact
idx_contact = args.x_contact(:)';          % ensure column
%idx_free    = true(size(x_cm));
%idx_free(idx_contact) = false;

%% --- 3.  Plot: free surface (red) + raft (blue) --------------------
fig = figure('Position',[200 200 750 350], 'Color','w');
hold on;

% Free-surface waves (red), suppress over raft by NaNs
h_free = h_um;
h_free(idx_contact) = NaN;
plot(x_cm, h_free, 'r', 'LineWidth', 2);

% Raft region: flat blue segment at h = 0 over contact nodes
plot(x_cm(idx_contact), zeros(size(x_cm(idx_contact))), ...
     'b', 'LineWidth', 6);

% Axes, limits, labels
maxAbsH = max(abs(h_um(~isnan(h_free))),[],'all');
if isempty(maxAbsH) || maxAbsH == 0
    maxAbsH = 1;
end
ylim(2.05 * maxAbsH * [-1 1]);
yticks(-300:100:300);
xticks(-6:2:6);
xlim([min(x_cm) max(x_cm)]);

xlabel('$x~(\mathrm{cm})$','Interpreter','latex');
ylabel('$h~(\mu\mathrm{m})$','Interpreter','latex');
set(gca,'FontSize',24);
grid on; box on;

% Optional: remove legend for a clean APS-style figure
% If you want a legend, uncomment next two lines:
% legend({'Free surface','Raft contact'}, 'Location','best', ...
%        'Interpreter','latex');

% Tight layout
set(gca,'LooseInset',[0.02 0.02 0.02 0.02]);

%% --- 4.  Save ------------------------------------------------------
%out_svg = fullfile(run_dir,'eta_profile_1D.svg');
%out_fig = fullfile(run_dir,'eta_profile_1D.fig');
%saveas(fig, out_svg);
%savefig(fig, out_fig);
print(fig, fullfile(run_dir,'eta_profile_1D.pdf'), '-dpdf','-painters','-r300', '-bestfit');

fprintf('1D profile figure saved in %s\n', run_dir);

end
