function plot_surferbot_run(run_dir, silent)
% PLOT_SURFERBOT_RUN
% Purpose:
%   Load a saved Surferbot simulation and produce an MP4 plus key figures.
%
% How to run:
%   - Call PLOT_SURFERBOT_RUN() to pick from surferbot_results/manifest.csv.
%   - Or call PLOT_SURFERBOT_RUN(run_dir) with a run folder path.
%   - Or pass a loaded struct S directly (skips disk I/O; saves plots locally).
%
% Inputs:
%   run_dir  String/char path to a run directory, or a struct S with fields:
%            U, x, z, phi, eta, args (with omega, x_contact, pressure, etc.).
%
% What it does:
%   - Loads results.mat from run_dir (if not given S).
%   - Builds a short MP4 of ?(x,t) with contact nodes highlighted.
%   - Saves static ?(x) at t=0 with U annotation.
%   - Renders Imag(?)(x,z) as pcolor + contours.
%   - Saves |?|(x) and phase(?)(x) panels.
%   - plots fore/aft surferbot trajectories in (x,y) colored by time.
%
% Outputs (saved in run_dir or current folder if S provided):
%   - waves.mp4, eta_t0.png/.fig, phi_imag_t0.png/.fig, eta_mag_phase.png/.fig,
%     surferbot_xy_fore_aft.png/.fig.
% Notes:
%   - Video y-limits auto-scale from |?|; x-limits from contact span.
%   - Requires VideoWriter and standard plotting functions.

%% --- 0.  Select folder / struct if not provided --------------------
S = [];                        % <-- will hold data if provided as struct
if nargin < 2 || isempty(silent), silent = false; end
if nargin < 1 || isempty(run_dir)
    % Define path and check for the manifest file
    outdir = 'surferbot_results';
    manifestFile = fullfile(outdir, 'manifest.csv');
    if ~isfile(manifestFile)
        error('Manifest file not found: %s', manifestFile);
    end

    % Read the manifest into a table
    T = readtable(manifestFile);

    % Add a 'Row' column for easy selection and display the table
    numbered_T = addvars(T, (1:height(T))', 'Before', 1, 'NewVariableNames', 'Index');
    fprintf('Available simulations:\n');
    disp(numbered_T);

    % Prompt user for selection with input validation
    selection = NaN;
    while isnan(selection) || floor(selection) ~= selection || selection < 1 || selection > height(T)
        try
            selection = input(sprintf('Enter the row number to plot [1-%d]: ', height(T)));
            if isempty(selection) || ~(isnumeric(selection) && isscalar(selection))
                selection = NaN; % Trigger invalid input message
            end
        catch
            selection = NaN; % Trigger invalid input message
        end
        if isnan(selection) || floor(selection) ~= selection || selection < 1 || selection > height(T)
            fprintf('Invalid input. Please enter a single integer number from the "Row" column.\n');
        end
    end
    
    % Get the run directory from the user's selection
    run_id = T.run_id{selection};
    run_dir = fullfile(outdir, run_id);
    fprintf('\nPlotting run: %s\n', run_id);

elseif isstruct(run_dir)
    % User passed the loaded data directly
    S = run_dir;                         % treat input as S
    run_dir = fullfile(pwd, 'figures');  % save outputs to current folder
    fprintf('Using provided struct S; saving outputs in: %s\n', run_dir);
end

% If we are using a directory, make sure it exists
if ~isempty(run_dir) && ~(exist('S','var') && ~isempty(S))
    assert(isfolder(run_dir), 'Folder not found: %s', run_dir);
end

%% --- 1.  Load data -------------------------------------------------
if isempty(S)
    dataFile = fullfile(run_dir,'results.mat');
    S = load(dataFile);
end
U   = S.U;  
x   = S.x;  
z   = S.z;  
phi = S.phi; 
eta = S.eta; 
args = S.args;

%% --- 2.  Quick MP4 of ?(t,x) ---------------------------------------
% --- (A) Video setup + timing ---
vid_len_sec = 10;           % total video duration
fps         = 30;           % playback FPS

vidFile = fullfile(run_dir,'waves.mp4');
vid     = VideoWriter(vidFile,'MPEG-4');
vid.FrameRate = fps;
open(vid);

omega  = args.omega;
T      = 2*pi/omega;        
N      = vid_len_sec * fps; % total frames
tvec   = linspace(0, vid_len_sec * T, N);   % real-time timeline

scaleX = 1e2;  % cm
scaleY = 1e6;  % µm

% limits and geometry
sy = max(abs(eta),[],'all');
y_limit_microns = sy * scaleY * 1.1;

x_scaled = x * scaleX;
x_min = x_scaled(1);
x_max = x_scaled(end);

ocean = [0.20 0.45 0.80];
% --- compute indices and styling ---
hasRaft   = isfield(args,'x_contact') && ~isempty(args.x_contact);
hasMotor  = isfield(args,'motor_position') && ~isempty(args.motor_position);

% find nearest x index for motor_position (assumes coordinate, not index)
if hasMotor
    [~, i_motor] = min(abs(x - args.motor_position));
end

fig = figure('Visible', ternary(~silent,'on','off'), 'Position',[200 200 900 240]);
function out = ternary(cond,a,b), if cond, out=a; else, out=b; end, end

% --- graphics priming ---
yy0 = real(eta .* exp(1i*omega*tvec(1))) * scaleY;
bottomVec = -y_limit_microns * ones(1, numel(x_scaled));

hFill = patch([x_scaled, fliplr(x_scaled)], ...
              [yy0(:).', bottomVec], ...
              ocean, 'EdgeColor','none','FaceAlpha',0.25, ...
              'HandleVisibility','off'); hold on
hLine = plot(x_scaled, yy0, 'b', 'LineWidth', 2, 'HandleVisibility','off');

if hasRaft
    hContact = plot(x_scaled(args.x_contact), yy0(args.x_contact), ...
                    'k', 'LineWidth', 4, 'HandleVisibility','off'); % raft is black
end

% yellow marker at motor position
if hasMotor
    dark_yellow = [0.85 0.65 0.00];
    hMotor = plot(x_scaled(i_motor), yy0(i_motor), 'o', ...
                  'MarkerFaceColor', dark_yellow, ...
                  'MarkerEdgeColor', dark_yellow, ...
                  'MarkerSize', 10, ...
                  'LineStyle', 'none', ...
                  'DisplayName', 'Motor position');
end

xlim([x_min, x_max]);

% add a little buffer on TOP so legend never overlaps
top_buffer = 0.35; % 35% extra headroom
ylim([-y_limit_microns, y_limit_microns*(1+top_buffer)]);

xlabel('$x\;(\mathrm{cm})$', 'Interpreter','latex');
ylabel('$y\;(\mu\mathrm{m})$', 'Interpreter','latex');
set(gca,'FontSize',20); box on;

% place legend for the star only
if hasMotor
    legend(hMotor, 'Location','northeast');
end


%% VIDEO
for k = 1:numel(tvec)
    yy = real(eta .* exp(1i*omega*tvec(k))) * scaleY;

    set(hFill, 'YData', [yy(:).', bottomVec]);
    set(hLine,  'YData', yy);

    if hasRaft
        set(hContact, 'YData', yy(args.x_contact));
    end
    if hasMotor
        set(hMotor, 'YData', yy(i_motor));
    end

    title(sprintf('f = %d Hz, t = %.3f s', omega/(2*pi), tvec(k)));
    drawnow limitrate nocallbacks
    frame = getframe(fig);
    writeVideo(vid, frame);
end

close(vid);

%% 2b. Surferbot fore/aft trajectory in (x,y) --------------------
% Use your formula: x(t) = real(eta(1)*exp(i*omega*t)) + U*t
% and analogously for eta(end). y(t) = real(eta(k)*exp(i*omega*t)).
[~, fore_idx] = max(abs(S.eta(1:end/2)));
[~,  aft_idx] = max(abs(S.eta(end/2:end))); aft_idx = aft_idx + round(numel(S.eta)/2);
eta_fore = eta(fore_idx);
eta_aft  = eta(aft_idx);

phi_surface = transpose(S.phi(end, :)); 
Dx = getNonCompactFDmatrix(S.args.N, S.args.dx, 1, S.args.ooa);
phi_x_fore = Dx(fore_idx, :) * phi_surface;
phi_x_aft  = Dx(aft_idx, :)  * phi_surface;

tvec = linspace(0, 56, 100)/1000;
y_fore = real(eta_fore .* exp(1i*omega*tvec));  % y position (fore)
y_aft  = real(eta_aft  .* exp(1i*omega*tvec));  % y position (aft)

x_fore = (real(phi_x_fore .* exp(1i*omega*tvec)) + U) .* tvec; % x position (fore)
x_aft  = (real(phi_x_aft  .* exp(1i*omega*tvec)) + U) .* tvec; % x position (aft)

f4 = figure('Visible', ternary(~silent,'on','off'), 'Position',[200 200 700 600]);
subplot(2, 1, 1);
hold on;
scatter(1e3 * x_fore, 1e6 * y_fore, 50, 1e3 * tvec, 'filled');   % fore
xlabel('x (mm)');
ylabel('y (um)');
title('Surferbot fore trajectory');
grid on;
set(gca,'FontSize',16);
cb = colorbar;
cb.Label.String = 'time (ms)';

subplot(2, 1, 2);
s2 = scatter(1e3 * x_aft,  1e6 * y_aft,  50, 1e3 * tvec, 'filled');   % aft
set(s2, 'Marker','^');                                      % different marker
xlabel('x (mm)');
ylabel('y (um)');
title('Surferbot aft trajectory');
grid on;
set(gca,'FontSize',16);

cb = colorbar;
cb.Label.String = 'time (ms)';

%legend([s1 s2], {'Fore','Aft'}, 'Location','best');

saveas(f4, fullfile(run_dir,'surferbot_xy_fore_aft.svg'));
savefig(f4, fullfile(run_dir,'surferbot_xy_fore_aft.fig'));

%% --- 3.  Static ?(x,0) with loads ----------------------------------
scaleY = 1e6;
f1 = figure('Visible','on','Position',[0 600 1600 420]);
plot(x, real(eta)*scaleY ,'b','LineWidth',1.5); hold on
plot(x(args.x_contact),real(eta(args.x_contact))*scaleY , ...
        'r','LineWidth',2);
xlabel('x (m)'); ylabel('y (um)'); set(gca,'FontSize',16)
title(sprintf('Surface deflection   U = %.3f mm/s',U*1e3))
saveas(f1, fullfile(run_dir,'eta_t0.svg'));
savefig(f1, fullfile(run_dir,'eta_t0.fig'));

%% --- 4.  Imaginary ? field -----------------------------------------
f2 = figure('Visible','on','Position',[0 1000 1600 420]);
colormap(f2,'winter'); shading interp
p  = pcolor(x', z, imag(phi));  set(p,'EdgeColor','none');
colorbar; hold on
contour(x', z, imag(phi)/(args.omega*args.L_raft^2), 8,'k');
set(gca,'FontSize',20); title('Imag(\phi) field');
saveas(f2, fullfile(run_dir,'phi_imag_t0.svg'));
savefig(f2, fullfile(run_dir,'phi_imag_t0.fig'));

%% --- 3b.  Surface |eta| and phase(eta) panels ----------------------
scaleY = 1e6;              
mag_eta   = abs(eta) * scaleY;
phase_eta = unwrap(angle(eta)); 

f3b = figure('Visible','on','Position',[100 500 1600 820]);

% Top: |eta|(x) at surface
subplot(2,1,1)
plot(x, mag_eta, 'b', 'LineWidth', 1.5); hold on
if isfield(args,'x_contact') && ~isempty(args.x_contact)
    plot(x(args.x_contact), mag_eta(args.x_contact), 'r.', 'MarkerSize', 16);
end
xlabel('x (m)'); ylabel('|eta| (um)'); set(gca,'FontSize',16)
title('Magnitude of \eta at surface')

% Bottom: phase(eta)(x) at surface
subplot(2,1,2)
plot(x, phase_eta, 'k', 'LineWidth', 1.5); hold on
if isfield(args,'x_contact') && ~isempty(args.x_contact)
    plot(x(args.x_contact), phase_eta(args.x_contact), 'r.', 'MarkerSize', 16);
end
xlabel('x (m)'); ylabel('phase(\eta) (rad)'); set(gca,'FontSize',16)
title('Phase of \eta at surface')

saveas(f3b, fullfile(run_dir,'eta_mag_phase.svg'));
savefig(f3b, fullfile(run_dir,'eta_mag_phase.fig'));

fprintf('Plots saved in %s\n', run_dir);

end
