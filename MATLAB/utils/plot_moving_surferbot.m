function plot_moving_surferbot(structS, silent)
% PLOT_SURFERBOT_RUN
% Purpose:
%   Load a saved Surferbot simulation and produce an MP4 of eta(x,t) with
%   contact nodes and motor position, surferbot translating at speed U,
%   and a camera that pans across 60% of the domain.
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
% Outputs (saved in run_dir or current folder if S provided):
%   - waves.mp4
%
% Notes:
%   - Surferbot (contact region + motor marker) translates at speed U.
%   - Camera shows 60% of x-span and pans from one end to the other
%     (direction set by sign(U)).
%   - Requires VideoWriter and standard plotting functions.

%% --- 0.  Select folder / struct if not provided --------------------
if nargin < 2 || isempty(silent), silent = false; end
if isstruct(structS)
    % User passed the loaded data directly
    S = structS;                         % treat input as S
    structS = fullfile(pwd, 'figures');  % save outputs to current folder
    fprintf('Using provided struct S; saving outputs in: %s\n', structS);
else
    error('Not a struct');
end

% If we are using a directory, make sure it exists
if ~isempty(structS) && ~(exist('S','var') && ~isempty(S))
    assert(isfolder(structS), 'Folder not found: %s', structS);
end

%% --- 1.  Load data -------------------------------------------------
if isempty(S)
    dataFile = fullfile(structS,'results.mat');
    S = load(dataFile);
end
U    = -S.U;  
x    = S.x;  
%z    = S.z;  %#ok<NASGU>  % not used in this shortened version, kept for completeness
%phi  = S.phi; %#ok<NASGU>
eta  = S.eta; 
args = S.args;

%% --- 2.  Quick MP4 of eta(t,x) with moving surferbot ---------------

% --- (A) Video setup + timing ---
vid_len_sec = 15;           % total video duration (in periods)
fps         = 30;           % playback FPS

vidFile = fullfile(structS,'waves.mp4');
vid     = VideoWriter(vidFile,'MPEG-4');
vid.Quality = 100;
vid.FrameRate = fps;
open(vid);

omega  = args.omega;
Tper   = 2*pi/omega;        
N      = vid_len_sec * fps;           % total frames
tvec   = linspace(0, vid_len_sec*Tper, N);   % real-time timeline

scaleX = 1e2;  % cm
scaleY = 1e6;  % µm

% limits and geometry
sy = max(abs(eta),[],'all');
y_limit_microns = sy * scaleY * 1.1;

x_scaled = x * scaleX;
x_min = x_scaled(1);
x_max = x_scaled(end);

% Camera: 60% field of view that pans
Lx        = x_max - x_min;
win_frac  = 0.45;
win_width = win_frac * Lx;

% Direction of pan determined by sign(U):
%  U >= 0: left -> right
%  U  < 0: right -> left
if U >= 0
    x_start0 = x_min;
    x_start1 = x_max - win_width;
else
    x_start0 = x_max - win_width;
    x_start1 = x_min;
end
x_start_vec = linspace(x_start0, x_start1, N);  % left bound over time

ocean = [0.20 0.45 0.80];

% --- surferbot geometry (mask + base positions) --------------------
hasRaft   = isfield(args,'x_contact') && ~isempty(args.x_contact);
hasMotor  = isfield(args,'motor_position') && ~isempty(args.motor_position);

% contact is a logical mask over x
if hasRaft
    contact_mask = logical(args.x_contact(:).');   % 1×Nx logical
end

% find nearest x index for motor_position (assumes coordinate, not index)
if hasMotor
    [~, motor_idx0] = min(abs(x - args.motor_position));
    x_motor0 = x_scaled(motor_idx0);               % base motor x (cm)
end

if hasRaft
    x_contact0 = x_scaled(contact_mask);           % base raft x (cm)
end


% integer index offsets for each frame due to translation at speed U
%   offset = U * t / dx  (m/s * s / m -> index)
%idx_offset_vec = round((U * tvec) / dx);

% --- graphics priming -----------------------------------------------
fig = figure('Position',[200 200 1400 480], 'Color', 'w', 'Renderer', 'opengl');
set(gca,'LooseInset',[0.02 0.02 0.02 0.02]);

yy0 = real(eta .* exp(1i*omega*tvec(1))) * scaleY;
bottomVec = -y_limit_microns * ones(1, numel(x_scaled));

hFill = patch([x_scaled, fliplr(x_scaled)], ...get(
              [yy0(:).', bottomVec], ...
              ocean, 'EdgeColor','none','FaceAlpha',0.25, ...
              'HandleVisibility','off'); hold on
hLine = plot(x_scaled, yy0, 'b', 'LineWidth', 4, 'HandleVisibility','off');

if hasRaft
    hContact = line( ...
        'XData', x_contact0, ...
        'YData', yy0(contact_mask), ...
        'Color', 'k', ...
        'LineWidth', 8, ...
        'LineStyle', '-', ...
        'Marker', 'none', ...
        'HandleVisibility', 'off');
end



% yellow marker at motor position
if hasMotor
    dark_yellow = [0.85 0.65 0.00];
    hMotor = plot(x_scaled(motor_idx0), yy0(motor_idx0), 'o', ...
                  'MarkerFaceColor', dark_yellow, ...
                  'MarkerEdgeColor', dark_yellow, ...
                  'MarkerSize', 25, ...
                  'LineStyle', 'none', ...
                  'DisplayName', 'Motor position');
end

% Initial camera window
xlim([x_start_vec(1), x_start_vec(1) + win_width]);

% add a little buffer on TOP so legend never overlaps
top_buffer = 0.35; % 35% extra headroom
ylim([-y_limit_microns, y_limit_microns*(1+top_buffer)]);

xlabel('$x\;(\mathrm{cm})$', 'Interpreter','latex');
ylabel('$y\;(\mu\mathrm{m})$', 'Interpreter','latex');
set(gca,'FontSize',40); box on;

% place legend for the motor only
if hasMotor
    legend(hMotor, 'Location','northeast');
end

%% VIDEO LOOP
for k = 1:numel(tvec)
    t  = tvec(k);
    yy = real(eta .* exp(1i*omega*t)) * scaleY;

    % displacement in x (cm)
    dx_cm = U * t * scaleX;
    
    % --- 1. Update Wave Shape (SHIFTED) ---
    % We must shift the water's X-coordinates so the "wake" travels with the bot
    x_shifted = x_scaled + dx_cm;
    
    set(hLine, ...
        'XData', x_shifted, ...    % Shift the water right
        'YData', yy);

    set(hFill, ...
        'XData', [x_shifted, fliplr(x_shifted)], ... % Shift the fill right
        'YData', [yy(:).', bottomVec]);

    % --- 2. Update Surferbot Raft ---
    if hasRaft
        % Raft Y-coordinates come directly from yy(mask) because 
        % the raft is fixed relative to the wave field 'yy'
        set(hContact, ...
            'Visible', 'on', ...
            'XData', x_contact0 + dx_cm, ...
            'YData', yy(contact_mask), ... 
            'Color', 'k');
    end

    % --- 3. Update Motor Marker ---
    if hasMotor
        set(hMotor, ...
            'Visible', 'on', ...
            'XData', x_motor0 + dx_cm, ...
            'YData', yy(motor_idx0));
    end

    % Update camera window (panning)
    x_left = x_start_vec(k);
    xlim([x_left, x_left + win_width]);

    title(sprintf('f = %.2f Hz, t = %.3f s', omega/(2*pi), t));
    drawnow limitrate nocallbacks

    if silent == false
        frame = getframe(fig);
        writeVideo(vid, frame);
    end
end


if silent == false
    close(vid);
end


end
