function plot_surferbot_run(run_dir)
%PLOT_SURFERBOT_RUN  Generate plots for a saved Surferbot run.
%
%   PLOT_SURFERBOT_RUN()           -> displays manifest and prompts for selection.
%   PLOT_SURFERBOT_RUN(run_dir)    -> uses the specified run directory (char/str),
%                                     or uses struct S directly (no disk I/O).

    %% --- 0.  Select folder / struct if not provided --------------------
    S = [];                        % <-- will hold data if provided as struct
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
    
    %% --- 2.  Quick MP4 of η(t,x) ---------------------------------------
    vidFile = fullfile(run_dir,'waves.mp4');
    vid     = VideoWriter(vidFile,'MPEG-4');
    vid.FrameRate = 4*args.omega/(2*pi);
    open(vid);
    
    omega  = args.omega;
    tvec   = linspace(0,10*pi/omega,500);
    scaleX = 1e2;  % cm
    scaleY = 1e6;  % µm
    
    % --- DYNAMIC Y-LIMITS ---
    sy = max(abs(eta),[],'all');
    sx = max(abs(x(args.x_contact)), [], 'all');
    plot_buffer = 1.4; 
    y_limit_microns = sy * scaleY * plot_buffer;
    
    fig = figure('Visible','on','Position',[200 200 900 240]);
    
    for k = 1:numel(tvec)
        yy = real(eta .* exp(1i*omega*tvec(k)));
        pp = real(args.pressure .* exp(1i*omega*tvec(k)));
        
        plot(x*scaleX , yy*scaleY ,'b','LineWidth',2); hold on
        plot(x(args.x_contact)*scaleX , yy(args.x_contact)*scaleY , ...
             'r','LineWidth',3);
         
        %rangeY = max(yy) - min(yy);
        %arrowScale = 0.1 * rangeY / max(abs(pp));
         
        %quiver(x(args.x_contact) , yy(args.x_contact)', ...
        %    zeros(1, nnz(args.x_contact)), pp' * arrowScale * scaleY, ...
        %    0, 'MaxHeadSize', 0.5); 
        
        ylim([-y_limit_microns, y_limit_microns]);
        xlim([-sx, sx]*scaleX * 5);
        xlabel('x (cm)'); ylabel('y (um)');
        title(sprintf('t = %.5f s',tvec(k)));
        set(gca,'FontSize',16);
        
        frame = getframe(fig);
        writeVideo(vid,frame);
        cla
    end
    close(vid);  %close(fig);
    
    %% --- 3.  Static η(x,0) with loads ----------------------------------
    scaleY = 1e6;
    f1 = figure('Visible','on','Position',[0 600 1600 420]);
    plot(x, real(eta)*scaleY ,'b','LineWidth',1.5); hold on
    plot(x(args.x_contact),real(eta(args.x_contact))*scaleY , ...
         'r','LineWidth',2);
    %quiver(x(args.x_contact),real(eta(args.x_contact).')*scaleY , ...
    %       zeros(1,nnz(args.x_contact)), args.loads.'/5e4*scaleY ,0, ...
    %       'MaxHeadSize',1e-6);
    xlabel('x (m)'); ylabel('y (um)'); set(gca,'FontSize',16)
    title(sprintf('Surface deflection   U = %.3f mm/s',U*1e3))
    saveas(f1, fullfile(run_dir,'eta_t0.png'));
    savefig(f1, fullfile(run_dir,'eta_t0.fig'));
    %close(f1);
    
    %% --- 4.  Imaginary φ field -----------------------------------------
    f2 = figure('Visible','on','Position',[0 1000 1600 420]);
    colormap(f2,'winter'); shading interp
    p  = pcolor(x', z, imag(phi));  set(p,'EdgeColor','none');
    colorbar; hold on
    contour(x', z, imag(phi)/(args.omega*args.L_raft^2), 8,'k');
    set(gca,'FontSize',20); title('Imag(\phi) field');
    saveas(f2, fullfile(run_dir,'phi_imag_t0.png'));
    savefig(f2, fullfile(run_dir,'phi_imag_t0.fig'));
    %close(f2);
    
    fprintf('Plots saved in %s\n', run_dir);
end
