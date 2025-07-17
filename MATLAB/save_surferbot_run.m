function run_id = save_surferbot_run(outdir_base, varargin)
%OUTDIR_BASE  e.g. 'surferbot_results'
%
% VARARGIN     same list you pass to flexible_surferbot_v2

    % ---------- 0.  Create timestamped subfolder -----------------------
    timestamp = datestr(now, 'yyyy-mm-ddTHH-MM-SSZ');
    run_id    = fullfile(outdir_base, timestamp);
    mkdir(run_id);

    % ---------- 1.  Call the simulator --------------------------------
    [U, x, z, phi, eta, args] = flexible_surferbot_v2(varargin{:});

    % ---------- 2.  Save results --------------------------------------
    save(fullfile(run_id, 'results.mat'), 'U', 'x', 'z', 'phi', 'eta', '-v7.3');
    
    % ---------- 2b.  make a quick mp4 of ?(t,x) ---------------------------------
    vidFile = fullfile(run_id,'waves.mp4');
    vid     = VideoWriter(vidFile,'MPEG-4');
    vid.FrameRate = args.omega/(2*pi);               % ?200 frames in 2 s of real time
    open(vid)

    omega  = args.omega;
    tvec   = linspace(0,10*pi/omega,500);       % one standing-wave period
    scaleX = 1e2;  scaleY = 1e6;                % cm, ?m

    fig = figure('Visible','off','Position',[0 0 900 240]);
    for k = 1:numel(tvec)
        yy = real(eta .* exp(1i*omega*tvec(k)));
        plot(x*scaleX , yy*scaleY ,'b','LineWidth',2); hold on
        plot(x(args.x_contact)*scaleX , yy(args.x_contact)*scaleY ,'r','LineWidth',3)
        xlim([-0.1 0.1]*scaleX); ylim([-300e-6 300e-6]* scaleY);
        xlabel('x (cm)'); ylabel('y (?m)');
        title(sprintf('t = %.5f s',tvec(k)));
        set(gca, 'FontSize', 16)
        frame = getframe(fig);
        writeVideo(vid,frame);
        cla
    end
    close(vid);  close(fig)

    % ---------- 2c.  static t = 0 visualisations -----------------------------
    % common scaling
    scaleY = 1e6;                                      % ?m for ?
    % --- figure 1 : surface deflection + loads --------------------------------
    f1 = figure('Visible','off','Position',[0 0 1600 420]);
    plot(x,  real(eta)*scaleY ,'b','LineWidth',1.5); hold on
    plot(x(args.x_contact), real(eta(args.x_contact))*scaleY ,'r','LineWidth',2)
    quiver(x(args.x_contact), real(eta(args.x_contact).')*scaleY , ...
           zeros(1, nnz(args.x_contact)), args.loads.'/5e4*scaleY ,0,'MaxHeadSize',1e-6)
    xlabel('x (m)'); ylabel('y (um)'); set(gca,'FontSize',16)
    title(sprintf('Surface deflection  ?  U = %.3f mm/s',U*1e3))
    saveas(f1, fullfile(run_id,'eta_t0.png'))
    savefig(f1, fullfile(run_id,'eta_t0.fig'))
    close(f1)

    % --- figure 2 : imag(?) field -------------------------------------------
    f2 = figure('Visible','off','Position',[0 0 1600 420]);
    colormap(f2,'winter'); shading interp
    p  = pcolor(x', z, imag(phi));  set(p,'EdgeColor','none')
    colorbar; hold on
    contour(x', z, imag(phi)/(args.omega*args.L_raft^2), 8,'k')
    set(gca,'FontSize',20); title('Imag(\phi) field')
    saveas(f2, fullfile(run_id,'phi_imag_t0.png'))
    savefig(f2, fullfile(run_id,'phi_imag_t0.fig'))
    close(f2)

    

    % ---------- 3.  Save the config as JSON ---------------------------
    fid = fopen(fullfile(run_id, 'config.json'), 'w');
    % --- write args as compact JSON ----------------------------------------
    cfg = struct(); fn = fieldnames(args);                     % ? keep-able copy
    for k = 1:numel(fn), try jsonencode(args.(fn{k}));         % ? test encodability
            cfg.(fn{k}) = args.(fn{k}); end, end               % ? keep if OK
    try  js = jsonencode(cfg,'PrettyPrint',true);              % ? pretty when possible
    catch, js = jsonencode(cfg); end                           %    compact otherwise
    fwrite(fid, js, 'char');                                   % ? write file

    fclose(fid);
    % --------- pretty-print this single JSON (Python helper) ---------------
    
    jsonFile = fullfile(run_id,'config.json');
    tmpFile  = [jsonFile '.tmp'];
    if ispc                                     % Windows
        cmd = sprintf('python3 -m json.tool "%s" > "%s" && move /Y "%s" "%s"', ...
                      jsonFile,tmpFile,tmpFile,jsonFile);
    else                                        % macOS / Linux
        cmd = sprintf('python3 -m json.tool "%s" > "%s" && mv "%s" "%s"', ...
                      jsonFile,tmpFile,tmpFile,jsonFile);
    end
    if system(cmd)~=0
        warning('Pretty-print skipped (no system Python found).');
    end



    % ---------- 4.  Append one-line manifest entry --------------------
    manifest = fullfile(outdir_base, 'manifest.csv');
    headline = isempty(dir(manifest));
    f = fopen(manifest, 'a');
    if headline
        fprintf(f, "run_id,U_m,f_hz,L_raft_m,n,M,BC\n");
    end
    fprintf(f, "%s,%.6g,%.6g,%.6g,%d,%d,%s\n", ...
            timestamp, U, args.omega/(2*pi), args.L_raft, args.n, args.M, args.BC);
    fclose(f);

    % ---------- 5.  (Optional) write a blank README for notes ---------
    fclose( fopen(fullfile(run_id,'README.txt'),'w') );

    fprintf('Saved run %s\n', timestamp);
end
