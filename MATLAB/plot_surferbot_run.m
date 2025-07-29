function plot_surferbot_run(run_dir)
%PLOT_SURFERBOT_RUN  Generate MP4 + static plots for a saved Surferbot run.
%
%   PLOT_SURFERBOT_RUN()           -> prompts for a folder via UI.
%   PLOT_SURFERBOT_RUN(run_dir)    -> uses the specified run directory.
%
%   The function expects the .mat file created by SAVE_SURFERBOT_RUN.

    %% --- 0.  Select folder if not provided -----------------------------
    if nargin < 1 || isempty(run_dir)
        sel = uigetdir(pwd,'Select a Surferbot run folder');
        if isequal(sel,0);  disp('Cancelled.');  return;  end
        run_dir = sel;
    end
    assert(isfolder(run_dir), 'Folder not found: %s', run_dir);

    %% --- 1.  Load data ---------------------------------------------------
    dataFile = fullfile(run_dir,'results.mat');
    S = load(dataFile);                    % loads U,x,z,phi,eta,args

    U    = S.U;  x = S.x;  z = S.z;  phi = S.phi; eta = S.eta; args = S.args;

    %% --- 2.  Quick MP4 of η(t,x) ----------------------------------------
    vidFile = fullfile(run_dir,'waves.mp4');
    vid     = VideoWriter(vidFile,'MPEG-4');
    vid.FrameRate = args.omega/(2*pi);   % 1 frame per solver time-step (≈1 Hz)
    open(vid);

    omega  = args.omega;
    tvec   = linspace(0,10*pi/omega,500);         % one standing-wave period
    scaleX = 1e2;  scaleY = 1e6;                  % cm, µm

    fig = figure('Visible','off','Position',[0 0 900 240]);
    sc = max(abs(eta),[],'all');
    scaleY = 10^(6-ceil(-log10(sc)));             % auto-scale

    for k = 1:numel(tvec)
        yy = real(eta .* exp(1i*omega*tvec(k)));
        plot(x*scaleX , yy*scaleY ,'b','LineWidth',2); hold on
        plot(x(args.x_contact)*scaleX , yy(args.x_contact)*scaleY , ...
             'r','LineWidth',3)
        xlim([-0.1 0.1]*scaleX);  ylim([-10 30]*scaleY);
        xlabel('x (cm)'); ylabel('y (μm)');
        title(sprintf('t = %.5f s',tvec(k)));
        set(gca,'FontSize',16)
        frame = getframe(fig);
        writeVideo(vid,frame);
        cla
    end
    close(vid);  close(fig);

    %% --- 3.  Static η(x,0) with loads -----------------------------------
    scaleY = 1e6;                               % µm
    f1 = figure('Visible','off','Position',[0 0 1600 420]);
    plot(x, real(eta)*scaleY ,'b','LineWidth',1.5); hold on
    plot(x(args.x_contact),real(eta(args.x_contact))*scaleY , ...
         'r','LineWidth',2);
    quiver(x(args.x_contact),real(eta(args.x_contact).')*scaleY , ...
           zeros(1,nnz(args.x_contact)), args.loads.'/5e4*scaleY ,0, ...
           'MaxHeadSize',1e-6);
    xlabel('x (m)'); ylabel('y (μm)'); set(gca,'FontSize',16)
    title(sprintf('Surface deflection   U = %.3f mm/s',U*1e3))
    saveas(f1, fullfile(run_dir,'eta_t0.png'));
    savefig(f1, fullfile(run_dir,'eta_t0.fig'));
    close(f1);

    %% --- 4.  Imaginary φ field ------------------------------------------
    f2 = figure('Visible','off','Position',[0 0 1600 420]);
    colormap(f2,'winter'); shading interp
    p  = pcolor(x', z, imag(phi));  set(p,'EdgeColor','none');
    colorbar; hold on
    contour(x', z, imag(phi)/(args.omega*args.L_raft^2), 8,'k');
    set(gca,'FontSize',20); title('Imag(\phi) field');
    saveas(f2, fullfile(run_dir,'phi_imag_t0.png'));
    savefig(f2, fullfile(run_dir,'phi_imag_t0.fig'));
    close(f2);

    fprintf('Plots saved in %s\n', run_dir);
end
