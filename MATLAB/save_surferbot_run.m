function run_id = save_surferbot_run(outdir_base, varargin)
%SAVE_SURFERBOT_RUN  Run Surferbot simulation and save results/metadata.
%
%   run_id = SAVE_SURFERBOT_RUN(outdir_base, name,value,...)
%
%   This is a **streamlined** version: it no longer creates figures or
%   movies.  Use PLOT_SURFERBOT_RUN to visualise a finished run.

    % ---------- 0.  Time-stamped subfolder -------------------------------
    timestamp = datestr(now, 'yyyy-mm-ddTHH-MM-SSZ');
    run_id    = fullfile(outdir_base, timestamp);
    if ~exist(run_id,'dir');  mkdir(run_id);  end

    % ---------- 1.  Call the simulator -----------------------------------
    [U, x, z, phi, eta, args] = flexible_surferbot_v2(varargin{:});

    % ---------- 2.  Save raw results -------------------------------------
    save(fullfile(run_id, 'results.mat'), ...
         'U','x','z','phi','eta','args','-v7.3');

    % ---------- 3.  Save config as pretty-printed JSON --------------------
    jsonFile = fullfile(run_id,'config.json');
    fid = fopen(jsonFile, 'w');
    cfg = struct(); fn = fieldnames(args);
    for k = 1:numel(fn)
        try jsonencode(args.(fn{k}));  cfg.(fn{k}) = args.(fn{k}); end
    end
    try   js = jsonencode(cfg,'PrettyPrint',true);
    catch js = jsonencode(cfg);  end
    fwrite(fid, js, 'char');  fclose(fid);
    % re-format via system Python (optional)
    tmpFile = [jsonFile '.tmp'];
    if ispc
        cmd = sprintf('python -m json.tool "%s" > "%s" && move /Y "%s" "%s"', ...
                      jsonFile,tmpFile,tmpFile,jsonFile);
    else
        cmd = sprintf('python3 -m json.tool "%s" > "%s" && mv "%s" "%s"', ...
                      jsonFile,tmpFile,tmpFile,jsonFile);
    end
    if system(cmd) ~= 0
        warning('Pretty-print skipped (system Python not found).');
    end

    % ---------- 4.  Append one-line manifest entry -----------------------
    manifest = fullfile(outdir_base,'manifest.csv');
    headline = ~isfile(manifest);
    f = fopen(manifest,'a');
    if headline
        fprintf(f,"run_id,U_m,f_hz,EI_Nm2,motor_pos_m,bath_depth_m,L_raft_m,n,M,BC\n");
    end
    fprintf(f,"%s,%.6g,%.6g,%.6g,%g,%g,%g,%d,%d,%s\n", ...
            timestamp,U,args.omega/(2*pi),args.EI,args.motor_position, ...
            args.domainDepth,args.L_raft,args.n,args.M,args.BC);
    fclose(f);

    % ---------- 5.  Blank README for notes --------------------------------
    fclose(fopen(fullfile(run_id,'README.txt'),'w'));

    fprintf('Saved run %s\n', timestamp);
end
