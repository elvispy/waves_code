function regenerate_manifest(outdir)
%REGENERATE_MANIFEST  Rebuild manifest.csv from the saved run folders.
%
%   regenerate_manifest()                % assumes 'surferbot_results'
%   regenerate_manifest(outdir)          % specify another root folder
%
%   The script expects each run directory to contain:
%     ? results.mat   (with variable U)
%     ? config.json   (pretty or compact, created by save_surferbot_run)

% ------------------------------------------------------------------------
if nargin < 1 || isempty(outdir), outdir = 'surferbot_results'; end
assert(isfolder(outdir), 'Folder not found: %s', outdir);

manifestFile = fullfile(outdir, 'manifest.csv');
tmpFile      = [manifestFile '.tmp'];          % atomic write

% ------------------------------------------------------------------------
% 1.  Gather run directories
% ------------------------------------------------------------------------
d = dir(outdir);
d = d([d.isdir] & ~startsWith({d.name}, '.'));
% sort by datenum (optional, newest last)
[~,idx] = sort([d.datenum]);
d = d(idx);

% ------------------------------------------------------------------------
% 2.  Open temporary CSV
% ------------------------------------------------------------------------
fid = fopen(tmpFile,'w');
header = ["run_id" "U_m" "f_hz" "EI_Nm2" "motor_pos_m" "motor_inertia_kgm" ...
          "domain_depth_m" "L_raft_m" "n" "M" "BC"];
fprintf(fid, "%s\n", strjoin(header,','));

% ------------------------------------------------------------------------
% 3.  Loop over runs
% ------------------------------------------------------------------------
for k = 1:numel(d)
    runDir      = fullfile(outdir, d(k).name);
    json_file   = fullfile(runDir, 'config.json');
    result_file = fullfile(runDir, 'results.mat');

    if ~(isfile(json_file) && isfile(result_file))
        warning('Skipping %s (missing files).', d(k).name);
        continue
    end

    try
        args = jsondecode(fileread(json_file));
        S    = load(result_file, 'U');
        U    = S.U;
    catch ME
        warning('Skipping %s (%s).', d(k).name, ME.message);
        continue
    end

    % build CSV row
    row = sprintf("%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%s\n", ...
         d(k).name, ...
         U, ...
         args.omega/(2*pi), ...
         args.EI, ...
         args.motor_position, ...
         args.motor_inertia, ...
         args.domainDepth, ...
         args.L_raft, ...
         args.n, ...
         args.M, ...
         args.BC);

    fprintf(fid, "%s", row);
end

fclose(fid);

% ------------------------------------------------------------------------
% 4.  Move tmp ? final (atomic on the same filesystem)
% ------------------------------------------------------------------------
movefile(tmpFile, manifestFile, 'f');          % 'f' = overwrite

fprintf('Manifest regenerated at %s (%d runs)\n', manifestFile, numel(d));
end
