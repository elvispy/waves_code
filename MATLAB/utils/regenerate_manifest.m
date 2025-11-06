function regenerate_manifest(outdir)

% REGENERATE_MANIFEST
% Purpose:
%   Rebuild surferbot_results/manifest.csv by scanning run folders in parallel.
%
% How to run:
%   regenerate_manifest            % uses default "surferbot_results"
%   regenerate_manifest(outdir)    % specify alternative output folder
%
% Requirements:
%   - Each run folder contains config.json (input args) and results.mat with U.
%   - Parallel Computing Toolbox for parfor (falls back to serial if not available).
%
% What it does:
%   1) Lists subfolders in outdir, newest last.
%   2) In parallel, loads U from results.mat and args from config.json.
%   3) Writes a temporary CSV, then atomically moves it to manifest.csv.
%
% CSV columns:
%   run_id, U_m, f_hz, EI_Nm2, motor_pos_m, motor_inertia_kgm, domain_depth_m,
%   L_raft_m, n, M, BC
%
% Output:
%   - manifest.csv in outdir with one row per valid run.
% Notes:
%   - Skips folders missing config.json or results.mat; reports count at end.


if nargin < 1 || isempty(outdir), outdir = 'surferbot_results'; end
assert(isfolder(outdir), 'Folder not found: %s', outdir);

manifestFile = fullfile(outdir, 'manifest.csv');
tmpFile      = [manifestFile '.tmp'];  % atomic write

% ------------------------------------------------------------------------
% 1. Gather run directories
% ------------------------------------------------------------------------
d = dir(outdir);
d = d([d.isdir] & ~startsWith({d.name}, '.'));
[~,idx] = sort([d.datenum]);  % newest last
d = d(idx);
runDirs = fullfile(outdir, {d.name});
nRuns   = numel(runDirs);

% ------------------------------------------------------------------------
% 2. Preallocate cell array to collect results
% ------------------------------------------------------------------------
rows = cell(nRuns, 1);

% ------------------------------------------------------------------------
% 3. Parallel loop over runs
% ------------------------------------------------------------------------
parfor k = 1:nRuns
    runDir = runDirs{k};
    runName = d(k).name;
    json_file   = fullfile(runDir, 'config.json');
    result_file = fullfile(runDir, 'results.mat');

    if ~(isfile(json_file) && isfile(result_file))
        continue
    end

    try
        args = jsondecode(fileread(json_file));
        S    = load(result_file, 'U');
        U    = S.U;

        % build CSV row string
        row = sprintf("%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%s", ...
             runName, ...
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

        rows{k} = row;
    catch
        % skip and leave rows{k} empty
    end
end

% ------------------------------------------------------------------------
% 4. Write collected rows to temporary CSV
% ------------------------------------------------------------------------
fid = fopen(tmpFile,'w');
header = ["run_id" "U_m" "f_hz" "EI_Nm2" "motor_pos_m" "motor_inertia_kgm" ...
          "domain_depth_m" "L_raft_m" "n" "M" "BC"];
fprintf(fid, "%s\n", strjoin(header,','));

for k = 1:nRuns
    if ~isempty(rows{k})
        fprintf(fid, "%s\n", rows{k});
    end
end

fclose(fid);

% ------------------------------------------------------------------------
% 5. Move tmp ? final
% ------------------------------------------------------------------------
movefile(tmpFile, manifestFile, 'f');
fprintf('Manifest regenerated at %s (%d valid runs)\n', manifestFile, sum(~cellfun(@isempty,rows)));
end
