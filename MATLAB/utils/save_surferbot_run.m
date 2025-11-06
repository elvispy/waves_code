function run_id = save_surferbot_run(outdir_base, varargin)
% SAVE_SURFERBOT_RUN
%
% Runs surferbot simulation and saves results to timestamped directory.
% Streamlined version without figure/movie generation for batch processing.
%
% Usage:
%   run_id = save_surferbot_run(outdir_base, 'param1',value1, ...)
%
% Requirements:
%   - flexible_surferbot_v2 function
%   - System Python (optional, for JSON formatting)
%
% Main steps:
%   1. Create unique timestamped output directory
%   2. Run flexible_surferbot_v2 simulation
%   3. Save raw results as MAT file
%   4. Export configuration as formatted JSON
%   5. Attempt JSON pretty-printing via system Python
%
% Inputs:
%   - outdir_base: Base output directory path
%   - varargin: Parameter name-value pairs for simulation
%
% Outputs:
%   - run_id: Path to created run directory
%   - results.mat: Simulation data (U,x,z,phi,eta,args)
%   - config.json: Formatted simulation parameters
%
% Notes:
%   - Uses random suffix for unique directory names
%   - Python pretty-printing is optional fallback
%   - No visualization - use plot_surferbot_run for figures
% ---------- 0.  Time-stamped subfolder -------------------------------
timestamp = datestr(now, 'yyyy-mm-ddTHH-MM-SSZ');
% ---------- 0.  Unique run folder ------------------------------------
run_id = fullfile(outdir_base, ...
            [char(datetime('now','Format','yyyy-MM-dd''T''HH-mm-ss_')), num2str(randi(10000))]);
mkdir(run_id);

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
    cmd = sprintf('python3 -m json.tool "%s" > "%s" && move /Y "%s" "%s"', ...
                    jsonFile,tmpFile,tmpFile,jsonFile);
else
    cmd = sprintf('python3 -m json.tool "%s" > "%s" && mv "%s" "%s"', ...
                    jsonFile,tmpFile,tmpFile,jsonFile);
end
if system(cmd) ~= 0
    warning('Pretty-print skipped (system Python not found).');
end

fprintf('Saved run %s\n', timestamp);

end
