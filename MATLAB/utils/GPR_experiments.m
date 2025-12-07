% BO_SURFERBOT.M
% Purpose:
%   Bayesian optimization wrapper for FLEXIBLE_SURFERBOT_V2. Searches over
%   omega, motor_position, and EI to optimize a user-defined metric.
%
% How to run:
%   - Ensure ./src contains FLEXIBLE_SURFERBOT_V2 and its dependencies.
%   - Requires Statistics and Machine Learning Toolbox (bayesopt).
%   - Edit the 'metric' handle and 'maximize' flag as needed.
%   - Run the script; it launches bayesopt with plots and parallel evals.
%
% What it does:
%   1) Declares decision bounds (omega [5,100], motor_position [0, L/2], EI range).
%   2) Defines fixed physical/geometry parameters and BCs.
%   3) Builds an objective that calls FLEXIBLE_SURFERBOT_V2 with name-value pairs,
%      computes metric(out, params), and flips sign if maximizing.
%   4) Runs bayesopt for up to 100 evaluations with EI+ acquisition.
%
% Outputs:
%   - 'results': bayesopt object (inspect Results and plots).
%   - 'bestX', 'bestF': best decision variables and objective value.
%   - 'bestParams': fixed+best variables for a final model run.
%
% Customize:
%   - Replace 'metric = @(out,params) out.U' with any scalar metric.
%   - Adjust 'MaxObjectiveEvaluations', acquisition, or parallel settings.



addpath '../src'
warning('off','all');


L_raft = 0.015;
E      = 330e3;     % Pa
h      = 0.0015;     % m, thickness
d      = 0.003;     % m, depth
EI0    = E * d * h^3/12;
omega0 = 2*pi*10;   % rad/s (reference)
rho_oomoo = 1.34e3;   % kg/m3



%% 1) Configure decision variables and constants
decision = [ ... %struct('name','L_raft'         ,'range',[L_raft/5 L_raft],'transform','none')
 struct('name','motor_position' ,'range',0.95*[-L_raft/2,L_raft/2]  ,'transform','none')
 struct('name','omega'          ,'range',2*pi*[5 40],  'transform','none') %struct('name','d'              ,'range',[d/10 d], 'transform','none')
 ];


fixed = struct( ...
  'sigma',72.2e-3, 'rho',1000.0, 'nu',0*1.0e-6, 'g',9.81, ...
  'EI', E * d * h^3/12, ...
  'rho_raft', rho_oomoo * d * h, 'motor_force',50e-6, ...
  'BC','radiative');

%% 2) Choose the metric and sense
% Metric is a function of model output and parameters. Replace as needed.

alpha = -10;
beta = -1;
metric = @(out,params) -(abs(out.eta(1)) + abs(out.eta(end)))/out.args.L_raft;
%alpha * (abs(out.eta(1)) - abs(out.eta(end)))^2 /out.args.L_raft^2 ...
%     + beta *(abs(out.eta(1)) + abs(out.eta(end))) /out.args.L_raft; %  ...
    %/ (abs(out.eta(1))^2 + abs(out.eta(end))^2) * abs(out.U);     % e.g., minimize thrust U

% Smooth scalar BO function
%epsilon = 1;
%beta = 100;
%metric = @(out, params) - abs(out.eta(end)) / (abs(out.eta(end)) + beta) * ...
%    exp(-abs(out.eta(1))/epsilon);
    
maximize = false;                  % set true to maximize (the code flips sign)

%% 3) Build optimizableVariable array from config
vars = arrayfun(@(d) optimizableVariable(d.name,d.range, ...
    'Transform',d.transform), decision, 'UniformOutput', false);
vars = [vars{:}];

%% 4) Generic objective built from config
obj = makeObjective(@flexible_surferbot_v2, fixed, metric, maximize);

%% 5) Run BO
results = bayesopt(obj, vars, ...
  'MaxObjectiveEvaluations', 200, ...
  'IsObjectiveDeterministic', true, ...
  'ExplorationRatio', 0.4, ...
  'AcquisitionFunctionName','expected-improvement-plus', ...
  'UseParallel', true, ...
  'PlotFcn', {@plotMinObjective,@plotObjectiveModel,@plotAcquisitionFunction});

bestX = results.XAtMinObjective;      % table of best variables
bestF = results.MinObjective;

% Merge best variables with fixed params for a final run if you want:
bestParams = fixed;
fn = bestX.Properties.VariableNames;
for i=1:numel(fn), bestParams.(fn{i}) = bestX{1,fn{i}}; end
args = struct2nv(bestParams);
[out_best] = flexible_surferbot_v2(args{:});

warning('on','all');


%% ---- Helpers ----
function obj = makeObjective(modelFcn, fixed, metric, maximize)
  obj = @(tbl) inner(tbl, modelFcn, fixed, metric, maximize);
end


function f = inner(tbl, modelFcn, fixed, metric, maximize)
    % 1. Merge table overrides into fixed struct
    p = fixed;
    o = table2struct(tbl,'ToScalar',true);
    n = fieldnames(o); 
    for k=1:numel(n), p.(n{k}) = o.(n{k}); end
    
    % 2. Check Geometry Constraints
    % If the motor is off the edge of the raft, return a bad penalty (Inf)
    % instead of crashing the physics engine.
    %if p.L_raft/2 < abs(p.motor_position)
    %    f = Inf; 
        %return; 
    %end

    % 3. RUN MODEL WITH ERROR TRAPPING
    try
        % Convert struct to Name-Value pairs for the function call
        nv = struct2nv(p);
        
        % Call the physics model
        [U,~,~,~,eta,args] = modelFcn(nv{:});
        
        % Calculate metric
        out = struct('U',U,'eta',eta,'args',args);
        val = metric(out, p);
        
        % Handle Maximization
        if maximize, val = -val; end
        f = val;

    catch ME
        % --- ERROR HANDLING ---
        fprintf(2, '\n======================================\n');
        fprintf(2, 'CRASH DETECTED inside flexible_surferbot_v2\n');
        fprintf(2, 'Error Message: %s\n', ME.message);
        fprintf(2, 'Line: %d in %s\n', ME.stack(1).line, ME.stack(1).name);
        
        fprintf('\n--- Parameters causing crash ---\n');
        disp(p); 
        
        fprintf('Pausing execution. Check variables "p" or "ME" now.\n');
        fprintf('Type "dbcont" to ignore and continue, or "dbquit" to stop.\n');
        
        keyboard; % <--- MATLAB WILL STOP HERE
        
        f = NaN; % Tell bayesopt this run failed
    end
end



function nv = struct2nv(s)
  names = fieldnames(s);
  vals  = struct2cell(s);
  % Convert chars to string scalars to keep name-value pairs valid
  for i=1:numel(vals)
    if ischar(vals{i}), vals{i} = string(vals{i}); end
  end
  nv = reshape([names.'; vals.'], 1, []);  % {'name1',val1,'name2',val2,...}
end
