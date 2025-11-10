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

L_raft = 0.05; % Size of surferbot

%% 1) Configure decision variables and constants
decision = [ ...
 struct('name','omega'         ,'range',[10,90]           ,'transform','none')
 struct('name','motor_position','range',[-L_raft/2,L_raft/2]        ,'transform','none')
 struct('name','EI'            ,'range',[5e-3 10] * 3.0e9 * 3e-2 * (9.9e-4)^3 / 12, 'transform','log')];


fixed = struct( ...
  'sigma',72.2e-3, 'rho',1000.0, 'nu',1.0e-6, 'g',9.81, ...
  'L_raft',L_raft,  'd',0.03, ...
  'rho_raft',0.052, 'motor_inertia',0.13e-3*2.5e-3, ...
  'BC','radiative');

%% 2) Choose the metric and sense
% Metric is a function of model output and parameters. Replace as needed.
metric = @(out,params) -abs(out.U) * abs(out.args.thrust)/abs(out.args.power);%(abs(out.eta(1))^2 - abs(out.eta(end))) ...
    %/ (abs(out.eta(1))^2 + abs(out.eta(end)));%- abs(out.U);     % e.g., minimize thrust U
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
  'ExplorationRatio', 0.65, ...
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



%% ---- Helpers ----
function obj = makeObjective(modelFcn, fixed, metric, maximize)
  obj = @(tbl) inner(tbl, modelFcn, fixed, metric, maximize);
end

function f = inner(tbl, modelFcn, fixed, metric, maximize)
  % merge table overrides into fixed struct
  p = fixed;
  o = table2struct(tbl,'ToScalar',true);
  n = fieldnames(o); for k=1:numel(n), p.(n{k}) = o.(n{k}); end
  % call model with name-value pairs
  nv = struct2nv(p);
  [U,x,z,phi,eta,args] = modelFcn(nv{:});
  out = struct('U',U,'eta',eta,'args',args);
  val = metric(out, p);
  if maximize, val = -val; end
  f = val;
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
