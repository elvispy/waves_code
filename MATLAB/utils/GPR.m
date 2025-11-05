addpath ./src

%% 1) Configure decision variables and constants
decision = [ ...
 struct('name','omega'         ,'range',[5,100]           ,'transform','none')
 struct('name','motor_position','range',[0,0.05/2]        ,'transform','none')
 struct('name','EI'            ,'range',[1e-1 10] * 1e-4*3.0e9 * 3e-2 * (9.9e-4)^3 / 12,'transform','none')];

fixed = struct( ...
  'sigma',72.2e-3, 'rho',1000.0, 'nu',0*1.0e-6, 'g',9.81, ...
  'L_raft',0.05, 'L_domain',5*0.05, 'd',0.03, ...
  'rho_raft',0.052, 'motor_inertia',0.13e-3*2.5e-3, ...
  'BC','radiative');

%% 2) Choose the metric and sense
% Metric is a function of model output and parameters. Replace as needed.
metric = @(out,params) out.U;     % e.g., minimize thrust U
maximize = false;                  % set true to maximize (the code flips sign)

%% 3) Build optimizableVariable array from config
vars = arrayfun(@(d) optimizableVariable(d.name,d.range, ...
    'Transform',d.transform), decision, 'UniformOutput', false);
vars = [vars{:}];

%% 4) Generic objective built from config
obj = makeObjective(@flexible_surferbot_v2, fixed, metric, maximize);

%% 5) Run BO
results = bayesopt(obj, vars, ...
  'MaxObjectiveEvaluations', 100, ...
  'IsObjectiveDeterministic', true, ...
  'AcquisitionFunctionName','expected-improvement-plus', ...
  'UseParallel', true, ...
  'PlotFcn', {@plotMinObjective,@plotObjectiveModel,@plotAcquisitionFunction});

bestX = results.XAtMinObjective;      % table of best variables
bestF = results.MinObjective;

% Merge best variables with fixed params for a final run if you want:
bestParams = fixed;
fn = bestX.Properties.VariableNames;
for i=1:numel(fn), bestParams.(fn{i}) = bestX{1,fn{i}}; end
[out_best] = flexible_surferbot_v2(struct2nv(bestParams){:});

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
  out = modelFcn(nv{:});
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
