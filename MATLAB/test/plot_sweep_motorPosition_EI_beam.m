function plot_sweep_motorPosition_EI_beam(saveDir, export, dataFile)
%PLOT_SWEEP_MOTORPOSITION_EI_BEAM Plot x_M-EI figures using beam-end eta.

if nargin < 1, saveDir = 'data'; end
if nargin < 2, export = false; end
if nargin < 3 || isempty(dataFile), dataFile = 'sweepMotorPositionEI.mat'; end

plot_sweep_motorPosition_EI(saveDir, export, dataFile, 'beam');
