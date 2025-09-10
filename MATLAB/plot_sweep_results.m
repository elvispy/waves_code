function plot_sweep_results
% ------------------------------------------------------------------------
%  Reads the manifest, calculates a characteristic frequency, and plots
%  U vs. a non-dimensional frequency (omega / omega_c).
% ------------------------------------------------------------------------

% ---------- 1. Define File Paths ----------------------------------------
outdir = 'surferbot_results';
manifestFile = fullfile(outdir, 'manifest.csv');

if ~isfile(manifestFile)
    error('Manifest file not found: %s\nDid you run a sweep first?', manifestFile);
end

% ---------- 2. Read Manifest and Supporting Data ------------------------
fprintf('Reading data from %s...\n', manifestFile);
T = readtable(manifestFile);

% To get m_c and L_c, we must load the 'args' struct from a run file,
% as they are not in the manifest. We'll use the first run.
firstRunID = T.run_id{1};
resultFile = fullfile(outdir, firstRunID, 'results.mat');

if ~isfile(resultFile)
    error('Could not find results file for run %s to load args.', firstRunID);
end

fprintf('Loading characteristic parameters from %s...\n', resultFile);
S = load(resultFile, 'args');
m_c = S.args.m_c; % [kg] characteristic mass 
L_c = S.args.L_c; % [m] characteristic length 
% ---------- 3. Create the Plot ------------------------------------------
unique_EI = unique(T.EI_Nm2);

figure;
hold on;

% Loop through each unique bending stiffness (EI)
for i = 1:numel(unique_EI)
    current_EI = unique_EI(i);
    
    % Select data subset for the current EI
    subset = T(T.EI_Nm2 == current_EI, :);
    subset = sortrows(subset, 'f_hz');
    
    % --- Non-dimensionalization ---
    % Calculate the characteristic angular frequency for this EI
    omega_c = sqrt(current_EI / (m_c * L_c));
    
    % Get the driving angular frequency from the manifest's frequency in Hz
    omega_driving = subset.f_hz * (2*pi);
    
    % Calculate the non-dimensional frequency ratio
    nondim_freq = omega_driving / omega_c;
    
    % Plot U vs. the non-dimensional frequency
    plot(nondim_freq, subset.U_m, '-o', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('EI = %.2e N·m^2', current_EI));
end

hold off;

% ---------- 4. Add Labels and Title -------------------------------------
title('Surferbot Velocity vs. Non-Dimensional Frequency');
xlabel('Non-Dimensional Frequency (\omega / \omega_c)');
ylabel('Resulting Velocity U (m/s)');
legend('show', 'Location', 'best');
set(gca, 'XScale', 'log');
grid on;
set(gca, 'FontSize', 12);

fprintf('Plot generated successfully.\n');

end