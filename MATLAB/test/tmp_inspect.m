D = load('data/sweepOmegaMotorPosition.mat'); S = D.S;
if istable(S), S = table2struct(S); end
disp('--- top-level fields ---');
disp(fieldnames(S(1)));
disp('--- args fields ---');
disp(fieldnames(S(1).args));
