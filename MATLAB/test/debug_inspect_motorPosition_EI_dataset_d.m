% Check d value in both datasets
D1 = load('data/sweepMotorPositionEI.mat'); S1 = D1.S;
if istable(S1), S1 = table2struct(S1); end
fprintf('sweepMotorPositionEI.mat:  d = %g\n', S1(1).args.d);

D2 = load('data/sweepMotorPositionEI2.mat'); S2 = D2.S;
if istable(S2), S2 = table2struct(S2); end
fprintf('sweepMotorPositionEI2.mat: d = %g\n', S2(1).args.d);
