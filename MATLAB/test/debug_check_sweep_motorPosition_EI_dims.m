Dat = load('data/sweepMotorPositionEI.mat'); St = Dat.S;
if istable(St), St = table2struct(St); end
mp_list = unique([St.motor_position]);
EI_list = unique([St.EI]);
fprintf('n_mp=%d, n_EI=%d\n', numel(mp_list), numel(EI_list));
asymmetry = randn(numel(mp_list), numel(EI_list));
% contourc wants: X = cols of Z, Y = rows of Z
% Z is (n_mp, n_EI), so X = EI_list (n_EI), Y = mp_list (n_mp)
fprintf('asymmetry size: %d x %d\n', size(asymmetry));
fprintf('EI_list: %d, mp_list: %d\n', numel(EI_list), numel(mp_list));
% This should work: contourc(EI_list, mp_list, asymmetry, [0 0])
% No transpose needed since asymmetry is already (n_mp, n_EI)
