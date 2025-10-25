nn = 10;
slope = zeros(1, nn);
intercept = zeros(1, nn);
idx = length(eta);
for jj = 1:nn
    
    p = polyfit(z((end-30):end), log(abs(w((end-30):end, 2*jj)))', 1);
    slope(jj) = p(1);
    intercept(jj) = p(2);
end

%% Now for the wave number in the x direction
% Horizontal velocity cosine fit
[kg, phig,ag,bg,~,~] = fit_cosine(x(1:floor(end/4)),real(u(end, 1:floor(end/4))),90);
ufit = ag * cos(kg * x(1:floor(end/4)) + phig);

figure;
plot(x(1:floor(end/4)), real(u(end, 1:floor(end/4))), 'DisplayName', 'Measured u');
hold on;
plot(x(1:floor(end/4)), ufit, 'DisplayName', 'Cosine fit');
xlabel('x (m)', 'FontSize', 14);
ylabel('u(x) (m/s)', 'FontSize', 14);
title('Fitted horizontal velocity wave along raft', 'FontSize', 16);
legend show; set(gca, 'FontSize', 14);

% Surface elevation cosine fit
[kg2, phig2,ag2,bg2,~,~] = fit_cosine(x(1:floor(end/4)),real(eta(1:floor(end/4))),90);
etafit = ag2 * cos(kg2 * x(1:floor(end/4)) + phig2);

figure;
plot(x(1:floor(end/4)), real(eta(1:floor(end/4))), 'DisplayName', 'Measured \eta');
hold on;
plot(x(1:floor(end/4)), etafit, 'DisplayName', 'Cosine fit');
xlabel('x (m)', 'FontSize', 14);
ylabel('\eta(x) (m)', 'FontSize', 14);
title('Fitted surface elevation wave along raft', 'FontSize', 16);
legend show; set(gca, 'FontSize', 14);

% Exponential decay of vertical velocity
ufit = exp(real(args.k) * z) .* real(u(end, idx));
sign = 1;
if all(ufit < 0);  sign = -1; end

figure;
semilogy(z, abs(real(u(:, idx))), 'DisplayName', 'Simulated u');
hold on;
semilogy(z, sign* ufit, 'DisplayName', 'Theoretical exp(kz)');
xlabel('z (m)', 'FontSize', 14);
ylabel('|u(z)| (m/s)', 'FontSize', 14);
title('Comparison of vertical velocity decay with theory', 'FontSize', 16);
legend show; set(gca, 'FontSize', 14);


%% Calculating longet'higgins

I  = trapz(z, abs(u(:, idx)).^2)/2;
I2 = abs(u(end, idx))^2 / (4* real(args.k)) - ...
    0*args.sigma / args.rho *(1 - 1/4* ag2^2 * args.k^2);
I3 = abs(eta(idx))^2 * args.omega^2 / (4 * real(args.k)) -...
    0* 1/4 * args.sigma / args.rho * ag2^2 * args.k^2;

