nn = 10;
slope = zeros(1, nn);
intercept = zeros(1, nn);
for jj = 1:nn
    
    p = polyfit(z(3*floor(end/4):end), log(abs(w(3*floor(end/4):end, 2*jj)))', 1);
    slope(jj) = p(1);
    intercept(jj) = p(2);
end

%% Now for the wave number in the x direction
[kg, phig,ag,bg,~,~] = fit_cosine(x(1:floor(end/4)),real(u(end, 1:floor(end/4))),90);
ufit = ag * cos(kg * x(1:floor(end/4)) + phig);

figure;
plot(x(1:floor(end/4)), real(u(end, 1:floor(end/4)))); hold on;
plot(x(1:floor(end/4)), ufit);

%% Wave number for the eta variable
[kg2, phig2,ag2,bg2,~,~] = fit_cosine(x(1:floor(end/4)),real(eta(1:floor(end/4))),90);

etafit = ag2 * cos(kg2 * x(1:floor(end/4)) + phig2);
figure;
plot(x(1:floor(end/4)), real(eta(1:floor(end/4)))); hold on;
plot(x(1:floor(end/4)), etafit);

%% Now comparing exponential decay
ufit = exp(args.k * z) .* real(u(end, 1));

figure
semilogy(z, real(u(:, 1))); hold on;
semilogy(z, ufit);


%% Calculating longet'higgins

I  = trapz(z, abs(u(:, 1)).^2)/2;
I2 = abs(u(end, 1))^2 / (4* args.k);
I3 = abs(eta(1))^2 * args.omega^2 / (4 * args.k);

