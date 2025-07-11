
%% We call the surferbot
[U, x, z, phi, eta, args] = flexible_surferbot('omega', 2*pi*20); %, 'motor_inertia', 1e-10);

close all;
plot(x, real(eta), 'b'); hold on;
plot(x(args.x_contact), real(eta(args.x_contact)), 'r', 'LineWidth', 2)
set(gcf, 'Position', [52 557 1632 420]);
set(gca, 'FontSize', 16);
xlabel('x (m)'); xlim([-.1, .1]);
ylabel('y (m)'); ylim([-1e-3, 1e-3]);
quiver(x(args.x_contact), real(eta(args.x_contact))', zeros(1, sum(args.x_contact)), args.loads'/3e+4, 0);

figure(2); hold on;
set(gca, 'FontSize', 20);
plt = pcolor(x', z, imag(phi));
set(gcf, 'Position', [56 49 1638 424]);
set(plt, 'edgecolor', 'none')

fprintf("Velocity is %g mm/s\n", U*1000)


%% Checking that solution satisfies PDE
ooa  = args.ooa;
[Dxx, Dzz] = getNonCompactFDmatrix2D(args.N, args.M, args.dx, args.dz, 2, ooa);

Lapl = Dxx + Dzz;

bulkIdx = false(args.M, args.N); bulkIdx(2:(end-1), 2:(end-1)) = true; bulkIdx = reshape(bulkIdx, [], 1);
lapl = Lapl * phi(:); lapl = lapl(bulkIdx);
fprintf("Norm of laplacian: %g\n", norm(lapl));

[Dx, Dz] = getNonCompactFDmatrix2D(args.N, args.M, args.dx, args.dz, 1, ooa);
I_NM = speye(args.N * args.M);
bernoulli = Dz - args.sigma/(args.rho * args.g) * Dxx * Dz - args.omega^2/args.g * I_NM - 4 * 1i * args.nu * args.omega / (args.g * args.L_raft) * Dxx;
bernoulli = bernoulli(1:args.M:end, :);
fprintf("Norm of bernoulli: %g\n", norm(bernoulli * phi(:)));


noPenetration = Dz * phi(:); bottomIdx = false(args.M, args.N); bottomIdx(end, 2:(end-1)) = 1; bottomIdx = bottomIdx(:);
fprintf("Norm of no penetration: %g\n", norm(Dz(bottomIdx, :) * phi(:)));
