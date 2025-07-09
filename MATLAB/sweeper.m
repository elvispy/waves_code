
%% We call the surferbot
[U, x, z, phi, eta, args] = flexible_surferbot('motor_position', 0, 'omega', 20*pi*20, 'EI', 1e-4); %, 'motor_inertia', 1e-10);

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
plt = pcolor(x', z, imag(phi)');
set(gcf, 'Position', [56 49 1638 424]);
set(plt, 'edgecolor', 'none')