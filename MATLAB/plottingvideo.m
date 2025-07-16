omega = 2*pi*20;
tvec = linspace(0, 10*pi/omega, 200);


myVideo = VideoWriter('makingwaves','MPEG-4'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)
[U, x, z, phi, eta, args] = flexible_surferbot_v2('omega', omega); %, 'EI', 3.0e9 * 3e-2 * 1e-4^3 / 12 * 100, 'nu', 0);%, 'motor_position', 0); %, 'motor_inertia', 1e-10);
scaleY = 1e+6; scaleX = 1e+2;
fig2=figure(3);
set(gcf, 'Position', [52 557 1632 420]);


for kk=1:length(tvec)
    
    plot(x*scaleX, real(eta .* exp(1i * omega * tvec(kk))) * scaleY, 'b', 'LineWidth', 2); hold on;
    set(gca, 'FontSize', 16);
    plot(x(args.x_contact) * scaleX, real(eta(args.x_contact) * exp(1i * omega * tvec(kk))) * scaleY, 'r', 'LineWidth', 3)
    xlabel('x (cm)'); xlim([-.1, .1] * scaleX);
    ylabel('y (um)'); ylim([-2e+8/scaleY, 2e+8/scaleY]);
    title(sprintf('t = %.2g s', tvec(kk)), 'FontSize', 16); pause(0.01);
    
    hold off;
    %quiver(x(args.x_contact), (real(eta(args.x_contact))') * scale, zeros(1, sum(args.x_contact)), ...
    %    args.loads'/5e+4 * scale, 0, 'MaxHeadSize', 1e-6);

% 
%     boat=real(zeta*exp(1i*2*pi*freq*tvec(kk)) +xs*theta*exp(1i*2*pi*freq*tvec(kk))  );
%     boat(abs(xs)>1/2)=NaN;
%     bb = 1000;
%     plot(xs*L*1e2+Udim*tvec(kk),real(eta*exp(1i*2*pi*freq*tvec(kk)))*L*1e6,'r-','linewidth',2)
%     hold on
%     plot(xs*L*1e2+Udim*tvec(kk),boat*L*1e6,'b','linewidth',4)
% 
%     xlim([-xd,xd]*L*1e2)
%     ylim([-1,1]*2000)
%     xlabel('$$x$$ (cm)','interpreter','latex')
%     ylabel('$$h$$ ($$\mu$$m)','interpreter','latex')
%     set(gca,'fontsize',20)
%     set(fig2,'position',[1000 900 500 500])
%     ylim([-bb,bb])
     frame = getframe(gcf); %get frame
     writeVideo(myVideo, frame);
%     hold off

end
close(myVideo)