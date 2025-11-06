% plottingvideo.m
%
% Description:
%   Generates an animation of the flexible surferbot’s surface motion over time 
%   and saves it as an MPEG-4 video file named "makingwaves.mp4".
%
% Usage:
%   Run this script from within the project’s /scripts or /examples directory.
%   It requires the function FLEXIBLE_SURFERBOT_V2 to be accessible in ../src/.
%
% Functionality:
%   1. Adds the source code folder to the MATLAB path.
%   2. Defines an excitation frequency (omega) and a corresponding time vector.
%   3. Calls FLEXIBLE_SURFERBOT_V2 to compute displacement fields and geometry.
%   4. Scales and plots the surface deflection η(x, t) over time.
%   5. Highlights the contact points of the surferbot in red.
%   6. Captures each frame and compiles them into a video file.
%
% Inputs:
%   (Specified inside the script)
%     omega   – Angular frequency of oscillation [rad/s].
%     EI      – Bending stiffness parameter used by FLEXIBLE_SURFERBOT_V2.
%
% Outputs:
%   - A video file "makingwaves.mp4" stored in the current working directory.
%
% Notes:
%   - Adjust 'myVideo.FrameRate' to control animation smoothness.
%   - Modify 'scaleX' and 'scaleY' to change axis scaling for visualization.
%   - The video captures approximately 10 cycles of motion at 20 Hz.


addpath '../src/'

omega = 2*pi*20;
tvec = linspace(0, 10*pi/omega, 200);


myVideo = VideoWriter('makingwaves','MPEG-4'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)
[U, x, z, phi, eta, args] = flexible_surferbot_v2('omega', omega, 'EI', 3.0e+9 * 3e-2 * 1e-4^3 / 12 * 10); %, 'motor_inertia', 1e-10);
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

     frame = getframe(gcf); %get frame
     writeVideo(myVideo, frame);
%     hold off

end
close(myVideo)