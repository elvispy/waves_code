function simple_mode_video()
    % Parameters
    L = 5;
    omega = 5;          
    vid_len_sec = 8;
    fps = 30;
    vidFile = 'figures/simple_mode_video.mp4';
    
    % Vertical Offsets
    Y_OFFSET_1   =  3.2;  
    Y_OFFSET_2   =  -0.3;    
    Y_OFFSET_SUM = -3.8;  
    
    % Grid
    N_points = 500;
    x_calc = linspace(0, L, N_points);
    x_plot = x_calc - L/2; 
    
    % --- Physics Calculations (Same as before) ---
    betaL_1 = 4.73004074; 
    W1 = get_free_free_mode(x_calc, L, betaL_1);
    W1 = W1 / max(abs(W1)); 
    
    betaL_2 = 7.85320462;
    W2 = get_free_free_mode(x_calc, L, betaL_2);
    W2 = W2 / max(abs(W2)); 
    
    idx_raft = 1:round(N_points/2);
    numerator = sum(W1(idx_raft) .* W2(idx_raft));
    denominator = sum(W2(idx_raft) .^ 2);
    alpha_opt = -numerator / denominator;
    
    W_super = W1 + alpha_opt * W2;
    
    % --- Video Setup ---
    vid = VideoWriter(vidFile, 'MPEG-4');
    vid.Quality = 100;
    vid.FrameRate = fps;
    open(vid);
    
    N_frames = vid_len_sec * fps;
    tvec = linspace(0, vid_len_sec * 2*pi/omega, N_frames);
    
    % --- Graphics Setup ---
    fig = figure('Position', [200 200 800 800], 'Color', 'w');
    ax = gca;
    hold(ax, 'on');
    
    % 1. Reference lines
    yline(Y_OFFSET_1,   '-', 'Color', [0.9 0.9 0.9], 'LineWidth', 1);
    yline(Y_OFFSET_2,   '-', 'Color', [0.9 0.9 0.9], 'LineWidth', 1);
    yline(Y_OFFSET_SUM, '-', 'Color', [0.9 0.9 0.9], 'LineWidth', 1);
    
    % 2. ADD MATH SYMBOLS
    % Position: x=0 (center), y=midpoint between layers
    mid_1_2 = (Y_OFFSET_1 + Y_OFFSET_2) / 2;
    mid_2_sum = (Y_OFFSET_2 + Y_OFFSET_SUM) / 2;
    
    text(0, mid_1_2, '+', ...
        'FontSize', 75, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2], ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
    text(0, mid_2_sum, '=', ...
        'FontSize', 75, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2], ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

    % 3. Plot Objects
    hF1 = plot(x_plot, nan(size(x_plot)), 'b--', 'LineWidth', 2);
    hF2 = plot(x_plot, nan(size(x_plot)), 'r--', 'LineWidth', 2);
    hSum = plot(x_plot, nan(size(x_plot)), 'k-', 'LineWidth', 5);
    
    % --- Styling ---
    axis off;               
    ylim([Y_OFFSET_SUM - 2.5, Y_OFFSET_1 + 2.5]);           
    xlim([-L/2 L/2]);      
    set(ax, 'LooseInset', [0 0 0 0]); 
    set(ax, 'Position', [0 0 1 1]);   
    
    legend([hF1, hF2, hSum], 'Mode 1', 'Mode 2', 'Superposition', ...
           'Location', 'northwest', 'Box', 'off', 'FontSize', 36);

    % --- Loop ---
    for k = 1:N_frames
        t = tvec(k);
        time_mod = cos(omega * t);
        
        y1_vals   = (W1 * time_mod) + Y_OFFSET_1;
        y2_vals   = (alpha_opt * W2 * time_mod) + Y_OFFSET_2;
        ySum_vals = (W_super * time_mod) + Y_OFFSET_SUM;
        
        set(hF1, 'YData', y1_vals);
        set(hF2, 'YData', y2_vals);
        set(hSum, 'YData', ySum_vals);
        
        drawnow limitrate
        frame = getframe(fig);
        writeVideo(vid, frame);
    end
    
    close(vid);
    close(fig);
    fprintf('Video saved: %s\n', vidFile);
end

function W = get_free_free_mode(x, L, betaL)
    beta = betaL / L;
    bx = beta * x;
    numerator = sin(betaL) - sinh(betaL);
    denominator = cosh(betaL) - cos(betaL);
    alpha = numerator / denominator;
    W = (sin(bx) + sinh(bx)) + alpha .* (cos(bx) + cosh(bx));
end