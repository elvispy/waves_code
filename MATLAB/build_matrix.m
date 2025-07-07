function A = build_matrix(dx, dz, N, M, H, coeffs, sparse)
    if sparse == false
        % Operators
        d_dx = Diff(dx, 1, dz, 0, [N, M]);
        d_dz = Diff(dx, 0, dz, 1, [N, M]);

        % I_NM tensor: identity in (i,j)-(k,l)
        I_NM = zeros(N, M, N, M);
        for i = 1:N
            for j = 1:M
                I_NM(i, j, i, j) = 1;
            end
        end

        % Build equations
        % E1: Bernoulli
        E1 = zeros(N, M, N, M, 5);
        for d = 1:5
            if d == 1
                E1(:, :, :, :, d) = coeffs.C11 * d_dz + coeffs.C13 * I_NM;
            elseif d == 3
                E1(:, :, :, :, d) = coeffs.C12 * d_dz + coeffs.C14 * I_NM;
            end
        end

        % Remove contact equations on surface
        E1(~x_free, 1, :, :, :) = 0;
        E1(:, 2:end, :, :, :) = 0;

        % Euler beam equation
        dz2 = Diff(dx, 0, dz, 1, [H M]); I_HM = reshape(eye(H*M), [H M H M]);
        E12 = zeros(H, M, H, M, 5);
        for d  = 1:5
            if d == 1
                E12(:, :, :, :, d) = coeffs.C22 * dz2 + coeffs.C24 * I_HM + coeffs.C25 * dz2;
            elseif d == 3
                E12(:, :, :, :, d) = coeffs.C26 * I_HM;
            elseif d == 5
                E12(:, :, :, :, d) = coeffs.C21 * dz2;
            end
        end
        E1(x_contact, :, x_contact, :, :) = E12;

        % Surface tension term
        dxdz = Diff(dx, 1, dz, 1, [sum(x_free)/2 M]);
        %dxdz = dxdz(1, 1, :, :);
        E1(right_raft_boundary, 1, x_free & x > 0, :, 1) = coeffs.C27/dx * dxdz(1,   1, :, :);
        E1(left_raft_boundary,  1, x_free & x < 0, :, 1) = coeffs.C27/dx * dxdz(end, 1, :, :);

        % E2: Laplace
        E2 = zeros(N, M, N, M, 5);
        lap = Diff(dx, 2, dz, 0, [N M]) + Diff(dx, 0, dz, 2, [N M]);
        E2(:, :, :, :, 1) = lap;
        E2([1, end], :, :, :, :) = 0;
        E2(:, [1, end], :, :, :) = 0;
        %E2_full = zeros(N, M, N, M, 5);
        %E2_full(:, :, :, :, 1) = E2;

        % E3: Bottom BC
        E3 = zeros(N, M, N, M, 5);
        E3(:, end, :, :, 1) = d_dz(:, end, :, :);
        E3([1, end], :, :, :) = 0;

        % E4: Radiative BCs
        E4 = zeros(N, M, N, M, 5);
        E4(1, :, 1, :, 1) = -coeffs.C32;
        E4(1, :, 1, :, 2) =  coeffs.C31;
        E4(end, :, end, :, 1) = coeffs.C32;
        E4(end, :, end, :, 2) = coeffs.C31;

        % Final operator
        A = zeros(5, N, M, N, M, 5);
        A(1, :, :, :, :, :) = E1 + E2 + E3 + E4;
        for d = 1:4
            A(d+1, :, :, :, :, d) = d_dx; 
            A(d+1, :, :, :, :, d+1) = -I_NM;
        end
    else
        
    end

end