using Surferbot
using LinearAlgebra
using Printf
using QuadGK
using FFTW

# Purpose: Falsify the "Modal Projection of Admittance" a priori law.
# Hypothesis: ma,n = (rho * d / (2*pi)) * Integral( |FT(psi_n)|^2 / (k * tanh(kH)) dk ) / Integral( psi_n^2 dx )

function freefree_mode_shape(xi::AbstractVector{<:Real}, L::Real, betaL::Real)
    beta = betaL / L
    bx = beta .* collect(float.(xi))
    alpha = (sin(betaL) - sinh(betaL)) / (cosh(betaL) - cos(betaL))
    psi = (sin.(bx) .+ sinh.(bx)) .+ alpha .* (cos.(bx) .+ cosh.(bx))
    scale = maximum(abs.(psi))
    return isfinite(scale) && scale > 0 ? psi ./ scale : psi
end

function get_theoretical_man(L, H, d, rho, modes_to_test)
    # Define a high-res spatial grid that is much larger than the beam
    # to get a good Fourier transform
    L_domain = max(10 * L, 5.0)
    N_grid = 4096 * 4
    x = range(-L_domain/2, L_domain/2, length=N_grid)
    dx = step(x)
    
    # Wavenumber grid
    k = fftfreq(N_grid, 2*pi/dx)
    
    # Admittance kernel (avoid division by zero at k=0)
    # The actual DtN for finite depth includes gravity:
    # Actually, the DtN relates phi_z to phi. 
    # Laplace eq: phi_xx + phi_zz = 0 -> phi_zz = k^2 phi
    # phi(z) = A cosh(k(z+H)). phi_z(0) = A k sinh(kH). phi(0) = A cosh(kH).
    # K(k) = k * tanh(kH).
    admittance = zeros(N_grid)
    for i in 1:N_grid
        abs_k = abs(k[i])
        if abs_k > 1e-5
            admittance[i] = 1.0 / (abs_k * tanh(abs_k * H))
        else
            # For very small k, k*tanh(kH) approx k^2 H
            admittance[i] = 1.0 / (1e-5 * tanh(1e-5 * H)) 
        end
    end
    
    ma_n = Float64[]
    
    for n in modes_to_test
        # Construct mode shape on the domain (zero outside beam)
        psi_spatial = zeros(N_grid)
        beam_mask = abs.(x) .<= L/2
        
        if n == 0
            psi_spatial[beam_mask] .= 1.0
        elseif n == 1
            psi_spatial[beam_mask] .= x[beam_mask] ./ (L/2)
        else
            n_elastic = n - 1
            betaL = Surferbot.Modal.freefree_betaL_roots(n_elastic)[n-1]
            # xi ranges from 0 to L
            xi = x[beam_mask] .+ L/2
            psi_spatial[beam_mask] .= freefree_mode_shape(xi, L, betaL)
        end
        
        # Fourier Transform
        psi_k = fft(psi_spatial) .* dx
        
        # Parseval's integral
        # Integral of |psi_k|^2 * admittance dk / (2*pi)
        dk = k[2] - k[1]
        numerator = sum(abs2.(psi_k) .* admittance) * dk / (2*pi)
        
        denominator = sum(psi_spatial.^2) * dx
        
        ma = (rho * d) * (numerator / denominator)
        push!(ma_n, ma)
    end
    
    return ma_n
end

function main()
    println("--- Falsification Experiment ---")
    
    # 1. Pick new, completely different parameters
    L = 0.12 # Longer raft
    domain_depth = 0.25 # Deeper water
    omega = 2 * pi * 40.0 # 40 Hz, completely different frequency
    d = 0.05 # Wider raft
    EI = 5e4 # Stiffer
    rho_raft = 0.1
    rho = 1000.0
    g = 9.81
    
    params = Surferbot.FlexibleParams(
        L_raft = L,
        domain_depth = domain_depth,
        L_domain = 1.5,
        omega = omega,
        d = d,
        EI = EI,
        rho_raft = rho_raft,
        n = 100, # Domain grid cells
        M = 30, # Depth grid cells
        motor_position = L/4,
        motor_force = 1.0
    )
    
    @printf("Setup: L=%.2f m, H=%.2f m, f=%.1f Hz, d=%.2f m\n\n", L, domain_depth, omega/(2pi), d)
    
    modes_to_test = 0:3
    
    # 2. A Priori Estimate of ma,n
    println("1. Calculating A Priori Added Mass (Spectral Admittance)...")
    ma_apriori = get_theoretical_man(L, domain_depth, d, rho, modes_to_test)
    
    for (i, n) in enumerate(modes_to_test)
        @printf("   Mode %d: ma,n = %8.4f kg/m\n", n, ma_apriori[i])
    end
    
    # 3. Perform a completely new solve!
    println("\n2. Running Full Flexible Surferbot Solver...")
    result = Surferbot.flexible_solver(params)
    
    # 4. Extract A Posteriori Qn from the Solver
    # We use our clean ModalDecomposition to project the pressure into the W basis
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    Q_num = modal.Q_w
    q_num = modal.q_w
    
    # 5. Compare A Priori Qn vs Numerical Qn
    # Recall the balance: G D q = Q - F. 
    # Q_apriori approx G * diag(ma,n * omega^2 - d*rho*g) * q_num
    
    # We need the G matrix
    x_raft = collect(range(-L/2, L/2, length=length(modal.x_raft)))
    w_grid = Surferbot.trapz_weights(x_raft)
    raw_grid = Surferbot.Modal.build_raw_freefree_basis(x_raft, L; num_modes=4)
    Phi = raw_grid.Phi
    G = Phi' * (Phi .* w_grid)
    
    hydrostatic = d * rho * g
    
    Q_apriori = zeros(ComplexF64, 4)
    
    println("\n3. Comparing Qn Estimates (A Priori vs A Posteriori)...")
    @printf("%-6s | %-12s | %-12s | %-12s\n", "Mode n", "Re(Q_apriori)", "Re(Q_num)", "Error %")
    println("-"^50)
    
    # We apply the admittance prediction as a diagonal matrix in the basis
    # Because DtN operator was projected mode-by-mode
    M_added_diag = diagm(0 => ma_apriori)
    
    # Q_apriori = G * (omega^2 * M_added_diag - hydrostatic * I) * q_num
    # Wait, the spectral projection means the coupling is M_added_matrix. 
    # If we assume off-diagonals are small, we can use the diagonal M_added_diag
    for i in 1:4
        # Calculate row i of the projection
        q_vec = q_num
        row_val = 0.0 + 0.0im
        for j in 1:4
            # Q_w = G * (fluid_impedance) * q_w
            # fluid_impedance = ma * omega^2 - d*rho*g
            impedance = ma_apriori[j] * omega^2 - hydrostatic
            row_val += G[i, j] * (impedance * q_vec[j])
        end
        Q_apriori[i] = row_val
    end
    
    for (i, n) in enumerate(modes_to_test)
        q_ap = Q_apriori[i]
        q_nu = Q_num[i]
        err = abs(real(q_ap) - real(q_nu)) / max(abs(real(q_nu)), 1e-12) * 100
        @printf("n=%-4d | %-12.4e | %-12.4e | %-8.2f%%\n", n, real(q_ap), real(q_nu), err)
    end
    
    println("\nHypothesis Check:")
    println("If errors are high, the pure spectral admittance is missing a component.")
    println("Possible bottlenecks: 3D edge effects, radiation damping (imaginary part missing), or cross-modal off-diagonal terms in the DtN projection.")
end

main()
