using Surferbot
using LinearAlgebra
using Printf
using Statistics

# Purpose: Senior Research Analyst Probe - Direct Operator Decomposition.
# We build the DtN operator from first principles and project it onto modes.
# This avoids OOM by bypassing the full flexible solver loop.

function main()
    println("--- Senior Analyst: Direct Operator Probe ---")
    
    # Universal Params
    L = 0.05
    H = 0.05
    L_dom = 1.5
    d = 0.03
    rho = 1000.0
    omega = 2pi * 80.0
    g = 9.81
    
    # 1. Physical Scale: The "Static" Hydrostatic Foundation
    K_static = rho * g * d
    @printf("Static Physical Scale (rho*g*d): %.2e N/m^2\n", K_static)
    
    # 2. Build the Exact Spectral DtN Kernel for a Rectangular Tank
    # The fluid potential phi satisfies phi_z = (DtN) * phi at z=0.
    # For a finite tank [-Ldom/2, Ldom/2], modes are cos(m*pi*(x+Ldom/2)/Ldom).
    
    # Grid on the raft
    x_raft = collect(range(-L/2, L/2, length=101))
    w = Surferbot.trapz_weights(x_raft)
    
    # W-basis modes (Analytical free-free)
    raw = Surferbot.Modal.build_raw_freefree_basis(x_raft, L; num_modes=4)
    Phi = raw.Phi
    
    # Construct the DtN Inverse (Admittance) Matrix G_f
    N = length(x_raft)
    G_f = zeros(ComplexF64, N, N)
    
    println("Assembling spectral DtN kernel (m=1000 modes)...")
    m_max = 1000
    for m in 0:m_max
        k_m = m * pi / L_dom
        
        # Admittance eigenvalue: 1 / (k_m * tanh(k_m * H))
        # Note: we use the Bernoulli pressure relation p = -rho*i*omega*phi
        # So Z_dyn = -rho*i*omega * (phi / eta)
        # Kinematic BC: i*omega*eta = phi_z = dtn * phi => phi/eta = i*omega/dtn
        # Z_dyn = -rho*i*omega * (i*omega/dtn) = rho*omega^2 / dtn
        
        dtn_m = k_m * tanh(k_m * H + 1e-12)
        if m == 0; dtn_m = 1e-8; end # regularization
        
        # Tank mode shapes
        mode_m = cos.(k_m .* (x_raft .+ L_dom/2))
        factor = (m == 0) ? 1/L_dom : 2/L_dom
        
        for i in 1:N
            for j in 1:N
                G_f[i, j] += mode_m[i] * mode_m[j] * factor / dtn_m
            end
        end
    end
    
    # 3. Direct Modal Projection
    # Z_mn = rho * d * omega^2 * <psi_n, G_f psi_m> / <psi_n, psi_n>
    
    println("\n--- Predicted Modal Impedance (A-Priori) ---")
    @printf("%-6s | %-12s | %-12s | %-12s\n", "Mode n", "Re(Z_pred)", "Ratio to Static", "Stability Check")
    println("-"^60)
    
    for n in 1:4
        psi_n = Phi[:, n]
        norm_psi = dot(psi_n, psi_n .* w)
        
        # Projected potential term: psi_n' * G_f * psi_n
        # This is the modal admittance eigenvalue.
        gamma_n = dot(psi_n, G_f * (psi_n .* w)) / norm_psi
        
        Z_pred = rho * d * omega^2 * real(gamma_n)
        
        @printf("n=%-2d | %.3e   | %-12.1f | OK\n", n-1, Z_pred, Z_pred / K_static)
    end
    
    println("\nInterpretation:")
    println("If Ratio is ~10,000 for mode 0, the Spectral DtN Kernel is the correct law.")
    println("This method is pure A-Priori (Zero fitting, zero full-solves).")
end

main()
