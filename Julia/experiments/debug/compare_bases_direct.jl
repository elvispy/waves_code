using Surferbot
using Printf
using LinearAlgebra
using Statistics

"""
compare_bases_direct.jl

Performs a direct numerical comparison between the orthonormal Psi basis 
(discrete Gram-Schmidt) and the raw analytical W basis (Phi). Reports Gram 
matrix identity errors and coefficient transformation residuals.
"""

# Compare the numerical basis (Psi_n) and the analytical basis (W_n) directly.

function main()
    # Use default params (uncoupled)
    params = FlexibleParams(d=0.0)
    result = flexible_solver(params)
    
    # 1. Standard decomposition (Psi_n)
    modal = decompose_raft_freefree_modes(result; num_modes=8, verbose=false)
    Psi = modal.Psi
    w = trapz_weights(modal.x_raft)
    
    # 2. Build raw Phi (W_n)
    raw = build_raw_freefree_basis(modal.x_raft, params.L_raft; num_modes=8, include_rigid=true)
    Phi = raw.Phi
    
    # 3. Check Gram Matrices
    G_psi = Psi' * (Psi .* w)
    G_phi = Phi' * (Phi .* w)
    
    println("--- BASIS ORTHOGONALITY CHECK ---")
    println("Gram matrix (Psi): Identity error = ", norm(G_psi - I))
    println("Gram matrix (Phi): Identity error = ", norm(G_phi - I))
    println("Gram matrix (Phi) condition number: ", cond(G_phi))
    
    # 4. Compare q_n calculated in both bases
    # c_psi = q_n in Psi basis
    # c_phi = q_n in Phi basis
    c_psi = modal.q
    
    # To get c_phi such that Phi * c_phi ≈ Psi * c_psi
    # we solve (Phi' W Phi) c_phi = Phi' W Psi c_psi
    T = (Phi' * (Phi .* w)) \ (Phi' * (Psi .* w))
    c_phi_ported = T * c_psi
    
    # Or calculate c_phi directly by projection:
    eta_raft = result.eta[result.metadata.args.x_contact]
    c_phi_direct = (Phi' * (Phi .* w)) \ (Phi' * (eta_raft .* w))
    
    println("\n--- COEFFICIENT COMPARISON (Psi vs W) ---")
    println("mode n   |q_psi|      |q_w_ported|  |q_w_direct|  rel_diff(ported vs direct)")
    for i in 1:length(modal.n)
        rel_diff = abs(c_phi_ported[i] - c_phi_direct[i]) / max(abs(c_phi_direct[i]), 1e-12)
        @printf("%d    %d   %.6e   %.6e   %.6e   %.3e\n", i, modal.n[i], abs(c_psi[i]), abs(c_phi_ported[i]), abs(c_phi_direct[i]), rel_diff)
    end
    
    # 5. Check reconstruction error in both bases
    eta_recon_psi = Psi * c_psi
    eta_recon_phi = Phi * c_phi_direct
    
    err_psi = norm((eta_raft - eta_recon_psi) .* sqrt.(w)) / norm(eta_raft .* sqrt.(w))
    err_phi = norm((eta_raft - eta_recon_phi) .* sqrt.(w)) / norm(eta_raft .* sqrt.(w))
    
    println("\n--- RECONSTRUCTION ERROR ---")
    @printf("Psi basis rel err: %.6e\n", err_psi)
    @printf("Phi basis rel err: %.6e\n", err_phi)
    
    # 6. The "40% Error" Mystery: Check branch equation sensitivity
    # If we use raw Phi[1, :] instead of normalized Psi[1, :], does it change things?
    println("\n--- ENDPOINT SENSITIVITY (x=L/2) ---")
    println("mode n   Psi[end, n]      Phi[end, n]      Ratio")
    for i in 1:length(modal.n)
        ratio = Psi[end, i] / Phi[end, i]
        @printf("%d    %d   % .6e   % .6e   % .6f\n", i, modal.n[i], Psi[end, i], Phi[end, i], ratio)
    end
end

main()
