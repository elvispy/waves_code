using LinearAlgebra
using SparseArrays
using Printf
using Statistics

# Purpose: Prove the 6000x boost factor by solving the Pure Fluid BVP.
# Domain: [-Ldom/2, Ldom/2] x [-H, 0]
# Mixed BC: phi_z = 1 (Raft), phi = 0 (Free Surface)

function solve_pure_bvp(L, H, L_dom; Nx=400, Nz=40)
    dx = L_dom / (Nx - 1)
    dz = H / (Nz - 1)
    
    x = range(-L_dom/2, L_dom/2, length=Nx)
    z = range(-H, 0, length=Nz)
    
    # 2D Laplacian Matrix (Standard 5-point stencil)
    N = Nx * Nz
    A = spzeros(N, N)
    b = zeros(N)
    
    # Helper to map (i, j) to flat index
    idx(i, j) = (j - 1) * Nx + i
    
    for j in 1:Nz
        for i in 1:Nx
            k = idx(i, j)
            
            # Boundary Conditions
            if j == Nz # Top Surface (z=0)
                if abs(x[i]) <= L/2
                    # Raft: phi_z = 1 (Neumann)
                    # (phi[i, Nz] - phi[i, Nz-1]) / dz = 1
                    A[k, k] = 1/dz
                    A[k, idx(i, j-1)] = -1/dz
                    b[k] = 1.0
                else
                    # Free Surface: phi = 0 (Dirichlet)
                    A[k, k] = 1.0
                    b[k] = 0.0
                end
            elseif j == 1 # Bottom (z=-H)
                # phi_z = 0
                A[k, k] = 1/dz
                A[k, idx(i, j+1)] = -1/dz
                b[k] = 0.0
            elseif i == 1 # Left Wall
                # phi_x = 0
                A[k, k] = 1/dx
                A[k, idx(i+1, j)] = -1/dx
                b[k] = 0.0
            elseif i == Nx # Right Wall
                # phi_x = 0
                A[k, k] = 1/dx
                A[k, idx(i-1, j)] = -1/dx
                b[k] = 0.0
            else
                # Interior: dxx phi + dzz phi = 0
                A[k, idx(i-1, j)] = 1/dx^2
                A[k, idx(i+1, j)] = 1/dx^2
                A[k, idx(i, j-1)] = 1/dz^2
                A[k, idx(i, j+1)] = 1/dz^2
                A[k, k] = -2/dx^2 - 2/dz^2
            end
        end
    end
    
    phi_flat = A \ b
    phi_2d = reshape(phi_flat, Nx, Nz)
    return x, phi_2d[:, end] # Return x and surface potential
end

function main()
    # Sample 32 Parameters
    L = 0.05
    H = 0.05
    L_dom = 1.5
    omega = 502.65
    
    println("--- Numerical Proof: Pure Fluid Admittance Eigenvalue ---")
    @printf("Tank: L=%.2f, H=%.2f, Ldom=%.2f\n", L, H, L_dom)
    
    x, phi_surf = solve_pure_bvp(L, H, L_dom)
    
    # Calculate Admittance factor Gamma = phi_avg / displacement_rate
    # Since phi_z = 1, the displacement rate is 1.
    raft_mask = abs.(x) .<= L/2
    gamma_0 = mean(phi_surf[raft_mask])
    
    # The expected ratio R = phi/eta is i*omega * gamma_0
    R_predicted = abs(omega * gamma_0)
    
    # Theoretical ratio from Obsidian (Wave-k)
    R_obsidian = 0.0195 # from previous test output
    
    boost_factor = R_predicted / R_obsidian
    
    println("\n--- Results ---")
    @printf("Pure Fluid Potential Mean (gamma_0): %.4f\n", gamma_0)
    @printf("Predicted Ratio |phi/eta|:           %.4f\n", R_predicted)
    @printf("Obsidian Ratio |phi/eta|:            %.4f\n", R_obsidian)
    @printf("Calculated Boost Factor:             %.1f x\n", boost_factor)
    
    println("\nConclusion:")
    if boost_factor > 1000
        println("The Finite-Domain BVP successfully explains the magnitude gap.")
        println("The law is: Qn = -d * rho * (i*omega)^2 * gamma_n * qn")
        println("Where gamma_n is the eigenvalue of the Mixed-BC Laplace operator.")
    else
        println("Discrepancy remains. The high frequency pressure release may be too strong.")
    end
end

main()
