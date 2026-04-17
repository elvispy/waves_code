using DelimitedFiles
using Surferbot

# Purpose: dump the derived-parameter stage of the Julia reference case to CSV
# files for parity checks against MATLAB step 1.

function write_vec(path, data)
    writedlm(path, reshape(data, :, 1), ',')
end

"""
    main()

Write the Julia reference-case derived quantities into the directory pointed to
by `ENV["SURFERBOT_PARITY_DUMP_DIR"]`.
"""
function main()
    outdir = get(ENV, "SURFERBOT_PARITY_DUMP_DIR", nothing)
    outdir === nothing && error("SURFERBOT_PARITY_DUMP_DIR is not set.")
    mkpath(outdir)

    params = FlexibleParams(
        sigma = 0.0,
        rho = 1000.0,
        nu = 1e-6,
        g = 10 * 9.81,
        L_raft = 0.1,
        motor_position = 0.5 * 0.1 / 2,
        d = 0.1 / 2,
        EI = 100 * 3.0e9 * 3e-2 * (9.9e-4)^3 / 12,
        rho_raft = 0.018 * 3.0,
        domain_depth = 0.2,
        n = 41,
        M = 30,
        motor_inertia = 0.13e-3 * 2.5e-3,
        bc = :radiative,
        omega = 2 * pi * 10,
    )

    derived = derive_params(params)

    write_vec(joinpath(outdir, "x.csv"), derived.x .* derived.L_c)
    write_vec(joinpath(outdir, "z.csv"), derived.z .* derived.L_c)
    write_vec(joinpath(outdir, "loads.csv"), derived.loads .* derived.F_c ./ derived.L_c)
    write_vec(joinpath(outdir, "x_contact.csv"), Float64.(derived.x_contact))
    write_vec(joinpath(outdir, "x_free.csv"), Float64.(derived.x_free))

    open(joinpath(outdir, "summary.csv"), "w") do io
        write(io, "k_real,k_imag,N,M,dx,dz,Gamma,Fr,Re,kappa,We,Lambda\n")
        write(io, string(real(derived.k), ",", imag(derived.k), ",", derived.N, ",", derived.M, ",",
            derived.dx * derived.L_c, ",", derived.dz * derived.L_c, ",",
            derived.nd_groups.Gamma, ",", derived.nd_groups.Fr, ",", derived.nd_groups.Re, ",",
            derived.nd_groups.kappa, ",", derived.nd_groups.We, ",", derived.nd_groups.Lambda, "\n"))
    end
end

main()
