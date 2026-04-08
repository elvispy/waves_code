using DelimitedFiles
using Surferbot

function write_vec(path, data)
    writedlm(path, reshape(data, :, 1), ',')
end

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

    result, system = let rs = flexible_solver(params; return_system=true)
        rs.result, rs.system
    end
    args = result.metadata.args

    write_vec(joinpath(outdir, "x.csv"), result.x)
    write_vec(joinpath(outdir, "z.csv"), result.z)
    write_vec(joinpath(outdir, "eta_real.csv"), real.(result.eta))
    write_vec(joinpath(outdir, "eta_imag.csv"), imag.(result.eta))
    write_vec(joinpath(outdir, "loads.csv"), args.loads)
    write_vec(joinpath(outdir, "pressure_real.csv"), real.(result.pressure))
    write_vec(joinpath(outdir, "pressure_imag.csv"), imag.(result.pressure))
    write_vec(joinpath(outdir, "phi_surface_real.csv"), real.(vec(result.phi[end, :])))
    write_vec(joinpath(outdir, "phi_surface_imag.csv"), imag.(vec(result.phi[end, :])))
    write_vec(joinpath(outdir, "phi_z_surface_real.csv"), real.(vec(result.phi_z[end, :])))
    write_vec(joinpath(outdir, "phi_z_surface_imag.csv"), imag.(vec(result.phi_z[end, :])))

    open(joinpath(outdir, "summary.csv"), "w") do io
        write(io, "U,power,thrust,k_real,k_imag,N,M,dx,dz\n")
        write(io, string(result.U, ",", result.power, ",", result.thrust, ",",
            real(args.k), ",", imag(args.k), ",", args.N, ",", args.M, ",", args.dx, ",", args.dz, "\n"))
    end
end

main()
