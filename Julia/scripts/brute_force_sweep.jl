using Surferbot
using JLD2
using Printf

function main()
    # 1. Get task ID from Slurm (defaults to 1 for local testing)
    task_id = parse(Int, get(ARGS, 1, "1"))
    
    nx = 100
    nei = 300
    output_dir = joinpath(@__DIR__, "..", "output", "parts")
    mkpath(output_dir)
    output_path = joinpath(output_dir, "part_$(task_id).jld2")

    # 2. Reconstruct Grid
    L_raft = 0.05
    motor_position_list = collect(range(0.0, 0.49; length=nx) .* L_raft)
    logEI_range = range(-2.0, -8.0; length=nei)
    EI_list = 10.0 .^ logEI_range

    # This task handles ONE EI value
    target_ei = EI_list[task_id]
    results = Vector{Any}(nothing, nx)

    base_params = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    
    println("Task $task_id: Running $nx motor positions for EI = $target_ei")

    for im in 1:nx
        params = Surferbot.Sweep.apply_parameter_overrides(base_params, (
            motor_position = motor_position_list[im],
            EI = target_ei
        ))
        
        try
            res = flexible_solver(params)
            # Minimalist storage to prevent OOM
            results[im] = (
                U = res.U,
                power = res.power,
                thrust = res.thrust,
                eta = res.eta,
                pressure = res.pressure,
                max_curvature = res.max_curvature,
                metadata = (N=res.metadata.args.N, M=res.metadata.args.M) # Strip all but essentials
            )
        catch e
            @warn "Failed at im=$im: $e"
        end
    end

    # 3. Save individual part
    jldsave(output_path; results=results, task_id=task_id, EI=target_ei, motor_position_list=motor_position_list)
    println("Saved $output_path")
end

main()
