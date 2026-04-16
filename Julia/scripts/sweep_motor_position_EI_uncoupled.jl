using Surferbot

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function main(
    save_dir::AbstractString=joinpath(@__DIR__, "..", "data");
    base_params_override=nothing,
    motor_position_list_override=nothing,
    EI_list_override=nothing,
    outfile::AbstractString="sweep_motorPosition_EI_uncoupled.jld2",
)
    save_dir = ensure_dir(normpath(save_dir))
    preset = default_uncoupled_motor_position_EI_sweep()
    base_tuple = isnothing(base_params_override) ? preset.base_params : base_params_override
    base_params = FlexibleParams(; base_tuple...)
    motor_position_list = isnothing(motor_position_list_override) ? preset.motor_position_list : collect(motor_position_list_override)
    EI_list = isnothing(EI_list_override) ? preset.EI_list : collect(EI_list_override)

    println("Running Julia uncoupled motor-position EI sweep")
    println("grid: $(length(motor_position_list)) motor positions x $(length(EI_list)) EI values")

    grid = (
        motor_position = motor_position_list,
        EI = EI_list,
    )
    return sweep_parameters(
        base_params,
        grid;
        solver=flexible_solver,
        beam_metrics_fn=beam_edge_metrics,
        label="uncoupled_motor_position_EI",
        save_path=joinpath(save_dir, outfile),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    save_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data")
    main(save_dir)
end
