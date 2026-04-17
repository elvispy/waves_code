using Surferbot

# Purpose: one-off migration helper that exports the historical MATLAB
# `motorPosition-EI` sweeps and converts them into native Julia `JLD2`
# sweep artifacts.

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function matlab_export_command(helper_path::AbstractString, source_path::AbstractString, csv_path::AbstractString)
    return string(
        "addpath('", dirname(helper_path), "'); ",
        debug_export_motorPosition_EI_table_for_julia_call(helper_path, source_path, csv_path),
    )
end

function debug_export_motorPosition_EI_table_for_julia_call(helper_path::AbstractString, source_path::AbstractString, csv_path::AbstractString)
    helper_name = splitext(basename(helper_path))[1]
    return helper_name * "('" * source_path * "','" * csv_path * "');"
end

function run_matlab_export(helper_path::AbstractString, source_path::AbstractString, csv_path::AbstractString)
    cmd = `matlab -batch $(matlab_export_command(helper_path, source_path, csv_path))`
    run(addenv(cmd, "OMP_NUM_THREADS" => "1"))
    return csv_path
end

function build_base_params(kind::Symbol)
    preset = if kind == :coupled
        default_coupled_motor_position_EI_sweep()
    elseif kind == :uncoupled
        default_uncoupled_motor_position_EI_sweep()
    else
        error("Unsupported translation kind: $kind")
    end
    return FlexibleParams(; preset.base_params...)
end

"""
    main(root=normpath(joinpath(@__DIR__, "..", "..")))

Translate the historical coupled and uncoupled MATLAB `motorPosition-EI` sweep
artifacts into Julia-native `JLD2` sweep files.

Inputs:
- `root`: repository root containing both the MATLAB source sweep files and the
  Julia output directory.
"""
function main(root::AbstractString=normpath(joinpath(@__DIR__, "..", "..")))
    root = normpath(root)
    helper_path = joinpath(root, "MATLAB", "test", "debug_export_motorPosition_EI_table_for_julia.m")
    tmp_dir = mktempdir()
    ensure_dir(joinpath(root, "Julia", "data"))

    for (kind, spec) in pairs(matlab_motor_position_ei_sources())
        source_path = joinpath(root, spec.source)
        output_path = joinpath(root, spec.output)
        csv_path = joinpath(tmp_dir, string(kind, "_motor_position_EI_export.csv"))

        println("Translating $(source_path)")
        run_matlab_export(helper_path, source_path, csv_path)
        rows = load_motor_position_ei_export(csv_path)
        artifact = artifact_from_motor_position_ei_export(
            rows;
            label=spec.label,
            base_params=build_base_params(kind),
        )
        save_sweep(output_path, artifact)
        println("Saved $(output_path)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    root = length(ARGS) >= 1 ? ARGS[1] : normpath(joinpath(@__DIR__, "..", ".."))
    main(root)
end
