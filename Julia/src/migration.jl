module Migration

using DelimitedFiles

using ..Analysis: default_coupled_motor_position_EI_sweep,
                  default_uncoupled_motor_position_EI_sweep
using ..Sweep: SweepArtifact, SweepSummary

export matlab_motor_position_ei_sources,
       load_motor_position_ei_export,
       artifact_from_motor_position_ei_export

const MATLAB_EXPORT_HEADER = [
    "EI",
    "motor_position",
    "tail_flat_ratio",
    "eta_beam_ratio",
    "eta_edge_ratio",
    "eta_left_beam_re",
    "eta_left_beam_im",
    "eta_right_beam_re",
    "eta_right_beam_im",
    "eta_left_domain_re",
    "eta_left_domain_im",
    "eta_right_domain_re",
    "eta_right_domain_im",
    "power",
    "thrust",
    "success",
    "retries",
    "Sxx",
    "N_x",
    "M_z",
    "n_used",
    "M_used",
]

"""
    matlab_motor_position_ei_sources()

Return the supported MATLAB sweep sources and their Julia-native destination
metadata.
"""
function matlab_motor_position_ei_sources()
    return (
        coupled = (
            source = "MATLAB/test/data/sweepMotorPositionEI.mat",
            output = "Julia/data/sweep_motorPosition_EI_coupled_from_matlab.jld2",
            label = "coupled_motor_position_EI_from_matlab",
            preset = default_coupled_motor_position_EI_sweep,
        ),
        uncoupled = (
            source = "MATLAB/test/data/sweepMotorPositionEI2.mat",
            output = "Julia/data/sweep_motorPosition_EI_uncoupled_from_matlab.jld2",
            label = "uncoupled_motor_position_EI_from_matlab",
            preset = default_uncoupled_motor_position_EI_sweep,
        ),
    )
end

"""
    load_motor_position_ei_export(path)

Read a plain numeric CSV exported from the MATLAB motor-position/EI sweep
table.
"""
function load_motor_position_ei_export(path::AbstractString)
    rows, header = readdlm(path, ',', Float64, header=true)
    names = vec(String.(header))
    names == MATLAB_EXPORT_HEADER || error("Unexpected export header in $path")
    return rows
end

function axis_maps(rows::AbstractMatrix{<:Real})
    EI = sort(unique(vec(rows[:, 1])))
    motor_position = sort(unique(vec(rows[:, 2])))
    ei_map = Dict(v => i for (i, v) in enumerate(EI))
    mp_map = Dict(v => i for (i, v) in enumerate(motor_position))
    return EI, motor_position, ei_map, mp_map
end

"""
    artifact_from_motor_position_ei_export(rows; label, base_params)

Build a Julia-native `SweepArtifact` from exported MATLAB sweep rows.

`U` is set to `NaN` because the original MATLAB sweep files do not store it.
"""
function artifact_from_motor_position_ei_export(rows::AbstractMatrix{<:Real}; label::AbstractString, base_params)
    EI_axis, motor_position_axis, ei_map, mp_map = axis_maps(rows)
    summaries = Array{SweepSummary}(undef, length(motor_position_axis), length(EI_axis))

    for i in axes(rows, 1)
        row = rows[i, :]
        ip = mp_map[row[2]]
        ie = ei_map[row[1]]
        summaries[ip, ie] = SweepSummary(
            NaN,
            row[14],
            -row[14],
            row[15],
            complex(row[6], row[7]),
            complex(row[8], row[9]),
            complex(row[10], row[11]),
            complex(row[12], row[13]),
            row[4],
            row[5],
            row[3],
        )
    end

    parameter_axes = (
        motor_position = motor_position_axis,
        EI = EI_axis,
    )
    return SweepArtifact{2, typeof(parameter_axes)}(
        String(label),
        base_params,
        parameter_axes,
        summaries,
    )
end

end
