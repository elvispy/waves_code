module Sweep

using JLD2
using Statistics

export SweepSummary,
       SweepArtifact,
       apply_parameter_overrides,
       expand_parameter_grid,
       summarize_result,
       sweep_parameters,
       save_sweep,
       load_sweep

"""
    SweepSummary

Standard per-case summary for Julia-native parameter sweeps.
"""
struct SweepSummary
    U::Float64
    power::Float64
    power_input::Float64
    thrust::Float64
    eta_left_beam::ComplexF64
    eta_right_beam::ComplexF64
    eta_left_domain::ComplexF64
    eta_right_domain::ComplexF64
    eta_beam_ratio::Float64
    eta_domain_ratio::Float64
    tail_flat_ratio::Float64
end

"""
    SweepArtifact

Native Julia sweep artifact containing the swept parameter axes and one
`SweepSummary` for each Cartesian grid point.
"""
struct SweepArtifact{N,Axes<:NamedTuple}
    label::String
    base_params
    parameter_axes::Axes
    summaries::Array{SweepSummary,N}
end

"""
    apply_parameter_overrides(base_params, overrides)

Create a new `FlexibleParams` instance by overriding selected fields from a
base parameter set.
"""
function apply_parameter_overrides(base_params, overrides::NamedTuple)
    names = fieldnames(typeof(base_params))
    kwargs = NamedTuple{names}(Tuple(getfield(base_params, name) for name in names))
    merged = merge(kwargs, overrides)
    return typeof(base_params)(; merged...)
end

"""
    expand_parameter_grid(grid)

Return the full Cartesian product of a sweep grid as a vector of named tuples.
"""
function expand_parameter_grid(grid::NamedTuple)
    names = keys(grid)
    axes = map(collect, values(grid))
    combos = NamedTuple{names}[]
    for idx in CartesianIndices(map(length, axes))
        vals = ntuple(i -> axes[i][Tuple(idx)[i]], length(axes))
        push!(combos, NamedTuple{names}(vals))
    end
    return combos
end

"""
    summarize_result(result, beam_metrics_fn)

Extract the standard sweep summary from a solver result.
"""
function summarize_result(result, beam_metrics_fn)
    metrics = beam_metrics_fn(result)
    left_count = max(1, ceil(Int, 0.05 * length(result.eta)))
    tail = abs.(result.eta[1:left_count])
    tail_flat_ratio = Statistics.std(tail) / max(eps(), Statistics.mean(tail))
    return SweepSummary(
        result.U,
        result.power,
        -result.power,
        result.thrust,
        ComplexF64(metrics.eta_left_beam),
        ComplexF64(metrics.eta_right_beam),
        ComplexF64(metrics.eta_left_domain),
        ComplexF64(metrics.eta_right_domain),
        float(metrics.eta_beam_ratio),
        float(metrics.eta_domain_ratio),
        tail_flat_ratio,
    )
end

"""
    sweep_parameters(base_params, grid; solver, beam_metrics_fn, label=\"\", save_path=nothing)

Run the solver over the full Cartesian product defined by `grid`.
"""
function sweep_parameters(base_params, grid::NamedTuple; solver, beam_metrics_fn, label::AbstractString="", save_path::Union{Nothing,AbstractString}=nothing)
    axes = NamedTuple{keys(grid)}(Tuple(collect(v) for v in values(grid)))
    dims = Tuple(length(v) for v in values(axes))
    summaries = Array{SweepSummary}(undef, dims...)

    for idx in CartesianIndices(dims)
        overrides = NamedTuple{keys(axes)}(ntuple(i -> values(axes)[i][Tuple(idx)[i]], length(axes)))
        params = apply_parameter_overrides(base_params, overrides)
        summaries[idx] = summarize_result(solver(params), beam_metrics_fn)
    end

    artifact = SweepArtifact{length(dims), typeof(axes)}(String(label), base_params, axes, summaries)
    if !isnothing(save_path)
        save_sweep(save_path, artifact)
    end
    return artifact
end

"""
    save_sweep(path, artifact)

Write a sweep artifact to a Julia-native `JLD2` file.
"""
function save_sweep(path::AbstractString, artifact::SweepArtifact)
    jldsave(path; artifact=artifact)
    return path
end

"""
    load_sweep(path)

Load a sweep artifact previously written by `save_sweep`.
"""
function load_sweep(path::AbstractString)
    return JLD2.load(path, "artifact")
end

end
