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

Container for per-case summary data from a parameter sweep.

# Fields
- `U`: Drift speed.
- `power`: Power (solver convention).
- `power_input`: Actuator input power.
- `thrust`: Mean thrust.
- `eta_left_beam`: Left beam displacement.
- `eta_right_beam`: Right beam displacement.
- `eta_left_domain`: Left domain displacement.
- `eta_right_domain`: Right domain displacement.
- `eta_beam_ratio`: Ratio of beam edge displacements.
- `eta_domain_ratio`: Ratio of domain edge displacements.
- `tail_flat_ratio`: Metric for wave flatness at the tail.
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
    SweepArtifact{N, Axes<:NamedTuple}

A complete sweep dataset including axes and summaries.

# Fields
- `label`: Description of the sweep.
- `base_params`: Template parameters used.
- `parameter_axes`: Axes of the sweep grid.
- `summaries`: N-dimensional array of results.
"""
struct SweepArtifact{N,Axes<:NamedTuple}
    label::String
    base_params
    parameter_axes::Axes
    summaries::Array{SweepSummary,N}
end

"""
    apply_parameter_overrides(base_params, overrides::NamedTuple)

Generate parameters by overriding fields in a base set.

# Arguments
- `base_params`: Base parameter object (or JLD2 reconstructed).
- `overrides`: NamedTuple of field values to change.

# Returns
- A new `FlexibleParams` object.
"""
function apply_parameter_overrides(base_params, overrides::NamedTuple)
    target_type = Main.Surferbot.FlexibleParams
    valid_names = fieldnames(target_type)
    kwargs = Dict{Symbol, Any}()
    
    T_base = typeof(base_params)
    if hasfield(T_base, :fields) && length(T_base.parameters) >= 2
        fnames = T_base.parameters[2] 
        fvals = getfield(base_params, :fields)
        for (name, val) in zip(fnames, fvals)
            if name in valid_names
                kwargs[name] = val
            end
        end
    else
        for name in fieldnames(T_base)
            if name in valid_names
                kwargs[name] = getfield(base_params, name)
            end
        end
    end
    
    for (k, v) in pairs(overrides)
        if k in valid_names
            kwargs[k] = v
        end
    end
    
    return target_type(; kwargs...)
end

"""
    expand_parameter_grid(grid::NamedTuple)

Compute the Cartesian product of all parameters in a grid.

# Arguments
- `grid`: NamedTuple where values are collections.

# Returns
- Vector of NamedTuples representing all combinations.
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

Condense a solver result into a `SweepSummary`.

# Arguments
- `result`: Result from the solver.
- `beam_metrics_fn`: Function to extract beam-end metrics.

# Returns
- A `SweepSummary` instance.
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
    sweep_parameters(base_params, grid; solver, beam_metrics_fn, label="", save_path=nothing)

Run a parameter sweep over a Cartesian grid.

# Arguments
- `base_params`: Base parameters for the simulation.
- `grid`: NamedTuple defining the parameter axes.
- `solver`: Function to solve the system for a given parameter set.
- `beam_metrics_fn`: Function to extract metrics from the results.
- `label`: Optional label for the sweep artifact.
- `save_path`: Optional path to save the artifact in JLD2 format.

# Returns
- A `SweepArtifact` object.
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
    save_sweep(path::AbstractString, artifact::SweepArtifact)

Save a `SweepArtifact` to disk.
"""
function save_sweep(path::AbstractString, artifact::SweepArtifact)
    jldsave(path; artifact=artifact)
    return path
end

"""
    load_sweep(path::AbstractString)

Load a `SweepArtifact` from a JLD2 file.
"""
function load_sweep(path::AbstractString)
    return JLD2.load(path, "artifact")
end

end # module
