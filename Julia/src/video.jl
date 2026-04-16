module RunVideo

using Dates
using JLD2
using Printf

if !haskey(ENV, "GKSwstype") || isempty(ENV["GKSwstype"])
    ENV["GKSwstype"] = "100"
end

using Plots
gr()

export SurferbotRunRecord,
       normalize_run,
       write_provenance_json,
       render_surferbot_run

"""
    SurferbotRunRecord

Canonical in-memory record for run-video rendering.

- `U`: mean drift speed
- `x`: horizontal grid in meters
- `eta`: complex surface elevation amplitudes
- `args`: solver metadata and parameters
- `source`: provenance about where the record came from
"""
struct SurferbotRunRecord
    U::Float64
    x::Vector{Float64}
    eta::Vector{ComplexF64}
    args::NamedTuple
    source::NamedTuple
end

repo_root() = abspath(joinpath(@__DIR__, "..", ".."))

function maybe_get(nt, name::Symbol, default=nothing)
    hasproperty(nt, name) ? getproperty(nt, name) : default
end

function to_namedtuple(x)
    if x isa NamedTuple
        return x
    elseif x isa AbstractDict
        keys_vec = Symbol.(collect(keys(x)))
        vals_vec = collect(values(x))
        return NamedTuple{Tuple(keys_vec)}(Tuple(vals_vec))
    else
        error("Cannot convert $(typeof(x)) to NamedTuple")
    end
end

function load_saved_run(path::AbstractString)
    if isdir(path)
        for candidate in ("results.jld2", "run.jld2", "result.jld2")
            file = joinpath(path, candidate)
            if isfile(file)
                return JLD2.load(file)
            end
        end
        jld2_files = filter(f -> endswith(lowercase(f), ".jld2"), readdir(path; join=true))
        length(jld2_files) == 1 || error("Expected exactly one JLD2 file in $path or a standard run filename.")
        return JLD2.load(first(jld2_files))
    elseif isfile(path)
        return JLD2.load(path)
    else
        error("Run path not found: $path")
    end
end

function normalize_run(input; source_kind::AbstractString="unknown", source_path::Union{Nothing,AbstractString}=nothing)
    if input isa SurferbotRunRecord
        return input
    elseif hasproperty(input, :metadata) && hasproperty(input, :U) && hasproperty(input, :x) && hasproperty(input, :eta)
        args = to_namedtuple(input.metadata.args)
        return SurferbotRunRecord(
            float(input.U),
            Float64.(input.x),
            ComplexF64.(input.eta),
            args,
            (kind = "FlexibleResult", path = nothing),
        )
    elseif input isa NamedTuple
        required = (:U, :x, :eta, :args)
        all(hasproperty(input, key) for key in required) || error("NamedTuple input must contain U, x, eta, and args")
        args = to_namedtuple(getproperty(input, :args))
        return SurferbotRunRecord(
            float(getproperty(input, :U)),
            Float64.(getproperty(input, :x)),
            ComplexF64.(getproperty(input, :eta)),
            args,
            (kind = source_kind, path = source_path),
        )
    elseif input isa AbstractString
        loaded = load_saved_run(input)
        return normalize_run(loaded; source_kind="path", source_path=input)
    elseif input isa AbstractDict
        if haskey(input, "result")
            return normalize_run(input["result"]; source_kind=source_kind, source_path=source_path)
        elseif haskey(input, "run")
            return normalize_run(input["run"]; source_kind=source_kind, source_path=source_path)
        elseif all(haskey(input, key) for key in ("U", "x", "eta", "args"))
            payload = (
                U = input["U"],
                x = input["x"],
                eta = input["eta"],
                args = input["args"],
            )
            return normalize_run(payload; source_kind=source_kind, source_path=source_path)
        else
            error("Unsupported saved run payload. Expected result/run or U/x/eta/args keys.")
        end
    else
        error("Unsupported run input type: $(typeof(input))")
    end
end

function json_escape(s::AbstractString)
    io = IOBuffer()
    for c in collect(s)
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\b'
            print(io, "\\b")
        elseif c == '\f'
            print(io, "\\f")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

function write_json_value(io, value; indent::Int=0)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        if isfinite(value)
            print(io, value)
        elseif isnan(value)
            print(io, "\"NaN\"")
        elseif value > 0
            print(io, "\"Inf\"")
        else
            print(io, "\"-Inf\"")
        end
    elseif value isa AbstractString
        print(io, '"', json_escape(value), '"')
    elseif value isa Symbol
        print(io, '"', json_escape(String(value)), '"')
    elseif value isa AbstractVector
        if isempty(value)
            print(io, "[]")
        else
            print(io, "[\n")
            for (i, item) in enumerate(value)
                print(io, repeat(" ", indent + 2))
                write_json_value(io, item; indent=indent + 2)
                if i < length(value)
                    print(io, ",")
                end
                print(io, "\n")
            end
            print(io, repeat(" ", indent), "]")
        end
    elseif value isa NamedTuple
        fields = collect(pairs(value))
        if isempty(fields)
            print(io, "{}")
        else
            print(io, "{\n")
            for (i, (key, item)) in enumerate(fields)
                print(io, repeat(" ", indent + 2))
                print(io, "\"", json_escape(String(key)), "\": ")
                write_json_value(io, item; indent=indent + 2)
                if i < length(fields)
                    print(io, ",")
                end
                print(io, "\n")
            end
            print(io, repeat(" ", indent), "}")
        end
    elseif value isa AbstractDict
        pairs_vec = collect(pairs(value))
        if isempty(pairs_vec)
            print(io, "{}")
        else
            print(io, "{\n")
            for (i, (key, item)) in enumerate(pairs_vec)
                print(io, repeat(" ", indent + 2))
                print(io, "\"", json_escape(String(key)), "\": ")
                write_json_value(io, item; indent=indent + 2)
                if i < length(pairs_vec)
                    print(io, ",")
                end
                print(io, "\n")
            end
            print(io, repeat(" ", indent), "}")
        end
    elseif value isa Complex
        print(io, '"', json_escape(string(value)), '"')
    else
        write_json_value(io, string(value); indent=indent)
    end
end

function git_commit_hash()
    try
        return strip(readchomp(`git -C $(repo_root()) rev-parse HEAD`))
    catch
        return "unknown"
    end
end

function build_provenance(record::SurferbotRunRecord; output_basename::AbstractString, output_dir::AbstractString, script_name::AbstractString)
    return (
        script_name = script_name,
        output_basename = output_basename,
        output_dir = output_dir,
        generated_at = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        git_commit = git_commit_hash(),
        source = record.source,
        U = record.U,
        thrust = maybe_get(record.args, :thrust, nothing),
        power = maybe_get(record.args, :power, nothing),
        args = record.args,
    )
end

"""
    write_provenance_json(path, provenance)

Write provenance metadata as pretty JSON without adding a new package
dependency.
"""
function write_provenance_json(path::AbstractString, provenance)
    open(path, "w") do io
        write_json_value(io, provenance; indent=0)
        print(io, "\n")
    end
    return path
end

function plot_frame(record::SurferbotRunRecord, t::Real; omega::Real, x_contact_mask, motor_idx::Union{Nothing,Int}, show_motor::Bool)
    scaleY = 1e6
    y = real.(record.eta .* exp.(1im * omega * t)) .* scaleY
    x_scaled = record.x .* 1e2
    y_limit = maximum(abs.(record.eta)) * scaleY * 1.1

    p = plot(
        x_scaled,
        y;
        fillrange = -y_limit,
        fillalpha = 0.25,
        color = :steelblue,
        linewidth = 4,
        label = false,
        xlabel = "x (cm)",
        ylabel = "y (um)",
        ylim = (-y_limit, y_limit * 1.35),
        xlim = (first(x_scaled), last(x_scaled)),
        title = @sprintf("f = %.2f Hz, t = %.3f s", omega / (2π), t),
        legend = show_motor,
        size = (1400, 480),
    )

    if !isempty(x_contact_mask)
        plot!(p, x_scaled[x_contact_mask], y[x_contact_mask]; color = :black, linewidth = 8, label = false)
    end

    if show_motor
        motor_idx === nothing || scatter!(p, [x_scaled[motor_idx]], [y[motor_idx]]; color = :goldenrod, markersize = 8, label = "Motor")
    end

    return p
end

function render_surferbot_run(input; outdir::AbstractString=pwd(), basename::AbstractString="waves", fps::Int=30, duration_periods::Real=10, nframes::Union{Nothing,Int}=nothing, script_name::AbstractString=Base.basename(PROGRAM_FILE))
    record = normalize_run(input)
    mkpath(outdir)
    mp4_path = joinpath(outdir, basename * ".mp4")
    json_path = joinpath(outdir, basename * ".json")
    provenance = build_provenance(record; output_basename=basename, output_dir=outdir, script_name=script_name)
    write_provenance_json(json_path, provenance)

    omega = maybe_get(record.args, :omega, nothing)
    omega === nothing && error("Run metadata must include omega for video rendering.")
    omega = float(omega)

    contact_mask = Bool[]
    if hasproperty(record.args, :x_contact)
        contact_mask = collect(Bool.(getproperty(record.args, :x_contact)))
    end

    motor_position = maybe_get(record.args, :motor_position, nothing)
    motor_idx = motor_position === nothing ? nothing : argmin(abs.(record.x .- float(motor_position)))

    total_frames = something(nframes, Int(round(duration_periods * fps)))
    tvec = range(0, stop = duration_periods * (2π / omega), length = total_frames)

    anim = @animate for t in tvec
        plot_frame(record, t; omega = omega, x_contact_mask = contact_mask, motor_idx = motor_idx, show_motor = motor_idx !== nothing)
    end

    mp4(anim, mp4_path, fps = fps)
    return (mp4 = mp4_path, json = json_path)
end

end
