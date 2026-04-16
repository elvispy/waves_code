using Surferbot

function main(input; outdir=nothing, basename::AbstractString="waves", fps::Int=30, duration_periods::Real=10, nframes=nothing)
    if outdir === nothing
        outdir = input isa AbstractString ? (isdir(input) ? input : dirname(input)) : pwd()
    end
    return render_surferbot_run(
        input;
        outdir=outdir,
        basename=basename,
        fps=fps,
        duration_periods=duration_periods,
        nframes=nframes,
        script_name=Base.basename(@__FILE__),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Usage: julia --project=. scripts/plot_surferbot_run.jl <run-input> [outdir] [basename] [fps] [duration_periods] [nframes]")
    input = ARGS[1]
    outdir = length(ARGS) >= 2 ? ARGS[2] : nothing
    basename = length(ARGS) >= 3 ? ARGS[3] : "waves"
    fps = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
    duration_periods = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 10.0
    nframes = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : nothing
    main(input; outdir=outdir, basename=basename, fps=fps, duration_periods=duration_periods, nframes=nframes)
end

