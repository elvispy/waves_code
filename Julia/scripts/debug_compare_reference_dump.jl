using Surferbot

# Purpose: generate side-by-side Julia and MATLAB parity dumps for one shared
# reference case under `tmp/parity_dump/`.

"""
    main()

Write Julia and, when available, MATLAB parity dumps for the shared reference
case into `tmp/parity_dump/`.
"""
function main()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    dump_root = joinpath(root, "tmp", "parity_dump")
    julia_dir = joinpath(dump_root, "julia")
    matlab_dir = joinpath(dump_root, "matlab")
    mkpath(julia_dir)
    mkpath(matlab_dir)

    println("Writing Julia dump to: ", julia_dir)
    julia_bin = get(ENV, "SURFERBOT_JULIA_BIN", "/Users/eaguerov/.julia/juliaup/julia-1.12.1+0.x64.apple.darwin14/bin/julia")
    julia_project = joinpath(root, "Julia")
    julia_depot = joinpath(julia_project, ".julia_depot") * ":/Users/eaguerov/.julia"
    julia_cmd = `$julia_bin --project=$julia_project $(joinpath(root, "Julia", "scripts", "debug_dump_reference_case.jl"))`
    Base.run(addenv(julia_cmd,
        "SURFERBOT_PARITY_DUMP_DIR" => julia_dir,
        "JULIA_DEPOT_PATH" => julia_depot,
    ))

    matlab = Sys.which("matlab")
    if matlab === nothing
        println("matlab not found on PATH; only Julia dump was written.")
        return
    end

    matlab_src = replace(joinpath(root, "MATLAB", "src"), "\\" => "/")
    matlab_test = replace(joinpath(root, "MATLAB", "test"), "\\" => "/")
    batch = "addpath('$matlab_src'); addpath('$matlab_test'); run_reference_case_dump_cli"
    println("Writing MATLAB dump to: ", matlab_dir)
    try
        Base.run(addenv(`$matlab -batch $batch`,
            "SURFERBOT_PARITY_DUMP_DIR" => matlab_dir,
            "OMP_NUM_THREADS" => "1",
            "KMP_AFFINITY" => "disabled",
        ))
    catch err
        println("MATLAB dump failed to run in this environment: ", err)
        println("Julia dump is still available at: ", julia_dir)
    end
end

main()
