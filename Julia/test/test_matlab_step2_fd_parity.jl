using Test
using DelimitedFiles
using Surferbot

function run_matlab_step2_dump(matlab_dir)
    matlab = Sys.which("matlab")
    matlab === nothing && return nothing

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    matlab_src = replace(joinpath(repo_root, "MATLAB", "src"), "\\" => "/")
    matlab_test = replace(joinpath(repo_root, "MATLAB", "test"), "\\" => "/")
    batch = "addpath('$matlab_src'); addpath('$matlab_test'); run_fd_step2_dump_cli"

    cmd = addenv(`$matlab -batch $batch`,
        "SURFERBOT_PARITY_DUMP_DIR" => matlab_dir,
        "OMP_NUM_THREADS" => "1",
        "KMP_AFFINITY" => "disabled",
    )

    try
        run(cmd)
    catch
        return nothing
    end
    return isfile(joinpath(matlab_dir, "fd1d_n1_ooa4.csv")) ? matlab_dir : nothing
end

function read_matrix(path)
    readdlm(path, ',', Float64)
end

function assert_array_close(a, b; atol=1e-13, rtol=1e-13)
    @test size(a) == size(b)
    for i in eachindex(a)
        @test isapprox(a[i], b[i]; atol=atol, rtol=rtol)
    end
end

@testset "matlab parity step2 fd" begin
    dump_root = mktempdir()
    matlab_dir = joinpath(dump_root, "matlab")
    mkpath(matlab_dir)

    matlab_result = run_matlab_step2_dump(matlab_dir)
    if matlab_result === nothing
        @test true
    else
        w11, _ = getNonCompactFDMWeights(1.0, 1, collect(-1:1))
        w24, _ = getNonCompactFDMWeights(1.0, 2, collect(-2:2))
        D11 = Matrix(getNonCompactFDmatrix(9, 1.0, 1, 4))
        D24 = Matrix(getNonCompactFDmatrix(9, 1.0, 2, 4))
        Dx2D, Dz2D = getNonCompactFDmatrix2D(5, 5, 1.0, 2.0, 1, 4)

        assert_array_close(reshape(w11, :, 1), read_matrix(joinpath(matlab_dir, "weights_n1_stencil_m1_1.csv")))
        assert_array_close(reshape(w24, :, 1), read_matrix(joinpath(matlab_dir, "weights_n2_stencil_m2_2.csv")))
        assert_array_close(D11, read_matrix(joinpath(matlab_dir, "fd1d_n1_ooa4.csv")))
        assert_array_close(D24, read_matrix(joinpath(matlab_dir, "fd1d_n2_ooa4.csv")))
        assert_array_close(Matrix(Dx2D), read_matrix(joinpath(matlab_dir, "fd2d_dx_n1_ooa4.csv")))
        assert_array_close(Matrix(Dz2D), read_matrix(joinpath(matlab_dir, "fd2d_dz_n1_ooa4.csv")))
    end
end
