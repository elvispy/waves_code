using Test

include("test_package.jl")
include("test_derive_params.jl")
include("test_matlab_step1_parity.jl")
include("test_matlab_step2_fd_parity.jl")
include("test_matlab_step3_assembly_indices.jl")
include("test_matlab_step4_assembly_matrix.jl")
include("test_flexible_system.jl")
include("test_flexible_solver.jl")
include("test_analysis_helpers.jl")
include("test_modal_decomposition.jl")
include("test_migration.jl")
include("test_sweep.jl")
include("test_video.jl")
include("test_optimization_objective.jl")
include("test_optimization_gradients.jl")
include("test_graded_beam_params.jl")
if get(ENV, "SURFERBOT_ENABLE_FULL_PARITY", "0") == "1"
    include("test_matlab_parity.jl")
end
