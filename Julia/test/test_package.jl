using Test

@testset "Package surface" begin
    using Surferbot

    @test isdefined(Surferbot, :FlexibleParams)
    @test isdefined(Surferbot, :FlexibleResult)
    @test isdefined(Surferbot, :flexible_solver)
    @test isdefined(Surferbot, :flexible_surferbot_v2_julia)
    @test isdefined(Surferbot, :assemble_flexible_system)
    @test isdefined(Surferbot, :derive_params)
    @test isdefined(Surferbot, :simpson_weights)
    @test isdefined(Surferbot, :dispersion_k)
    @test !(:DtN_generator in names(Surferbot))
end
