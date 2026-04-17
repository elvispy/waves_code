using Test
using Surferbot

@testset "matlab sweep migration helpers" begin
    rows = [
        10.0  0.00  0.1  1.0   2.0   1.0  0.1  2.0  0.2  3.0  0.3  4.0  0.4  -5.0  6.0  1.0  0.0  7.0  8.0  9.0 10.0 11.0
        20.0  0.00  0.2  1.1   2.1   5.0  0.5  6.0  0.6  7.0  0.7  8.0  0.8  -9.0 10.0  1.0  0.0 11.0 12.0 13.0 14.0 15.0
        10.0  0.05  0.3  1.2   2.2   9.0  0.9 10.0  1.0 11.0  1.1 12.0  1.2 -13.0 14.0  1.0  0.0 15.0 16.0 17.0 18.0 19.0
        20.0  0.05  0.4  1.3   2.3  13.0  1.3 14.0  1.4 15.0  1.5 16.0  1.6 -17.0 18.0  1.0  0.0 19.0 20.0 21.0 22.0 23.0
    ]

    base = FlexibleParams(L_raft=0.05, d=0.03)
    artifact = artifact_from_motor_position_ei_export(rows; label="migration_test", base_params=base)

    @test artifact.label == "migration_test"
    @test artifact.base_params.L_raft == 0.05
    @test artifact.parameter_axes.motor_position == [0.0, 0.05]
    @test artifact.parameter_axes.EI == [10.0, 20.0]
    @test size(artifact.summaries) == (2, 2)

    s11 = artifact.summaries[1, 1]
    @test isnan(s11.U)
    @test s11.power == -5.0
    @test s11.power_input == 5.0
    @test s11.thrust == 6.0
    @test s11.eta_left_beam == 1.0 + 0.1im
    @test s11.eta_right_domain == 4.0 + 0.4im
    @test s11.eta_beam_ratio == 1.0
    @test s11.eta_domain_ratio == 2.0
    @test s11.tail_flat_ratio == 0.1

    s22 = artifact.summaries[2, 2]
    @test s22.power == -17.0
    @test s22.thrust == 18.0
    @test s22.eta_left_domain == 15.0 + 1.5im
end
