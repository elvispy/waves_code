# Surferbot Julia Codebase Map

This document provides a high-level overview of the Julia research ecosystem.

## 1. Core Source Code (`Julia/src/`)
These files contain the stable library functions and the core physical solver.

| File | Purpose | Key Symbols |
| :--- | :--- | :--- |
| `Surferbot.jl` | **Primary Entry Point**. Defines params, results, and the main solver. | `flexible_solver`, `FlexibleParams`, `derive_params` |
| `modal.jl` | **Research Core**. Modal decomposition using the analytical `w` basis. | `ModalDecomposition`, `decompose_raft_freefree_modes`, `G_matrix` |
| `analysis.jl` | **Metrics & Branching**. Calculates $\alpha$ and extracts $S \approx 0$ curves. | `beam_asymmetry`, `extract_lowest_beam_curve` |
| `sweep.jl` | **Infrastructure**. Orchestrates 2D parameter sweeps ($EI$ vs $x_M$). | `sweep_parameters`, `SweepArtifact`, `save_sweep` |
| `postprocess.jl` | **Physical Recovery**. Calculates thrust, power, and Bernoulli pressure. | `calculate_surferbot_outputs` |
| `migration.jl` | **Legacy Support**. Tools to import and convert old MATLAB sweep data. | `matlab_motor_position_ei_sources` |
| `rigid.jl` | **Reference Solver**. A simplified version for rigid-body surferbots. | `rigid_solver` |
| `utils.jl` | **Physics Utils**. Dispersion relation and tensor system solvers. | `dispersion_k`, `solve_tensor_system` |
| `fd.jl` | **Numerical Core**. Finite difference matrix generation (up to 4th order). | `getNonCompactFDmatrix` |
| `integration.jl` | **Numerical Core**. Simpson and Trapezoidal integration weights. | `simpson_weights` |
| `video.jl` | **Visualization**. Renders 2D harmonic fields into animations. | `render_surferbot_run` |

---

## 2. Infrastructure Scripts (`Julia/scripts/`)
Generic scripts for running solvers and processing batches.

| File | Purpose |
| :--- | :--- |
| `run_sweep.jl` | Runs a standard 2D sweep ($EI$ vs $x_M$) and saves a `.jld2` artifact. |
| `brute_force_sweep.jl` | Higher-resolution sweep focusing on specific parameter ranges. |
| `plot_surferbot_run.jl` | Generates a quick spatial plot for a single solver result. |
| `optimization.jl` | Script for gradient-based optimization of the motor position. |
| `debug_dump_...` | Set of scripts to dump reference cases for cross-language parity tests. |

---

## 3. Research Experiments (`Julia/experiments/`)
Specialized scripts used to derive the "A Priori" laws and investigate the "Second Family."

### The Branch Tracker
| File | Purpose |
| :--- | :--- |
| `analyze_single_alpha_zero_curve.jl` | **High-Fidelity Tracker**. Follows the $\alpha=0$ branch with high precision and dumps a detailed CSV. |

### Added Mass Investigation (The "Diary" Path)
| File | Purpose |
| :--- | :--- |
| `added_mass_diary.json` | **Iteration History**. Records every hypothesis and audit result for the coupled law. |
| `audit_displacement_law.jl` | The **Winning Audit**. Proves the $L^{-1.27} q_n^{-0.14}$ law (0.48% error). |
| `find_optimal_added_mass.jl` | Performs complex regression to find the scalar impedance. |
| `correlation_audit.jl` | Deconstructs $Q_n$ scaling against $L, H, d$. |
| `probe_fluid_impedance.jl` | Direct numerical probe of the fluid reaction (no structural noise). |
| `audit_obsidian_laws.jl` | Tests the $k \tanh kH$ and Schulkes laws from the researcher's notes. |

### Visualizations & Diagnostics
| File | Purpose |
| :--- | :--- |
| `plot_uncoupled_scatter_comparison.jl` | Side-by-side scatter plots of Theory vs Numerical for $d=0$. |
| `plot_coupled_scatter_comparison.jl` | Generates the "a priori" overlay for the coupled $d \neq 0$ case. |
| `report_second_family_point_diagnostics.jl` | Detailed modal and physical report for specific branch points. |
| `visualize_pressure_comparison.jl` | Spatial plots of Numerical vs Analytical pressure profiles. |

### Verification & Validation
| File | Purpose |
| :--- | :--- |
| `verify_apriori_law.jl` | Validates the algebraic uncoupled law for the $S \approx 0$ branch. |
| `validate_apriori_law.jl` | Final check of the geometric law against randomized out-of-sample solves. |
| `check_w_basis_port_stability.jl` | Verifies that the new `w` basis matches the numerical pressure signal. |
