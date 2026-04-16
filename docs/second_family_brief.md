# The Second Family of Asymmetry-Factor Zeros: An Open Problem

## Physical setup

A flexible raft (Euler-Bernoulli beam) of length $L$ and mass per unit length $\rho_R$ [kg/m] floats on an inviscid fluid of density $\rho$ [kg/m$^3$] and finite depth $H$. The raft has bending stiffness $EI$ [N m$^2$] and a third-dimension depth $d$ [m] (the raft's width into the page; it multiplies the hydrodynamic pressure to convert 2D pressure into a force per unit length on the beam). Surface tension $\sigma$ [N/m] and gravity $g$ [m/s$^2$] act on the free surface.

A motor with rotational inertia $I_{\mathrm{motor}}$ [kg m$^2$] is mounted at position $x_M$ from the raft center and oscillates at angular frequency $\omega$ [rad/s]. The motor exerts a localized force of amplitude $F_0 = I_{\mathrm{motor}} \omega^2$ on the raft, approximated as a Gaussian of width $\sigma_f = 0.05 L$ centered at $x_M$. Since $\sigma_f / L = 5\%$, the forcing is well-approximated as a delta function for the purposes of modal projection:

$$\hat{f}(x) \approx F_0 \cdot \delta(x - x_M)$$

The raft's flexural response radiates surface gravity-capillary waves into the surrounding fluid. The computational domain extends well beyond the raft, with **radiative boundary conditions** (outgoing waves only) at both ends.

### Governing equation

The raft displacement $\hat{\eta}(x)$ (complex amplitude at frequency $\omega$) satisfies on $|x| \le L/2$:

$$EI \, \hat{\eta}_{xxxx} - \rho_R \omega^2 \hat{\eta} = d \hat{p} - \hat{f}$$

where $\hat{p}(x)$ is the fluid pressure at $z = 0$ under the raft, given by linearized potential flow:

$$\hat{p} = -\rho\left(i\omega \hat{\phi} + g\hat{\eta}\right) \quad \text{at } z = 0$$

with $\hat{\phi}$ satisfying Laplace's equation $\nabla^2 \hat{\phi} = 0$ in the fluid, kinematic BC $\hat{\phi}_z = i\omega \hat{\eta}$ at $z = 0$ under the raft, the gravity-capillary free-surface condition outside the raft, $\hat{\phi}_z = 0$ at the seabed $z = -H$, and radiative (outgoing wave) conditions at $x \to \pm \infty$.

The free-surface wavenumber $k$ satisfies the gravity-capillary dispersion relation:

$$\omega^2 = \left(g k + \frac{\sigma}{\rho} k^3\right) \tanh(kH)$$

### Key output: asymmetry factor

The solver returns the complex wave elevation $\eta(x)$ across the entire domain. At the far-field domain edges, the outgoing waves have amplitudes $\eta_1$ (left edge, complex) and $\eta_{\mathrm{end}}$ (right edge, complex). The **asymmetry factor** is:

$$\alpha = -\frac{|\eta_1|^2 - |\eta_{\mathrm{end}}|^2}{|\eta_1|^2 + |\eta_{\mathrm{end}}|^2}$$

When $\alpha = 0$, the raft radiates equal-amplitude waves in both directions: zero net wave momentum flux, zero thrust.

## Modal framework

### Free-free beam modes

The raft displacement is expanded in the orthonormal eigenfunctions $\{W_n\}$ of the free-free beam:

$$W_n''''(x) = \beta_n^4 W_n(x), \qquad W_n'' = W_n''' = 0 \text{ at } x = \pm L/2$$

The eigenvalues $\beta_n$ satisfy $\cosh(\beta_n L)\cos(\beta_n L) = 1$ for elastic modes ($n \ge 2$), with the convention:
- $n = 0$: rigid translation, $W_0 = 1/\sqrt{L}$, $\beta_0 = 0$
- $n = 1$: rigid rotation, $W_1 = x \cdot 2\sqrt{3}/L^{3/2}$, $\beta_1 = 0$
- $n = 2, 3, \ldots$: elastic modes with $\beta_n L \approx 4.73, 7.85, 11.00, \ldots$

Even-indexed modes ($n = 0, 2, 4, \ldots$) are symmetric: $W_n(-x) = W_n(x)$. Odd-indexed modes ($n = 1, 3, 5, \ldots$) are antisymmetric: $W_n(-x) = -W_n(x)$.

### Modal equation

Expanding $\hat{\eta}(x) = \sum_n q_n W_n(x)$, multiplying the governing equation by $W_n$, and integrating over the raft gives:

$$(EI \beta_n^4 - \rho_R \omega^2) q_n = \hat{Q}_n - \hat{F}_n$$

where the forcing projection under the delta approximation is:

$$\hat{F}_n = I_{\mathrm{motor}} \omega^2 W_n(x_M)$$

and the pressure projection is:

$$\hat{Q}_n = d \int_{-L/2}^{L/2} \hat{p}(x) W_n(x) \, dx$$

## The two families of zeros

In the $(EI, x_M)$ parameter plane (at fixed $\omega = 2\pi \cdot 80$ rad/s), the level set $\alpha = 0$ consists of two distinct families of curves.

### First family (understood): vertical lines

At specific $EI$ values, independent of $x_M$. The derivation is in the companion note (file path: `/Users/eaguerov/Library/Mobile Documents/iCloud~md~obsidian/Documents/BrownObsidian/Research/Surferbot/Resonance distilled 2.md`).

In brief: using a diagonal approximation $\hat{Q}_n \approx (\omega^2 m_a + d\rho g) q_n$ where $m_a = d\rho/(k\tanh kH)$ is an added mass, the modal equation gives:

$$q_n = \frac{-\hat{F}_n}{D_n^{(0)}}, \qquad D_n^{(0)} = EI \beta_n^4 + d\rho g - \omega^2(\rho_R + m_a)$$

At resonance ($D_n^{(0)} = 0$), mode $n$ dominates all others. A single symmetric mode gives symmetric radiation ($\alpha = 0$); a single antisymmetric mode likewise. These resonance values of $EI$ are independent of $x_M$, producing vertical lines.

### Second family (open problem): non-vertical curves

Curves $x_M = f(EI)$ that cross all vertical lines and span $EI$ over 3+ decades. They depend on both $EI$ and $x_M$. Two sweep datasets confirm that these curves **vanish when $d = 0$** (uncoupled case): `sweepMotorPositionEI.mat` ($d = 0.03$, rich non-vertical structure) vs `sweepMotorPositionEI2.mat` ($d = 0$, only vertical features remain). Both datasets are at `MATLAB/test/data/`.

## Established results for the second family

### 1. Symmetric/antisymmetric far-field decomposition

Define:

$$S = \frac{\eta_{\mathrm{end}} + \eta_1}{2}, \qquad A = \frac{\eta_{\mathrm{end}} - \eta_1}{2}$$

Then $\eta_{\mathrm{end}} = S + A$ and $\eta_1 = S - A$, and:

$$|\eta_{\mathrm{end}}|^2 - |\eta_1|^2 = 4\operatorname{Re}(S A^*)$$

so $\alpha = 0 \iff \operatorname{Re}(S A^*) = 0$.

### 2. The mechanism is $|S| \approx 0$ or $|A| \approx 0$, not phase matching

**Method:** From the sweep data (`sweepMotorPositionEI.mat`, grid of 24 motor positions $\times$ 57 EI values), $S$ and $A$ are computed at every grid point from the stored $\eta_1$ and $\eta_{\mathrm{end}}$. Zero crossings of $\alpha$ are extracted column-by-column (for each EI value, scan $x_M$ for sign changes in $\alpha$). Each crossing is tagged with $\log_{10}(|S|/|A|)$ interpolated from the grid.

The lowest-$x_M$ zero crossings with $\log_{10}(|S|/|A|) < 0$ are tracked across all 57 EI columns, yielding 44 points on the lowest second-family curve. Adjacent curves (at higher $x_M$) are also extracted.

**Result:** Along any continuous second-family segment between consecutive vertical-line crossings, $\log_{10}(|S|/|A|)$ has consistent sign. Out of 27 such segments extracted, 24 have consistent sign; the 3 exceptions are 2–3 point segments at intersection points where the classification is ambiguous. The sign alternates between segments: S $\approx$ 0 segments interleave with A $\approx$ 0 segments from bottom to top in $x_M$.

This rules out the phase-matching hypothesis ($|S| \approx |A|$ with $\angle(S/A) = \pm 90°$). The mechanism is genuine vanishing of one far-field component.

The analysis script is `MATLAB/test/test_hypothesis_v3.m`.

### 3. Radiation coefficients

By symmetry, symmetric raft modes radiate symmetric far-field waves and antisymmetric modes radiate antisymmetric waves. Therefore:

$$S_{\mathrm{far}} = \sum_{n \text{ sym}} a_n q_n, \qquad A_{\mathrm{far}} = \sum_{n \text{ antisym}} a_n q_n$$

where $a_n$ (complex) are radiation coefficients quantifying how efficiently each beam mode generates outgoing waves.

**Fitting method:** At 15 points along the lowest S $\approx$ 0 curve, the solver is re-run and the modal decomposition (`MATLAB/src/decompose_raft_freefree_modes.m`) gives complex $q_n$. Simultaneously, $S_{\mathrm{far}}$ is computed from the solver output. The overdetermined system $S_{\mathrm{far}} = \mathbf{Q}_{\mathrm{sym}} \mathbf{a}$ (15 equations, 4 unknowns) is solved by least-squares.

**Result:** Mean relative residual $|S_{\mathrm{pred}} - S_{\mathrm{actual}}|/|S_{\mathrm{actual}}| = 1.6\%$. Fitted magnitudes:

| Mode | Type | $|a_n|$ |
|------|------|---------|
| 0 | rigid translation (sym) | 85.7 |
| 2 | 1st elastic (sym) | 7.9 |
| 4 | 3rd elastic (sym) | 6.9 |
| 6 | 5th elastic (sym) | 3.7 |

The rigid translation mode radiates ~10$\times$ more efficiently than the elastic symmetric modes. All $a_n$ share a common complex phase ($\approx -30°$ for elastic modes, $\approx 150°$ for rigid), so $S_{\mathrm{far}} = 0$ reduces to a single real equation.

The analysis script is `MATLAB/test/analyze_predict_second_family.m`.

### 4. Modal energy distribution along the curve

From the same 15 solver runs, the modal energy fractions $|q_n|^2 / \sum |q_n|^2$ show:
- Low EI ($\sim 10^{-5}$): higher elastic modes dominate near their resonances (e.g., mode 5 at 90%)
- Mid EI ($\sim 10^{-4}$): mode 3 (2nd elastic, antisymmetric) dominates at 75–85%
- High EI ($\sim 10^{-2}$): mode 1 (rigid rotation, antisymmetric) dominates at 85–91%, with mode 2 (1st elastic, symmetric) as secondary at 9–15%

The analysis script is `MATLAB/test/analyze_modal_decomposition_along_curve.m`.

## The implicit equation and what is missing

### Target equation

From the modal decomposition and the radiation coefficient fit, the condition defining the second family is:

$$f(x_M, EI) = \sum_{n \text{ sym}} a_n \frac{W_n(x_M)}{D_n(EI, x_M)} = 0$$

where $D_n$ is the effective modal impedance including hydrodynamic coupling. If $D_n$ were known as a function of $EI$ and $x_M$, this would be a closed implicit equation.

### What was tried and falsified

**Attempt 1 — mode-independent diagonal $D_n$:**

$$D_n^{(0)} = EI \beta_n^4 + d\rho g - \omega^2\left(\rho_R + \frac{d\rho}{k\tanh(kH)}\right)$$

Uses a single added mass $m_a = d\rho/(k\tanh kH)$ for all modes (plane-wave approximation). Prediction: zero found only in a tiny EI region near one resonance, at the wrong $x_M$. **Falsified.** Script: `MATLAB/test/test_two_mode_K_v2.m`.

**Attempt 2 — fit $D_n(EI) = \beta_n^4 EI + C_n$ from solver data:**

Extract $D_n = -\hat{F}_n / q_n$ at 20 points along the curve and fit a linear model in EI. Result: good fit for mode 2 (coefficient of variation 4.7%) but mode 0 has CoV 149% and modes 4, 6 exceed 400%. The rigid mode $D_0$ ($\beta_0 = 0$, so no EI dependence from the beam equation) varies by an order of magnitude along the curve, proving that $D_n$ depends on $x_M$ through the hydrodynamic coupling. Prediction: zero at $x_M/L \approx 0.02$ across all EI, completely wrong. **Falsified.** Script: `MATLAB/test/analyze_predict_second_family_v2.m`.

### The missing piece: off-diagonal hydrodynamic coupling

The pressure projection $\hat{Q}_n$ couples all modes through the fluid:

$$\hat{Q}_n = \sum_m H_{nm}(\omega) \, q_m$$

where $H_{nm}$ is the hydrodynamic coupling matrix. Its entries are:

$$H_{nm} = d \int_{-L/2}^{L/2} \hat{p}_m(x) \, W_n(x) \, dx$$

Here $\hat{p}_m(x)$ is the pressure at $z = 0$ on the raft when the raft oscillates in mode $W_m(x)$ alone, with the fluid satisfying Laplace's equation, kinematic BC $\hat{\phi}_z = i\omega W_m$ on the raft, gravity-capillary free-surface BC outside the raft, no-flow at the seabed, and radiative conditions at infinity. The matrix $\mathbf{H}$ depends on $\omega$, $L$, $H$, $\rho$, $\sigma$, $g$, $d$, and the mode shapes, but NOT on $EI$ or $x_M$.

With $\mathbf{H}$, the full modal system is:

$$\left(\mathbf{K}(EI) - \mathbf{H}\right) \mathbf{q} = -\hat{\mathbf{F}}(x_M)$$

where $\mathbf{K} = \operatorname{diag}(EI \beta_n^4 + d\rho g - \omega^2 \rho_R)$. The second family prediction becomes:

$$\boxed{S_{\mathrm{far}}(x_M, EI) = \mathbf{a}^T \left(\mathbf{K}(EI) - \mathbf{H}\right)^{-1} \hat{\mathbf{F}}(x_M) = 0}$$

All quantities are known except $\mathbf{H}$:
- $\mathbf{a}$: fitted (see table above)
- $\mathbf{K}(EI)$: diagonal, analytic
- $\hat{\mathbf{F}}(x_M)$: $\hat{F}_n = I_{\mathrm{motor}} \omega^2 W_n(x_M)$, analytic
- $\mathbf{H}$: **to be computed**

$\mathbf{H}$ can be obtained either:
1. Analytically, via a Green's function for a finite beam on a finite-depth free surface (standard in marine hydrodynamics)
2. Numerically, by running the existing solver $N_{\mathrm{modes}}$ times with forced raft displacement $\eta = W_m$ and extracting $\hat{p}_m$, then projecting

## Data and code

| Item | Path |
|------|------|
| Sweep data ($d = 0.03$) | `MATLAB/test/data/sweepMotorPositionEI.mat` |
| Sweep data ($d = 0$) | `MATLAB/test/data/sweepMotorPositionEI2.mat` |
| Sweep script | `MATLAB/test/sweep_motorPosition_EI.m` |
| Plot script | `MATLAB/test/plot_sweep_motorPosition_EI.m` |
| Solver | `MATLAB/src/flexible_surferbot_v2.m` |
| Modal decomposition | `MATLAB/src/decompose_raft_freefree_modes.m` |
| S/A analysis | `MATLAB/test/analyze_second_family_EI.m` |
| Hypothesis test (S$\approx$0 vs phase) | `MATLAB/test/test_hypothesis_v3.m` |
| Modal decomposition along curve | `MATLAB/test/analyze_modal_decomposition_along_curve.m` |
| Prediction attempts | `MATLAB/test/analyze_predict_second_family.m`, `analyze_predict_second_family_v2.m` |
| Resonance theory (first family) | `/Users/eaguerov/Library/Mobile Documents/iCloud~md~obsidian/Documents/BrownObsidian/Research/Surferbot/Resonance distilled 2.md` |

## Parameters for the sweep

```matlab
L_raft = 0.05;                        % [m] raft length
sigma  = 72.2e-3;                     % [N/m] surface tension
rho    = 1000;                        % [kg/m^3] fluid density
nu     = 0;                           % [m^2/s] kinematic viscosity (inviscid)
g      = 9.81;                        % [m/s^2] gravitational acceleration
d      = 0.03;                        % [m] raft depth (third dimension width)
EI     = 10.^linspace(-5.14, -1.92, 57);  % [N m^2] bending stiffness range
rho_raft = 0.052;                     % [kg/m] raft mass per unit length
motor_position = (0:0.02:0.48)*L_raft; % [m] motor position from center
motor_inertia = 0.13e-3 * 2.5e-3;    % [kg m^2] motor rotational inertia
omega  = 2*pi*80;                     % [rad/s] driving frequency
BC     = 'radiative';                 % outgoing wave boundary conditions
```
