# Second Family Key Equations

## Beam modal balance
$$
\left(EI\beta_n^4-\rho_R\omega^2\right) q_n = Q_n - F_n
$$
Per-mode force balance from the Fourier-space beam equation.

## Uncoupled specialization
$$
Q_n = 0
$$
For the uncoupled case, the hydrodynamic modal forcing is removed.

## Delta-load forcing
$$
F_n(x_M)\approx F_0\,W_n(x_M)
$$
Point-actuator approximation for the modal load projection.

## Discrete modal balance (W basis)
$$
\mathbf{G} \mathbf{D} \mathbf{q}^W = \mathbf{Q}^W - \mathbf{F}^W
$$
Coupled discrete balance accounting for non-orthogonality of the $W_n$ basis on the raft grid, where $G_{nm} = \langle W_n, W_m \rangle_w$ is the Gram matrix.

## Uncoupled specialization (A Priori)
$$
\mathbf{q}^W = -\mathbf{D}^{-1} \mathbf{G}^{-1} \mathbf{F}^W
$$
Exact *a priori* modal coefficients for the uncoupled ($d=0, Q=0$) discrete case.

## Delta-load forcing
$$
F_n^W(x_M)\approx F_0\,W_n(x_M)
$$
Point-actuator approximation for the modal load projection in the $W$ basis.

## Alpha condition
$$
\alpha = 0 \iff \operatorname{Re}(S A^*) = 0
$$
The alpha-zero curve is the locus where the symmetric ($S$) and antisymmetric ($A$) endpoint parts are orthogonal in the complex plane.

## Coupled A Posteriori Verification
Along the $\alpha=0$ curve in the coupled case ($d > 0$), the symmetric response $S$ must vanish through a cancellation of mechanical and hydrodynamic parts:
$$
S_{\text{total}} = S_{\text{mech}} + S_{\text{hydro}} \approx 0
$$
where
$$
S_{\text{mech}} = \sum_{n \text{ even}} q_{n, \text{mech}} W_n(L/2), \quad
S_{\text{hydro}} = \sum_{n \text{ even}} q_{n, \text{hydro}} W_n(L/2)
$$
Verification is performed by extracting $q_n$ directly from high-fidelity cluster solves.

## Two-mode branch equation (Coupled discrete)
$$
\mathbf{W}_{\text{end}}^T \mathbf{D}^{-1} \mathbf{G}^{-1} \mathbf{W}(x_M) \approx 0
$$
High-fidelity implicit equation for the $S=0$ branch. The previously used diagonal law $\sum \frac{W_n(x_M)W_n(\text{end})}{D_n} = 0$ is the continuous limit ($\mathbf{G} \to \mathbf{I}$) and carries significant errors on standard simulation grids.


## Symmetric endpoint closure
$$
S_{\mathrm{full}}=\sum_{n\ \mathrm{even}} q_n W_n(L/2),\qquad
S_{02}=q_0W_0(L/2)+q_2W_2(L/2),\qquad
S_{\mathrm{rest}}=S_{\mathrm{full}}-S_{02}
$$
Separates the full symmetric endpoint cancellation into the `0+2` part and the higher-even residual.
