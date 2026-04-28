# Architectural Decision Records

## ADR-001: Canonical Modal Basis for Surferbot Theory

**Date:** April 19, 2026  
**Status:** Decided (Consensus via PAL/Codex)  
**Context:** We identified a "smog" of 10-40% errors when porting from the numerical solver basis ($\Psi_n$) to the analytical free-free basis ($W_n$).

### Decision
We will **not** abandon the analytical basis $W_n$. It is the only basis with physical/theoretical meaning for an *a priori* $x_M^*(EI)$ law. However, we must treat the discrete simulation as a **coupled** system in this basis.

### Rationale
1.  **Semantic Purity:** $W_n$ (sin, cos, sinh, cosh) allows for closed-form theoretical laws. $\Psi_n$ is a grid-dependent artifact of the modified Gram-Schmidt process.
2.  **Discrete Coupling:** On any discrete grid, $W_n$ vectors are not orthogonal ($\langle W_n, W_m \rangle_w \neq \delta_{nm}$). The Gram matrix $\mathbf{G}$ has an identity error of ~2.7 and a condition number of ~4800 on a standard 21-node raft grid.
3.  **Accuracy:** The previously observed "40% error" was a failure to account for this non-orthogonality and the resulting coordinate transformation ($q^W = \mathbf{T} q^\Psi$).

### Consequences
*   **Theory:** The *a priori* law for $S \approx 0$ is $\sum \frac{W_n(x_M) W_n(L/2)}{D_n(EI)} = 0$ in the continuous limit. On a discrete grid, we must use the coupled version $\mathbf{G} \mathbf{q}^W = -\frac{1}{EI} \mathbf{F}^W$.
*   **Data:** All `.csv` databases containing $q_n$ and $F_n$ will be ported to $W_n$ coefficients.
*   **Code:** The `Surferbot` solver will continue to use $\Psi_n$ internally for numerical stability, but all high-level analysis and storage will map back to the $W_n$ representation.
