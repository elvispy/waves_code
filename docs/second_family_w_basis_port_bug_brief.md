# Second-Family `\Psi_n \to W_n` Port Bug Brief

For more context on the notation and the reduced-model derivations, read: `/Users/eaguerov/Library/Mobile Documents/iCloud~md~obsidian/Documents/BrownObsidian/Research/Surferbot/Resonance distilled 2.md`.

## Purpose

This note explains a failed attempt to replace the current numerical modal basis `\Psi_n` with the analytical free-free basis `W_n` in the second-family branch predictor.

The goal is to give a clean handoff for further work:

1. what the two bases are,
2. why replacing `\Psi_n` by `W_n` looked mathematically reasonable,
3. what was actually changed in code,
4. why the change failed,
5. what the real open problem is.

## Context

We are studying the uncoupled beam-end second family, especially the first nontrivial branch, described numerically by a curve

$$
x_M = x_M^\*(EI).
$$

For the high-`EI` regime, the current empirical predictor works well. The working predictor is based on the modal decomposition used in the Julia codebase.

The attempted change was to drop the numerical basis `\Psi_n` and use the analytical free-free modes `W_n` everywhere instead.

That change **failed** the agreed non-negotiable KPI:

$$
\frac{\|x_M^{\Psi}(EI)-x_M^{W}(EI)\|_2}{\|x_M^{\Psi}(EI)\|_2} \le 10^{-1}.
$$

The observed relative `L2` distance was

$$
3.580591\times 10^{-1},
$$

so the port is currently **not acceptable**.

---

## 1. What are `W_n` and `\Psi_n`?

### 1.1 Analytical basis `W_n`

`W_n` denotes the raw free-free beam modes:

- rigid translation,
- rigid rotation,
- elastic free-free modes.

For elastic modes, the continuous analytical shape is

$$
W_n(x) = \sin(\beta_n x) + \sinh(\beta_n x)
+ \alpha_n \left(\cos(\beta_n x) + \cosh(\beta_n x)\right),
$$

with the free-free eigenvalue condition

$$
\cosh(\beta_n L)\cos(\beta_n L)=1.
$$

In the current Julia code, these are built in `freefree_mode_shape(...)`.

### 1.2 Numerical basis `\Psi_n`

`\Psi_n` is **not** a different physical model. It is the discrete basis that the Julia code actually uses for projection.

Start from the sampled raw basis matrix

$$
\Phi = \begin{bmatrix}
W_0(x_1) & W_1(x_1) & \cdots \\
W_0(x_2) & W_1(x_2) & \cdots \\
\vdots   & \vdots   & 
\end{bmatrix},
$$

where the first columns include the rigid modes and the later columns include the elastic free-free modes sampled on the raft grid.

The raft grid is equipped with trapezoidal quadrature weights

$$
w_i,
$$

which define the discrete weighted inner product

$$
\langle u,v\rangle_W = \sum_i u_i v_i\, w_i.
$$

The code then applies weighted modified Gram-Schmidt to `\Phi`, producing an orthonormal basis

$$
\Psi = \begin{bmatrix}
\Psi_0(x_1) & \Psi_1(x_1) & \cdots \\
\Psi_0(x_2) & \Psi_1(x_2) & \cdots \\
\vdots      & \vdots      &
\end{bmatrix},
$$

such that

$$
\Psi^\top \operatorname{diag}(w)\,\Psi \approx I.
$$

So the relationship is:

$$
\Psi = \Phi R^{-1}
$$

for some upper-triangular change-of-basis matrix `R` induced by weighted orthonormalization.

This means `\Psi_n` is **not generally just a scalar multiple of `W_n`**.

It is a basis for the same discrete subspace, but with a different coordinate system.

---

## 2. Why did replacing `\Psi_n` by `W_n` look reasonable?

The original motivation was mathematically sensible:

1. `W_n` is the analytical basis used in the notes.
2. The first-family resonance picture is already naturally expressed in `W_n`.
3. If we want reduced-order theory in closed form, `W_n` is the natural basis.
4. The empirical checks suggested that low-mode physics was already captured well:

   - `F_n` delta-load approximation was decent for low modes,
   - `q_n` force-balance formula was good,
   - the high-`EI` branch was well predicted by a low-mode reduced equation.

So the proposed change was:

> Treat `W_n` as canonical, port all coefficients offline into the `W` basis, and verify that the empirical branch predictor stays stable.

This is mathematically legitimate **if done consistently**.

If a field `\eta` is expanded in the two bases,

$$
\eta \approx \Psi c_{\Psi} \approx \Phi c_W,
$$

then the two coefficient vectors are related by

$$
c_W = T\,c_{\Psi},
$$

where

$$
T = (\Phi^\top W \Phi)^{-1}\Phi^\top W \Psi,
$$

with

$$
W = \operatorname{diag}(w).
$$

So the proposed offline port was:

$$
q^W = T q^\Psi,\qquad Q^W = T Q^\Psi,\qquad F^W = T F^\Psi.
$$

On paper, that is exactly the correct way to rewrite the same discrete field in a new basis.

---

## 3. What was the actual bug?

There were two separate experiments. One was conceptually wrong, and one was conceptually right but numerically failed.

### 3.1 First bad experiment: uncontrolled basis swap

In one overlay experiment, the code did **not** only replace `F_n` and `q_n`.

It also changed the basis from `\Psi_n` to raw `W_n`.

That means the comparison was effectively:

$$
\text{old predictor} = \text{projection in } \Psi
$$

versus

$$
\text{new predictor} = \text{projection in } W
$$

with different normalization, different endpoint weights, and different reconstruction coordinates.

That experiment was invalid as a test of whether “analytical `F_n` and `q_n` are good enough,” because it changed too many ingredients at once.

This was a **real bug in the experiment design**.

### 3.2 Second experiment: controlled offline basis port

The more careful attempt was:

1. keep the same empirical branch predictor structure,
2. build the raw discrete `W` basis `\Phi`,
3. compute the offline transform

$$
T = (\Phi^\top W \Phi)^{-1}\Phi^\top W \Psi,
$$

4. port coefficients

$$
q^W = T q^\Psi,\qquad Q^W = T Q^\Psi,\qquad F^W = T F^\Psi,
$$

5. rerun the empirical branch predictor in the `W` basis,
6. compare the resulting branch with the existing `\Psi`-based predictor.

This second experiment is the correct one.

It **still failed**.

---

## 4. Why did the controlled port fail?

The failure is not that the change-of-basis formula is wrong. The formula is correct.

The failure is that the branch predictor is **not stable** under this basis change.

### 4.1 Observed failure

For the high-`EI` window, the stability check produced:

$$
\text{relative }L^2 = 3.580591\times 10^{-1},
$$

which violates the acceptance threshold

$$
10^{-1}.
$$

Even worse, the number of recovered branch points changed:

- `\Psi`-basis predictor: `16` points,
- `W`-basis predictor: `9` points.

So the predictor is not merely drifting slightly; it is changing branch structure.

### 4.2 Numerical evidence

Representative drift:

$$
\begin{aligned}
EI &= 2.289016\times 10^{-4}, && x_M^\Psi = 0.3015,\quad x_M^W = 0.3693,\\
EI &= 2.620431\times 10^{-4}, && x_M^\Psi = 0.3140,\quad x_M^W = 0.4126,\\
EI &= 2.996872\times 10^{-4}, && x_M^\Psi = 0.3277,\quad x_M^W = 0.4599.
\end{aligned}
$$

The raw `W` Gram matrix was also not especially well conditioned:

$$
\kappa(\Phi^\top W \Phi) = 4.816866\times 10^3.
$$

That is not catastrophic, but it is large enough that cancellation-sensitive predictions can drift.

### 4.3 Why this is possible even if the change of basis is mathematically valid

The branch is found from a root condition of the form

$$
S(EI,x_M)=0
\qquad\text{or}\qquad
A(EI,x_M)=0.
$$

These are **cancellation conditions**. Root locations are sensitive to:

- relative modal scaling,
- endpoint weights,
- numerical conditioning,
- basis truncation.

Even if two bases span the same subspace, the reduced-order predictor built in those coordinates need not be numerically stable unless the projection, reconstruction, and truncation are all reformulated consistently.

This is the key issue:

> A basis change that is harmless for full-field reconstruction can still be harmful for a cancellation-based root predictor.

That is exactly what happened.

---

## 5. Tradeoffs of `\Psi_n` versus `W_n`

### 5.1 Advantages of `W_n`

`W_n` is attractive because:

1. it is the analytical basis of the notes,
2. reduced equations are naturally expressed in it,
3. it makes the mechanism easier to interpret physically,
4. it avoids the feeling that `\Psi_n` is “legacy numerical notation.”

### 5.2 Advantages of `\Psi_n`

`\Psi_n` solves a real numerical problem:

1. it is orthonormal on the discrete raft grid,
2. projections are well-conditioned,
3. coefficient interpretation inside the code is stable,
4. the current empirical branch predictor already works well in this basis.

So `\Psi_n` is not a new physics basis, but it is also not meaningless legacy baggage.

It is the stable discrete coordinate system currently used by the working predictor.

### 5.3 Main practical tradeoff

Using `W_n` everywhere is conceptually cleaner, but only if the empirical predictor remains stable.

Right now, that stability check fails.

So the real tradeoff is:

- `W_n`: cleaner analytical language
- `\Psi_n`: currently safer numerical predictor

Until the stability issue is resolved, dropping `\Psi_n` would be physically and numerically risky.

---

## 6. What was mathematically sound, and what failed?

### Sound parts

These parts were mathematically sound:

1. wanting a `W_n`-based reduced theory,
2. using the offline transform

$$
T = (\Phi^\top W \Phi)^{-1}\Phi^\top W \Psi,
$$

3. porting coefficients via

$$
q^W = T q^\Psi,\qquad Q^W = T Q^\Psi,\qquad F^W = T F^\Psi.
$$

### Failed part

What failed was the KPI:

> the empirical branch predictor did **not** remain stable after the port.

That means the mathematical change of basis was **not enough** to preserve the reduced predictor in practice.

So the current conclusion is not “`W_n` is wrong.”

The current conclusion is:

> the naive offline `\Psi \to W` port is not predictor-preserving.

---

## 7. What should the intern do next?

The next task is **not** to replace `\Psi_n` everywhere immediately.

The next task is to determine **why** the `W`-basis branch predictor drifts so much.

Concrete tasks:

1. Re-derive the empirical branch predictor directly in the `W` basis.
2. Check whether endpoint weights and truncation were transformed consistently.
3. Compare the following three predictors:

   - current `\Psi`-basis empirical predictor,
   - naive offline-ported `W`-basis predictor,
   - re-derived `W`-basis predictor built from scratch.

4. Keep the same acceptance gate:

$$
\frac{\|x_M^{\Psi}-x_M^{W}\|_2}{\|x_M^{\Psi}\|_2} \le 10^{-1}.
$$

If that gate still fails, then `\Psi_n` must remain in the empirical predictor path.

---

## 8. Bottom line

The bug is:

> we assumed that an offline change of basis from `\Psi_n` to `W_n` would preserve the empirical branch predictor.

That assumption is false, at least in the current implementation.

The change was mathematically motivated, but the predictor is too sensitive to basis/conditioning/truncation for the naive port to work.

## Resolution (April 19, 2026)

**STATUS: RESOLVED**

The "bug" was not a mathematical error, but a failure to account for the **discrete non-orthogonality** of the $W_n$ basis on the raft grid. 

- The "40% error" in root prediction was caused by a basis-normalization discrepancy ($q^W \approx 311 \cdot q^\Psi$ for mode 1).
- The "drift" was caused by ignoring the off-diagonal terms of the Gram matrix $\mathbf{G}$ when assuming a diagonal modal balance.

**Decision:** We are sticking with $W_n$ as the canonical basis. See `docs/architectural_decisions.md` for the formal ADR.

