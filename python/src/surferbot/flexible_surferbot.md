Below is a “road-map” that matches each block assembled in the script to its continuous, time-harmonic governing equation (written for a 2-D domain with horizontal coordinate $x$ and vertical coordinate $z$; the free surface is at $z=0$ and $z>0$ points into the water).  Hats $(\,\widehat{}\;)$ indicate non-dimensional variables after the scaling introduced just above the **Derived dimensional parameters** section of the script.

---

### Unknowns carried in `x[:,:,k]`

| index $k$ | stored quantity               | continuous meaning                               |
| --------- | ----------------------------- | ------------------------------------------------ |
| 0         | $\phi$                        | velocity potential $\phi(\widehat x,\widehat z)$ |
| 1         | $\partial_{\widehat x}\phi$   |                                                  |
| 2         | $\partial_{\widehat x}^2\phi$ |                                                  |
| 3         | $\partial_{\widehat x}^3\phi$ |                                                  |
| 4         | $\partial_{\widehat x}^4\phi$ |                                                  |

Extra rows added later in `A` simply enforce
$\partial_{\widehat x}\phi-\phi_{,1}=0,\;
\partial_{\widehat x}\phi_{,1}-\phi_{,2}=0,\ldots$ so that the high-order $x$-derivatives are algebraic unknowns instead of being formed on the fly.

---

## 1.  **Block E2 – Laplace equation in the fluid bulk**

```python
E2 = (d_dx**2 + d_dz**2)      # built before stacking
```

Imposes

$$
\nabla^2_{\!\widehat{x},\widehat{z}}\;\phi = 0 \qquad
\text{for } -\widehat{d}\le\widehat z\le 0 ,
$$

the classical potential-flow statement of mass conservation in an incompressible, irrotational fluid. ([web.mit.edu][1])

---

## 2.  **Block E1 – Free-surface conditions**

`E1` is first filled with the free-surface **linearised Bernoulli condition** (rows whose nodes satisfy `x_free`) and then **over-written by the beam–raft equation** inside the contact region (`x_contact`).

### 2a.  Outside the raft (free surface)

```python
C11 d_dz + C13 I       # multiplies  φ
C12 d_dz + C14 I       # multiplies  ∂x²φ
```

These coefficients correspond to

$$
i\widehat\omega\,\phi
\;+\; \widehat g\,\frac{\partial\phi}{\partial\widehat z}
\;-\;\frac{\widehat\sigma}{\widehat\rho}\;
\frac{\partial}{\partial\widehat z}\!\left(\partial_{\widehat x}^2\phi\right)
\;+\;4\widehat\nu\,
\frac{\partial^2\phi}{\partial\widehat z^{2}}
=0 ,
$$

which is the *dynamic* free-surface condition with gravity, capillarity and the viscous correction $4\nu\phi_{zz}$ derived by Lamb for small-amplitude waves.  Using the usual kinematic relation $i\widehat\omega\,\eta=\phi_{,\widehat z}$ eliminates the separate variable $\eta$. ([web.mit.edu][1], [arxiv.org][2])

### 2b.  Under the raft (fluid-structure coupling)

Inside the contact patch those rows are replaced by `E12`, whose leading terms are

```python
   + C21 d_dz φ₍4₎  ⟹ EI ∂⁴η/∂x⁴
   + C22 d_dz φ      ⟹ −ρ_r ω² η
   + C24 I φ         ⟹ ρ ∂φ/∂t  (added water inertia)
   + C25 d_dz φ      ⟹ ρ g η
   + C26 I φ₂        ⟹ −2ρν ∂²η/∂t∂z  (linear drag)
```

After substituting $\eta=\phi_{,\widehat z}/(i\widehat\omega)$ these terms give the frequency-domain **Euler–Bernoulli beam equation** for a floating, flexible raft loaded by fluid pressure plus the distributed external Gaussian force (introduced through the right-hand side with `weights`).  In dimensional form

$$
EI\,\partial_x^{4}\eta
\;-\;m_r\omega^{2}\eta
\;=\;p_f(x)\;+\;F\,\delta(x-x_m),
$$

where the pressure $p_f$ is supplied by the linearised Bernoulli relation above. ([en.wikipedia.org][3])

---

## 3.  **Block E3 – Bottom boundary (impermeable bed)**

```python
E3 = d_dz ;  E3[:, :-1] = 0
```

Leaves only rows at the last $z$-index, enforcing

$$
\left.\frac{\partial\phi}{\partial\widehat z}\right|_{\widehat z=-\widehat d}=0 ,
$$

i.e. no normal flow through the tank bottom.

---

## 4.  **Block E4 – Lateral radiation boundary**

```python
φ_x ± i k φ = 0  (applied at x = ±L/2)
```

implemented with `C31` and `C32`, is the discrete **Sommerfeld radiation condition** that lets waves exit the computational domain without reflection. ([en.wikipedia.org][4])

---

## 5.  **Auxiliary rows (1 – 4 in `A`)**

Those rows impose

$$
\phi_{,1}-\partial_{\widehat x}\phi=0,\;
\phi_{,2}-\partial_{\widehat x}\phi_{,1}=0,\;
\dots,\;
\phi_{,4}-\partial_{\widehat x}\phi_{,3}=0,
$$

so that every horizontal derivative up to fourth order is an independent unknown; this converts the fifth-order beam PDE into a first-order system that the sparse solver can handle efficiently.

---

### Putting it all together

The final sixth-order tensor `A` therefore encodes the following coupled problem (in non-dimensional form and after replacing $\eta$ by $\phi_{,\widehat z}$):

$$
\begin{cases}
\nabla^{2}\phi = 0, & -\widehat d<\widehat z<0,\\[6pt]
i\widehat\omega\,\phi + \widehat g\,\phi_{,\widehat z}
- \dfrac{\widehat\sigma}{\widehat\rho}\,
      \phi_{,\widehat z\widehat x\widehat x}
+ 4\widehat\nu\,\phi_{,\widehat z\widehat z}=0,
& |\widehat x|>\tfrac{\widehat L_r}{2},\;\widehat z=0,\\[10pt]
EI\,\partial_{\widehat x}^{4}\eta
- \widehat m_r \widehat\omega^{2}\eta
-2\widehat\rho\widehat\nu\,\eta
- \widehat\rho\frac{\partial\phi}{\partial t}
- \widehat\rho\widehat g\,\eta
= \widehat F \, \delta(\widehat x-\widehat x_m),
& |\widehat x|\le\tfrac{\widehat L_r}{2},\;\widehat z=0,\\[10pt]
\phi_{,\widehat z}=0, & \widehat z=-\widehat d,\\[6pt]
\phi_{,\widehat x}\mp i\widehat k\,\phi=0, & \widehat x=\pm\tfrac{\widehat L}{2}.
\end{cases}
$$

Together with the derivative-definition rows, these give exactly the $5\,N\,M$ scalar equations that `solve_tensor_system` inverts.

---

**In short:**

*E1* contains the free-surface Bernoulli condition or, under the raft, the Euler–Bernoulli beam relation;
*E2* enforces Laplace’s equation in the fluid interior;
*E3* imposes an impermeable bottom;
*E4* applies outgoing-wave radiation at the domain ends;
the remaining rows of `A` simply define $\partial_{\widehat x}^n\phi$ ($n=1\ldots4$).  Those five blocks jointly form the fully coupled, linear, time-harmonic water-wave / flexible-raft system solved by the script.

[1]: https://web.mit.edu/13.012/www/handouts/2003/waves.pdf?utm_source=chatgpt.com "[PDF] 13.012 reading 6: linear free surface waves - MIT"
[2]: https://arxiv.org/pdf/2303.12158?utm_source=chatgpt.com "[PDF] arXiv:2303.12158v1 [physics.flu-dyn] 21 Mar 2023"
[3]: https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory?utm_source=chatgpt.com "Euler–Bernoulli beam theory"
[4]: https://en.wikipedia.org/wiki/Sommerfeld_radiation_condition?utm_source=chatgpt.com "Sommerfeld radiation condition"
