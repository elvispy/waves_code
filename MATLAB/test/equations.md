# Beam equation in fourier space
We analyze the following equation in complex notation
$$
EI\hat\eta_{xxxx} - \rho_R\omega^2 \hat\eta = d\hat p - \hat f,
\qquad |x|\le L/2,
\tag{1}
$$
With suitable boundary conditions. The free-free beam normal modes are a complete orthonormal set, $\{\psi_n\}_{n\geq0}$. For uniform properties they satisfy
$$
\psi_n''''=\beta_n^4 \psi_n,\qquad \psi_n''=\psi_n'''=0 \ \text{at}\ x=\pm L/2,
\tag{2}
$$
Expand the raft elevation as
$$
\hat\eta(x)=\sum_{n=0}^{\infty}\hat q_n\psi_n(x),
\qquad
\widehat{\eta_t} = i\omega \hat\eta =\sum_{n=0}^{\infty}i\omega\hat q_n\psi_n(x),
\tag{3}
$$

Multiply (1) by $\psi_n$ and integrate over $[-L/2,L/2]$. Using (3) and orthogonality,
$$
EI\int \psi_n\hat\eta_{xxxx}dx - \rho_R\omega^2\int \psi_n\hat\eta dx
= \int \psi_n(d\hat p - \hat f)dx.
\tag{7}
$$

Because $\hat\eta_{xxxx}=\sum_m \hat q_m \psi_m''''=\sum_m \hat q_m \beta_m^4 \psi_m$, orthogonality gives

$$
\int \psi_n\hat\eta_{xxxx}dx = \beta_n^4 \hat q_n,\qquad
\int \psi_n\hat\eta dx = \hat q_n.
\tag{8}
$$
So (7) becomes
$$
\Big(EI\beta_n^4 - \rho_R\omega^2\Big)\hat q_n
= \hat Q_n - \hat F_n,
\tag{9}
$$
where we define
$$
\hat Q_n := \int_{-L/2}^{L/2} d\hat p(x)\psi_n(x)dx,\qquad
\hat F_n := \int_{-L/2}^{L/2} \hat f(x, x_M)\psi_n(x)dx.
\tag{10}
$$
$\partial F_n /\partial x_M = 0$
# Calculating $\hat{Q}_n$ 
We start with our pressure definition:
$$
\hat p = -\rho\Big(i\omega\hat\phi + g\hat\eta + 2\nu \hat\phi_{zz}\Big)\quad(z=0).
\tag{11}
$$
We divide the pressure contribution into two parts: the hydrodynamic pressure, and static pressure. To leading order (small viscosity assumption), the term that contributes to added mass in the hydrodynamic pressure is
$$
\hat p_{\rm dyn}= -\rho i\omega\hat\phi - 2 \rho \nu \hat\phi_{zz}, \quad \hat p_{rest} = - \rho g \hat\eta
\tag{12}
$$

Split $\hat Q_n=\hat Q_{n,\mathrm{dyn}}+\hat Q_{n,\mathrm{rest}}$ using (3). The restoring part is explicit:
$$
\hat Q_{n,\mathrm{rest}}
=\int d\hat p_{\mathrm{rest}}\psi_n dx
=-d\rho g\int \hat\eta\psi_n dx
=-d\rho g\hat q_n  .
\tag{13}
$$
### Estimating $\hat Q_{n, dyn}$
The dynamic part is, could be calculated a posteriori. To make an a priori estimate, we approximate the fluid’s response by the potential flow solution:
$$
\hat\phi(x,z) \approx A\frac{\cosh k(z+H)}{\cosh(kH)}e^{ikx}.
\tag{14}
$$
Then at $z=0$,
$$
\hat\phi_z = k\tanh(kH)\hat\phi 
\tag{15}
$$
Using the inviscid kinematic condition (to leading order), $\partial_t\eta=\phi_z$, gives in complex amplitudes
$$
i\omega\hat\eta = \hat\phi_z = k\tanh(kH)\hat\phi
\quad\Rightarrow\quad
\hat\phi = \frac{i\omega}{k\tanh(kH)}\hat\eta.
\tag{16}
$$
Insert (6) into $\hat p_{\rm dyn}=-\rho i\omega \hat\phi$:
$$
\hat p_{\rm dyn}\approx -\rho i\omega \left(\frac{i\omega}{k\tanh(kH)}\hat\eta\right)
= \rho\frac{\omega^2}{k\tanh(kH)}\hat\eta.
\tag{17}
$$
Thus, the dynamic fourier coefficient becomes
$$
\hat Q_{n, dyn} = \int d \hat p_{dyn} \psi_n dx \approx \frac{d\rho \omega^2}{\beta_n \tanh(\beta_n H)} \hat q_n \tag{18}
$$

Obs: When the viscous part is not disregarded, the hydrodynamic component is usually written as
$$
\hat Q_{n,\mathrm{dyn}} = \Big(\omega^2 m_{a,n}(\omega) + i\omega b_n(\omega)\Big)\hat q_n,
\tag{19}
$$
where $m_{a, n}$ is the added mass associated with component $n$. In our case, we estimate
$$
m_{a, n} = \frac{d\rho}{k \tanh(kH)}
$$
### Fourier coefficients
Insert (11)–(12) into (9):
$$
\Big(EI\beta_n^4 - \rho_R\omega^2\Big)\hat q_n  
=
\frac{d\rho \omega^2}{k \tanh(k H)}\hat q_n
- d\rho g\hat q_n  
- \hat F_n.
\tag{20}
$$

Bring everything to the left except the force:
$$
\Big(EI\beta_n^4 + d\rho g\Big)\hat q_n
-\omega^2\Big(\rho_R + \frac{d\rho}{k \tanh(k H)}\Big)\hat q_n
  = -\hat F_n.
  \tag{22}
  $$
$$
\hat q_n
  = \frac{-\hat F_n}{EI\beta_n^4 + d\rho g -\omega^2\Big(\rho_R + m_{a,n}(\omega)\Big)
}\tag{23}
$$  
### 1) $\omega-EI$ resonance
Fix $F_n$. The closer the term in parenthesis is to zero, the bigger $\hat q_n$ has to be to compensate. The resonance criteria is thus
$$
\Big(EI\beta_n^4 + d\rho g
-\omega^2\Big(\rho_R + m_{a,n}(\omega)\Big)\Big) = 0
$$
$$
\omega = \sqrt{\frac{EI\beta_n^4 + d\rho g}{\rho_R + \frac{d\rho}{k \tanh(k H)}}}
$$

We also recall 
$$
\frac{\omega^2 - 4i\omega k^2}{g + \frac{\sigma}{\rho} k^2} = k \tanh(kH)
$$
We can also argue that the dominant wavenumber when on resonance is $k \approx \beta_n$, giving the frequency-independent approximation near the raft:
$$
\omega = \sqrt{\frac{EI\beta_n^4 + d\rho g}{\rho_R + \frac{d\rho}{\beta_n \tanh(\beta_n H)}}}
$$
Note: this is independent of the motor position!