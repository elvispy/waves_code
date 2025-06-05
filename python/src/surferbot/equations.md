f# Equations of motion for a flexible raft

## Case 1: DtN Operator, 1D problem

### Bernoulli equation
$$
N\hat{\phi} = \frac{\sigma}{\rho g} N\hat{\phi}_{xx} + \frac{\omega^2}{g}\hat{\phi} + \frac{4i\nu\omega}{g}\hat{\phi}_{xx}
$$

For $|x| > L/2$

### Force balance inside the raft
$$
EI\frac{\partial^4 \eta}{\partial x^4} = \rho_R \omega^2 \hat{\eta} + \hat{F}_{z} \delta_{x_A} - \rho(i\omega\hat{\phi} + g\hat{\eta} - 2\nu\hat{\phi}_{xx})
$$
For $|x| \leq L/2$

Q1. Should I multiply the pressure by a cosine? A. No. It would make the formulation nonlinear. 

### Kinematic boundary conditions
$$
N\hat{\phi} = i\omega\hat{\eta}
$$
$\forall x$


### Radiative boundary conditions
$$
\phi_x = \pm ik \phi
$$
for $x = \mp \ell$

## No Penetration boundary condition

$$
\hat{\phi}_z = 0, \quad z = - H
$$

---
# Idea 2: 2D problem, function on $\phi$ only

Analogous to saying that $p = -\rho (i \omega \phi + g \eta - 2 \nu \phi_{xx})$, we can replace the kinematic boundary conditions to get rid of $\eta$, $\eta = \frac{1}{i \omega} \phi_z$! (ND version is $\tilde{\eta} = \frac{1}{i \omega t_c} \tilde{\phi}_z$)

## Laplace's equation

$$
\Delta \hat{\phi} = 0, \quad x \in \Omega
$$
### Non dimensional equation
$$
\Delta \tilde{\phi} = 0
$$

## Bernoulli's equation
$$
\hat{\phi}_z = \frac{\sigma}{\rho g} \hat{\phi}_{zxx} + \frac{\omega^2}{g}\hat{\phi} + \frac{4i\nu\omega}{g}\hat{\phi}_{xx}
$$
At $z = 0$, $|x| > L/2$

### Non dimensional equation

$$
\tilde{\phi}_z = \frac{\sigma}{\rho_W g L_c^2} \tilde{\phi}_{xxz} + L_c \frac{\omega^2}{g} \tilde{\phi} + \frac{4 i \nu \omega}{g L_c} \tilde{\phi}_{xx}
$$

For $z = 0$. $|x| > 1/2$
## The beam equation inside the raft

$$
\frac{EI}{i \omega}\frac{\partial^5 \phi}{\partial z \partial x^4} = \frac{\rho_R \omega}{i} \hat{\phi}_z + \hat{F}_{z} \delta_{x_A} - \rho(i\omega\hat{\phi} + \frac{g}{i \omega}\hat{\phi}_z - 2\nu\hat{\phi}_{xx}) + \sigma \eta_x \delta_{\pm L/2}
$$

For $z = 0$, $|x| \leq L/2$

### Non dimensional equation

$$
\frac{EI}{i \omega L_c^3 t_c}\frac{\partial^5 \tilde{\phi}}{\partial z \partial x^4} = \frac{\rho_R \omega L_c}{i t_c} \tilde{\phi}_z + \frac{\hat{F}_{z}}{L_c} \tilde{\delta}_{x_A} - \rho\left(i\omega \frac{L_c^2}{t_c}\tilde{\phi} + \frac{g L_c}{i \omega t_c}\tilde{\phi}_z - 2\frac{\nu}{t_c} \tilde{\phi}_{xx}\right) + \frac{\sigma}{i \omega t_c L_c} \tilde{\phi}_{xz} \tilde{\delta}_{\pm 1/2}
$$
Which is equivalent to 

$$
\frac{EI t_c}{i \omega L_c^3 m_c} \tilde{\phi}_{xxxxz} = \frac{\rho_R \omega L_c t_c}{i m_c} \tilde{\phi}_z + \frac{\hat{F}_{z} t_c^2}{m_c L_c} \delta_{x_A} - \frac{\rho t_c}{m_c}\left(i\omega L_c^2\tilde{\phi} + \frac{g L_c}{i \omega}\tilde{\phi}_z - 2\nu \tilde{\phi}_{xx}\right) + \frac{\sigma t_c}{i \omega m_c L_c} \tilde{\phi}_{xz} \tilde{\delta}_{\pm 1/2}
$$

For $|x| \leq 1/2$

### Radiative boundary conditions
$$
\phi_x = \pm ik \phi
$$
for $x = \mp \ell, \forall z$

### Non dimensional equation
$$
\tilde{\phi}_x = \pm i k L_c \tilde{\phi}
$$
For $ x = \mp \ell/L_c$


## No Penetration boundary condition

$$
\hat{\phi}_z = 0, \quad z = - H
$$


Q2. The system is slightly overconstrained because of the Boundary conditions. Which one should be applied?

Q3. A fifth order differential equation has a $dx^5$ in the denominator. We should get rid of that numerical instability 

## Thrust equation

$$
F_D = \frac{1}{2} C_D \rho_W L U^2 = \frac{2}{3} \rho_W \sqrt{\nu L U^3}
$$
$$
F_T = \overline{\int_S (p - p_{\infty}) (\bm{n} \cdot \bm{i}) dx} + \sigma \left(\overline{\eta_x}|_{x = -L/2} - \overline{\eta_x}|_{x = +L/2}\right)
$$
$$
F_T = -\overline{\int_S Re\{\hat{p} e^{i \omega t}\} Re\{\hat{\eta}_x e^{i \omega t}\} dx} + \sigma \left(Re\{\hat{\eta}_x|_{x = -L/2}\} - Re\{\hat{\eta}_x|_{x = L/2}\}\right)
$$
$$
 = -\frac{1}{2}\int_S Re\{\hat{p}\} Re\{\hat{\eta}_x \} + Im\{\hat{p}\} Im\{\hat{\eta}_x \} dx + \sigma \left(Re\{\hat{\eta}_x|_{x = -L/2}\} - Re\{\hat{\eta}_x|_{x = L/2}\}\right)
$$

$$
U = \left(\frac{1}{\nu L}\left(\frac{3 F_T}{2 \rho_W}\right)^2\right)^{1/3}
$$


### Pressure on the raft

$$
\hat{p}
$$

## Dispersion relation derivation

We guess a solution of the form $\phi = f(z) e^{ik(x + \omega t)}$. We want to calculate the wavenumber $k$
with respect to the frequency of oscillation.

### BC1: Laplace's equation

$$
\phi_{xx} + \phi_{zz}  = \left( - k^2 f(z) +  f''(z)\right) e^{ikx} = 0 \implies f''= k^2 f
$$
Therefore, $f = A_1 e^{kz} + A_2 e^{-kz}$
### BC2: No-penetration BC

$$
\phi_z(z = -H) = (A_1 k e^{kH} - A_2 k e^{-kH}) e^{ikz} = 0
$$
This way, $A_1 = A_2 e^{2kH}$.

WLOG, we can set $A_2=1$ as the problem is linear. Thus, $\phi = (e^{kz + 2kH} + e^{-kz}) e^{ikz}$.

### THe bernoulli equation

$$
\phi_z = \frac{\sigma}{\rho g} + \frac{\omega^2}{g} \phi + \frac{4 \nu i \omega}{g} \phi_{xx}
$$

We replace the form of $\phi$ we obtain in the previous section, therefore
$$
k(e^{2kH} - 1) e^{ikx} = \frac{\sigma}{\rho g} (k e^{2kH} - k)(-k^2 e^{ikx}) + \frac{\omega^2}{g} (e^{2kH} + 1) e^{ikx} - \frac{k^2 4 i \nu \omega}{g} (e^{2kH} + 1) e^{ikx}
$$

We multiply everyting by $e^{-kh}e^{-kH}(e^{kH} + e^{-kH}) g$

Getting
$$
k tanh(kH) g = \frac{-\sigma}{\rho} k^3 tanh(kH) + \omega^2 - k^2 4 i \nu \omega
$$

