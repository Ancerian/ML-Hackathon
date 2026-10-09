# THEORY.md — Pre-Registration Specification for Task T12 (Field Line Tracer)

**Pre-registration Status:** FROZEN PRIOR TO FINAL RUN  
**Pre-registration Date:** 2026-10-09  
**Pilot Command:** `PYTHONPATH=tokamak-3d-viz/src:src "fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t12/pilot_t12.py"`  
**Pilot Dataset:** DIII-D shot `d3d_shot_00000a10ac.parquet`, flat-top frame 58 ($q_{95} = 4.220$, $F = 3.404\text{ m}\cdot\text{T}$, $R_0 = 1.725\text{ m}$, $Z_0 = -0.067\text{ m}$, $\Delta\psi = 0.2992\text{ Wb/rad}$).

---

## 1. Physical Foundations and Coordinate Systems

### 1.1 Canonical Coordinates $(\psi_t, \theta^*, \phi)$ vs Cartesian $(R, Z, \phi)$ (E19)
Magnetic field lines in toroidal geometry satisfy:
$$\frac{d\mathbf{x}}{d\phi} = \frac{R B_R}{B_\phi} \hat{\mathbf{R}} + \frac{R B_Z}{B_\phi} \hat{\mathbf{Z}}$$

1. **Canonical Magnetic Coordinates $(\psi_t, \theta^*, \phi)$:**
   - Coordinate: Straight-field-line poloidal angle $\theta^*$.
   - Momentum: Toroidal magnetic flux per radian $\psi_t = \frac{1}{2\pi} \int B_\phi dS_p = \int_0^{\psi_p} q(\psi') d\psi'$.
   - Time: Toroidal angle $\phi$.
   - Hamiltonian: Poloidal magnetic flux $H(\psi_t, \theta^*, \phi) = \psi_p(\psi_t) + \delta H(\psi_t, \theta^*, \phi)$.
   - Hamilton's equations:
     $$\frac{d\theta^*}{d\phi} = \frac{\partial H}{\partial \psi_t} = \frac{1}{q(\psi_t)} + \frac{\partial \delta H}{\partial \psi_t}$$
     $$\frac{d\psi_t}{d\phi} = -\frac{\partial H}{\partial \theta^*} = -\frac{\partial \delta H}{\partial \theta^*}$$
   - **Symplecticity Claim:** Symplecticity ($\det J \equiv 1$ and preservation of symplectic 2-form $d\psi_t \wedge d\theta^*$) is mathematically valid and claimed **exclusively** in canonical coordinates $(\psi_t, \theta^*, \phi)$.
   - **Integrator:** Implicit Midpoint Rule with Newton-Raphson solver (residual stopping tolerance $\|r\|_2 \le 10^{-12}$). Theoretical convergence order: $O(\Delta\phi^2)$.

2. **Cylindrical/Cartesian Coordinates $(R, Z, \phi)$:**
   $$\frac{dR}{d\phi} = -\frac{R}{F(\psi)} \frac{\partial \psi_{\text{total}}}{\partial Z}, \quad \frac{dZ}{d\phi} = +\frac{R}{F(\psi)} \frac{\partial \psi_{\text{total}}}{\partial R}$$
   - Flow preserves weighted measure $R dR dZ$, but $(R, Z)$ are non-canonical with respect to $\phi$ time.
   - **No Symplecticity Claim:** We do NOT claim $\det J = 1$ in $(R, Z)$. Instead, in the axisymmetric case ($\delta\psi = 0$), the poloidal flux $\psi(R, Z)$ is an exact first integral ($d\psi/d\phi = 0$). We evaluate accuracy via the drift of $\psi$.
   - **Integrator:** Classical 4th-order Runge-Kutta (RK4) with alive-masking to prevent NaNs on wall strike. Theoretical convergence order: $O(\Delta\phi^4)$.

---

## 2. Straight-Field-Line Angle $\theta^*(R, Z)$ and Perturbation Representation

### 2.1 Branch Cut Elimination via Trigonometric Splines
The straight-field-line angle $\theta^*$ satisfies:
$$\theta^*(R, Z) = \theta_{\text{geo}} + \Delta(\psi_N, \theta_{\text{geo}})$$
Direct interpolation of $\theta^*(R, Z)$ possesses a $2\pi$ branch cut discontinuity across the midplane, corrupting spline derivatives $\nabla\theta^*$.
To eliminate the branch cut, we construct 2D smooth grids for:
$$C_{\theta^*}(R, Z) = \cos(\theta^*(R, Z)), \quad S_{\theta^*}(R, Z) = \sin(\theta^*(R, Z))$$
Harmonic modes $\cos(m\theta^* - n\phi + \alpha)$ are evaluated analytically via de Moivre's formula:
$$(\cos\theta^* + i\sin\theta^*)^m = \cos(m\theta^*) + i\sin(m\theta^*)$$
$$\cos(m\theta^* - n\phi + \alpha) = \cos(m\theta^*)\cos(n\phi - \alpha) + \sin(m\theta^*)\sin(n\phi - \alpha)$$
which is everywhere $C^1$ smooth and free of branch cuts.

### 2.2 Boundary Restriction ($\psi_N \le 0.95$)
Because $B_p \to 0$ at the X-point, the pitch angle $\nu = \frac{F}{R^2 B_p}$ and $\theta^*$ exhibit a logarithmic singularity at $\psi_N = 1.0$.
Perturbations are multiplied by a $C^1$ smooth taper envelope:
$$f_{\text{env}}(\psi_N) = \begin{cases} 1.0 & \psi_N \le 0.90 \\ \cos^2\left(\frac{\pi}{2} \frac{\psi_N - 0.90}{0.05}\right) & 0.90 < \psi_N \le 0.95 \\ 0.0 & \psi_N \ge 0.95 \end{cases}$$
This protects the integration from spurious gradient blowup while preserving the exact perturbation around rational surfaces (e.g., $q=2$ at $\psi_N \approx 0.70$).

---

## 3. Magnetic Island Analytic Width (R4, VR15)

Near a rational magnetic surface with $q(s_{\text{res}}) = m/n = 2/1$:
$$w_{\psi_N} = 4 \sqrt{\frac{\tilde{\psi}_N q_0}{|dq/d\psi_N|}}$$
Where:
- $q_0 = 2.0004$ (rational safety factor at resonance $s_{\text{res}} = 0.7000$)
- $|dq/d\psi_N| = 3.8663$ (magnetic shear at resonance)
- $\tilde{\psi}_N = A = 10^{-3}$ (fractional perturbation amplitude in normalized flux)

Analytic island width:
$$w_{\psi_N} = 4 \sqrt{\frac{10^{-3} \times 2.0004}{3.8663}} = 0.09099 \text{ (normalized flux units)}$$
Radial width at midplane ($Z = Z_{\text{axis}}$):
$$w_{\text{mm}} = \frac{w_{\psi_N}}{|d\psi_N/dR|_{\text{midplane}}} \times 1000 = \frac{0.09099}{2.7116} \times 1000 = 33.55\text{ mm}$$

**Acceptance Criterion:** The numerical separatrix width $w_{\text{num}}$ measured across the O-point must match analytic $w_{\text{analytic}}$ to within $\le 5.0\%$ for $A = 10^{-3}$.

---

## 4. Chaos Criterion and Relative FTLE (E20)

Finite-Time Lyapunov Exponent:
$$\text{FTLE}(T) = \frac{1}{T} \ln \left( \frac{\|\delta(T)\|}{\|\delta_0\|} \right)$$
1. **Single Resonant Mode ($m/n = 2/1$, $A = 10^{-3}$):**
   By the KAM theorem / E20, a 1-DoF Hamiltonian in helical coordinates is integrable. There is NO CHAOS. Trajectories separate linearly (due to shear), so $\text{FTLE} \propto \frac{\ln(C T)}{T} \to 0$.
2. **Double Mode ($2/1 + 3/1$, $A_{2/1} = 3\times 10^{-3}, A_{3/1} = 3\times 10^{-3}$):**
   The islands overlap (Chirikov parameter $S \approx \frac{w_2 + w_3}{2|r_2 - r_3|} \ge 1.0$), generating a stochastic sea with exponential trajectory divergence.
3. **Pre-Registered Criterion:**
   $$\text{FTLE}_{\text{chaotic}} \ge k \cdot \text{FTLE}_{\text{regular}} \quad \text{at } T = 50 \text{ turns}$$
   From the pre-registration pilot run:
   - $\text{FTLE}_{\text{reg}} = 0.0177\text{ rad}^{-1}$
   - $\text{FTLE}_{\text{chaos}} = 0.0462\text{ rad}^{-1}$
   - Pilot ratio: $k_{\text{pilot}} = 2.61$
   - **Threshold:** We pre-register $k = 2.0$ at $T = 50$ turns ($T \approx 314.16\text{ rad}$).

---

## 5. Pre-Registered Criteria and Pass/Fail Thresholds

All thresholds are derived from the pilot run `pilot_summary.json`:

| Metric | Integrator / Mode | Pilot Value | Pre-Registered Threshold | Theoretical Rationale |
|---|---|---|---|---|
| **Convergence Order $p$** | Canonical Midpoint | $2.00 \dots 2.05$ | $p \in [1.85, 2.15]$ | 2nd-order symplectic midpoint |
| **Convergence Order $p$** | Cartesian RK4 | $3.85$ (coarse) | $p \ge 3.5$ ($\frac{2\pi}{30} \to \frac{2\pi}{60}$) | 4th-order classical Runge-Kutta |
| **Global Error $E$** | Cartesian RK4 ($\Delta\phi = \frac{2\pi}{60}$) | $4.68 \times 10^{-4}\text{ m}$ | $E \le 1.0 \times 10^{-3}\text{ m}$ | Bounded discretization error |
| **Symplecticity Error** | Canonical Midpoint | $0.0$ | $\|\det J - 1\| \le 1.0 \times 10^{-10}$ | Liouville / symplectic volume |
| **Hamiltonian Drift** | Canonical Midpoint ($100$ turns) | $0.0\text{ Wb/rad}$ | $\Delta H_{100} \le 1.0 \times 10^{-10}\text{ Wb/rad}$ | Energy conservation in integrable frame |
| **Flux Drift** | Cartesian RK4 ($100$ turns) | $1.38 \times 10^{-4}\text{ Wb/rad}$ | $\Delta\psi_{100} \le 3.0 \times 10^{-4}\text{ Wb/rad}$ | $< 0.10\%$ of total $\Delta\psi$ range ($0.299\text{ Wb}$) |
| **CPU DOP853 Match** | Cartesian RK4 ($10$ turns) | $4.26 \times 10^{-4}\text{ m}$ | Discrepancy $\le 1.0 \times 10^{-3}\text{ m}$ | Consistency with Scipy DOP853 reference |
| **Island Width Error** | Separatrix ($A = 10^{-3}$) | $0.00\%$ | Error $\le 5.0\%$ | Analytic formula valid for small $A$ |
| **Chaos Ratio $k$** | Double vs Single ($T = 50$) | $2.61$ | $\text{FTLE}_{\text{chaos}} \ge 2.0 \cdot \text{FTLE}_{\text{reg}}$ | Chirikov overlap stochasticity (E20) |
| **Residual Stopping** | Newton Solver | $1.77 \times 10^{-15}$ | $\|r\|_2 \le 1.0 \times 10^{-12}$ | Strict residual termination |
| **Alive Masking** | Wall strike / exit | Valid | No NaNs, coordinates frozen | Out-of-bounds safety |

---

## 6. Pre-Registration Freeze Declaration
This document was finalized on **2026-10-09** after the preliminary pilot run. All criteria and thresholds are permanently frozen prior to the execution of `run_t12.py`.
