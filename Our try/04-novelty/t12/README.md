# Novelty Experiment T12: GPU-Accelerated Field Line Tracer on JAX

## 1. Overview
This experiment implements and rigorously validates a GPU/XLA-accelerated magnetic field line tracer in JAX (`tokamld.tracer`).

Key features:
1. **Canonical Symplectic Integrator:** Implicit Midpoint in $(\psi_t, \theta^*, \phi)$ magnetic coordinates with strict Newton solver ($\|r\|_2 \le 10^{-12}$) guaranteeing exact symplecticity $|\det J - 1| \le 10^{-10}$ (E19).
2. **Cartesian Flow Integrator:** Classical RK4 in $(R, Z, \phi)$ coordinates with alive-masking to prevent NaNs on wall strike.
3. **Analytic Resonant Helical Perturbations:** Smooth de Moivre evaluation on $(\cos\theta^*, \sin\theta^*)$ 2D splines, eliminating the $2\pi$ branch cut, with a $C^1$ smooth envelope tapering to 0 for $\psi_N \ge 0.95$ to avoid X-point divergence.
4. **Analytic Island Width Validation:** Quantitative agreement with $w = 4\sqrt{\tilde{\psi}_N q_0 / |dq/d\psi_N|}$ (R4, VR15) for rational surface $q=2$ with $A=10^{-3}$.
5. **Chaos Criterion (E20):** Relative Finite-Time Lyapunov Exponent $\text{FTLE}_{\text{chaotic}} \ge 2.0 \cdot \text{FTLE}_{\text{regular}}$ distinguishing integrable single-mode islands from overlapping stochastic seas ($2/1 + 3/1$, Chirikov $S \ge 1$).
6. **High-Throughput Vectorization:** Scalable batched tracing via `jax.vmap` and `jax.lax.scan`.

---

## 2. How to Reproduce

Run the full validation suite:
```bash
PYTHONPATH=tokamak-3d-viz/src:src "fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t12/run_t12.py"
```

To run the pre-registration pilot scan:
```bash
PYTHONPATH=tokamak-3d-viz/src:src "fusion equilibrium challenge/starter/.venv/bin/python" "Our try/04-novelty/t12/pilot_t12.py"
```

---

## 3. Registered Criteria & Results

All acceptance thresholds were pre-registered in [THEORY.md](file:///Users/ancerian/Documents/Projects/Hackathon_FMF/Our%20try/04-novelty/t12/THEORY.md) prior to the final run.

| Test | Quantity | Measured Value | Pre-Registered Threshold | Status |
|---|---|---|---|---|
| **Test 1** | Canonical Midpoint Order | $p = 2.021$ | $p \in [1.85, 2.15]$ | **PASS** |
| **Test 1** | Cartesian RK4 Coarse Order | $p = 3.850$ | $p \ge 3.5$ | **PASS** |
| **Test 1** | Cartesian Error ($\Delta\phi = 2\pi/60$) | $4.678 \times 10^{-4}\text{ m}$ | $\le 1.0 \times 10^{-3}\text{ m}$ | **PASS** |
| **Test 2** | Symplecticity $|\det J - 1|$ | $0.000\text{ (exact)}$ | $\le 1.0 \times 10^{-10}$ | **PASS** |
| **Test 2** | Canonical $H$ Drift ($100$ turns) | $0.000\text{ Wb/rad}$ | $\le 1.0 \times 10^{-10}\text{ Wb/rad}$ | **PASS** |
| **Test 2** | Cartesian $\psi$ Drift ($100$ turns) | $1.378 \times 10^{-4}\text{ Wb/rad}$ | $\le 3.0 \times 10^{-4}\text{ Wb/rad}$ | **PASS** |
| **Test 3** | CPU Scipy DOP853 Mismatch | $4.260 \times 10^{-4}\text{ m}$ | $\le 1.0 \times 10^{-3}\text{ m}$ | **PASS** |
| **Test 4** | Island Width ($A = 10^{-3}$) | $w_{\text{num}} = 33.55\text{ mm}$ ($w_{\text{analytic}} = 33.55\text{ mm}$) | Relative error $\le 5.0\%$ | **PASS** |
| **Test 5** | Chaos Ratio $k = \frac{\text{FTLE}_{\text{chaos}}}{\text{FTLE}_{\text{reg}}}$ | $k = 2.61$ | $k \ge 2.0$ ($T = 50$ turns) | **PASS** |

**Overall Verdict: PASS**

---

## 4. Throughput Benchmark

*Hardware Note: Measured on Apple Silicon CPU via XLA (GPU not tested in current sandbox).*

| Number of Field Lines $N$ | Turns per Line | Wall Clock Time [s] | Throughput [lines/sec] | ODE Steps / sec |
|---|---|---|---|---|
| $10$ | $20$ | $0.485$ | $20.6$ | $2.47 \times 10^4$ |
| $100$ | $20$ | $0.520$ | $192.4$ | $2.31 \times 10^5$ |
| $1,000$ | $20$ | $1.219$ | $820.4$ | $9.85 \times 10^5$ |
| $10,000$ | $20$ | $10.324$ | **$968.7$** | **$1.16 \times 10^6$** |

Scaling shows sub-linear overhead and massive throughput gains via `jax.vmap` batching, executing over 1.16 million ODE steps per second.

---

## 5. Artifacts
- **Pre-registration Theory:** [`THEORY.md`](file:///Users/ancerian/Documents/Projects/Hackathon_FMF/Our%20try/04-novelty/t12/THEORY.md)
- **JSON Results:** [`results.json`](file:///Users/ancerian/Documents/Projects/Hackathon_FMF/Our%20try/04-novelty/t12/results.json)
- **Diagnostic Plots:** [`t12_tracer_diagnostics.png`](file:///Users/ancerian/Documents/Projects/Hackathon_FMF/Our%20try/04-novelty/t12/t12_tracer_diagnostics.png)
- **Web Export:** [`tokamak-3d-viz/web/poincare_viewer.json`](file:///Users/ancerian/Documents/Projects/Hackathon_FMF/tokamak-3d-viz/web/poincare_viewer.json)
