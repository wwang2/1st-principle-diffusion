## Diffusion (forward + reverse) — minimal notes

### Def.
- Data: $\mathbf{x}_0$ (clean sample), typically $\mathbf{x}_0\in\mathbb{R}^{N\times d}$ (often $d=3$)
- Time: $t\in[0,1]$ (0=data, 1=prior/noise)
- Schedules: $\alpha_t$ (signal scale), $\sigma_t$ (noise scale). VP: $\sigma_t^2=1-\alpha_t^2$
- Covariance: $\Sigma = \mathbf{R}\mathbf{R}^\top$ (isotropic special case: $\Sigma=\mathbf{I}$)
- Noise: $z\sim\mathcal N(0,\mathbf{I})$ so $\mathbf{R}z\sim\mathcal N(0,\mathbf{R}\mathbf{R}^\top)$
- Marginal: $p_t(\mathbf{x})$ (noise-convolved marginal)
- Score: $s_t(\mathbf{x})=\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$
- Denoiser: $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)\approx\mathbb E[\mathbf{x}_0\mid \mathbf{x}_t]$

### Forward diffusion (kernel + marginal)
Forward kernel:

```math
q(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal N\big(\mathbf{x}_t;\,\alpha_t \mathbf{x}_0,\,\sigma_t^2\Sigma\big)
```

Sampling form:

```math
\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \mathbf{R} z,\qquad z\sim\mathcal N(0,\mathbf{I})
```

Noise-convolved marginal:

```math
p_t(\mathbf{x}_t)=\int q(\mathbf{x}_t\mid \mathbf{x}_0)\,p_0(\mathbf{x}_0)\,d\mathbf{x}_0
```

### Forward SDE (Ornstein Uhlenbeck)
A linear SDE that yields the same Gaussian kernels:

```math
d\mathbf{x} = h_t\,\mathbf{x}\,dt + g_t\,\mathbf{R}\,d\mathbf{w}
```

with

```math
h_t=\frac{d}{dt}\log\alpha_t,\qquad \mathrm{SNR}_t=\frac{\alpha_t^2}{\sigma_t^2},\qquad g_t=\sqrt{-\sigma_t^2\,\frac{d}{dt}\log\mathrm{SNR}_t}
```

(VP special case: $d\mathbf{x}=-\tfrac12\beta_t \mathbf{x}\,dt+\sqrt{\beta_t}\,\mathbf{R}\,d\mathbf{w}$.)

### Reverse-time SDE (sampling)
Reverse SDE (maps $p_1\to p_0$):

```math
d\mathbf{x} = \big(h_t \mathbf{x} - g_t^2\,\mathbf{R} \mathbf{R}^{\intercal}\, \nabla_{\mathbf{x}}\log p(\mathbf{x})\big)\,dt + g_t\,\mathbf{R}\,d\bar{\mathbf{w}}
```

Denoiser-to-score identity (common parameterization):

```math
s_t(\mathbf{x})=(\sigma_t^2\Sigma)^{-1}\big(\alpha_t\,\hat{\mathbf{x}}_\theta(\mathbf{x},t)-\mathbf{x}\big)
```

### Fokker-Planck equation (density evolution)

The evolution of a probability density under an SDE is governed by the **Fokker-Planck equation** (FPE, also called Kolmogorov forward equation). For a general SDE $d\mathbf{x} = \mathbf{f}(\mathbf{x},t)\,dt + g_t\mathbf{R}\,d\mathbf{w}$:

```math
\frac{\partial p_t}{\partial t} = -\nabla \cdot (\mathbf{f} \, p_t) + \frac{g_t^2}{2} \nabla \cdot (\Sigma \nabla p_t)
```

For our linear drift $\mathbf{f} = h_t\mathbf{x}$:

```math
\frac{\partial p_t}{\partial t} = -\nabla \cdot (h_t \mathbf{x} \, p_t) + \frac{g_t^2}{2} \nabla \cdot (\Sigma \nabla p_t)
```

Expanding both terms:
- **Drift term:** $\nabla \cdot (h_t \mathbf{x} \, p_t) = h_t(d \cdot p_t + \mathbf{x} \cdot \nabla p_t)$ where $d$ is dimension
- **Diffusion term:** $\nabla \cdot (\Sigma \nabla p_t) = \text{tr}(\Sigma \nabla^2 p_t) = \text{tr}(\Sigma) \Delta p_t$ for isotropic $\Sigma = \sigma^2 \mathbf{I}$

---

### Sister equation: Fokker-Planck vs Schrödinger

The FPE has a deep connection to quantum mechanics via the **Cole-Hopf transformation**.

| **Aspect** | **Fokker-Planck** | **Schrödinger** |
|------------|-------------------|-----------------|
| Unknown | $p(\mathbf{x},t)$ (probability density) | $\psi(\mathbf{x},t)$ (wave function) |
| Time | Real, forward | Imaginary ($t \to it$) |
| Basic form | $\partial_t p = -\nabla \cdot (\mathbf{f} p) + D \Delta p$ | $i\hbar \partial_t \psi = -\frac{\hbar^2}{2m}\Delta \psi + V\psi$ |
| Probability | $p \geq 0$ directly | $|\psi|^2 \geq 0$ |

**Cole-Hopf transformation:** Setting $p = \psi^2$ (or $p = e^{-2\phi}$) transforms the FPE into a linear Schrödinger-like equation in imaginary time:

```math
\partial_t \psi = D \Delta \psi - U(\mathbf{x})\psi
```

where the "potential" $U$ relates to the drift. This connection underlies:
- **Schrödinger bridges:** Finding the most likely stochastic evolution between two endpoint distributions $p_0, p_1$
- **Variational formulations:** Viewing diffusion as optimization over path measures

---

### From PDE to ODE: the log-derivative trick

**Goal:** Given a PDE governing density evolution, derive an ODE for individual sample trajectories that preserves the same marginals.

#### Step 1: Continuity equation for ODEs

For a deterministic flow $\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t)$, conservation of probability mass gives the **continuity equation**:

```math
\frac{\partial p_t}{\partial t} + \nabla \cdot (\mathbf{v} \, p_t) = 0
```

This says: the rate of change of density at a point equals the net flux of probability out of that point.

#### Step 2: Log-derivative identity

Taking $\frac{\partial}{\partial t}\log p_t = \frac{1}{p_t}\frac{\partial p_t}{\partial t}$ and using the continuity equation:

```math
\frac{\partial \log p_t}{\partial t} = -\frac{1}{p_t}\nabla \cdot (\mathbf{v} \, p_t)
```

Expanding $\nabla \cdot (\mathbf{v} \, p_t) = p_t(\nabla \cdot \mathbf{v}) + \mathbf{v} \cdot \nabla p_t$:

```math
\frac{\partial \log p_t}{\partial t} = -\nabla \cdot \mathbf{v} - \mathbf{v} \cdot \frac{\nabla p_t}{p_t}
```

Using $\nabla \log p_t = \frac{\nabla p_t}{p_t}$:

```math
\boxed{\frac{\partial \log p_t(\mathbf{x})}{\partial t} = -\nabla \cdot \mathbf{v} - \mathbf{v} \cdot \nabla \log p_t}
```

#### Step 3: Instantaneous change of variables (FFJORD / Neural ODEs)

For a sample $\mathbf{x}_t$ following the ODE, the total derivative of log-density along its path is:

```math
\frac{d}{dt}\log p_t(\mathbf{x}_t) = \underbrace{\frac{\partial \log p_t}{\partial t}}_{\text{explicit}} + \underbrace{\nabla \log p_t \cdot \frac{d\mathbf{x}_t}{dt}}_{\text{implicit (chain rule)}}
```

Substituting $\frac{d\mathbf{x}_t}{dt} = \mathbf{v}$ and the log-derivative identity:

```math
\frac{d}{dt}\log p_t(\mathbf{x}_t) = \big(-\nabla \cdot \mathbf{v} - \mathbf{v} \cdot \nabla \log p_t\big) + \nabla \log p_t \cdot \mathbf{v} = -\nabla \cdot \mathbf{v}
```

The $\mathbf{v} \cdot \nabla \log p_t$ terms cancel! Integrating:

```math
\boxed{\log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \nabla \cdot \mathbf{v}(\mathbf{x}_t, t) \, dt}
```

This **instantaneous change of variables** formula is the foundation of FFJORD and continuous normalizing flows.

### Probability flow ODE (deterministic)

**Key fact:** For any SDE with drift $\mathbf{f}$ and diffusion $g$, there exists a deterministic ODE with identical marginals $p_t$.

**Derivation:** The Fokker-Planck equation can be rewritten using $\nabla \cdot (\Sigma \nabla p) = \nabla \cdot (p \, \Sigma \nabla \log p)$:

```math
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left( \underbrace{\left(\mathbf{f} - \frac{g^2}{2} \Sigma \nabla \log p_t \right)}_{\mathbf{v}_{\text{ODE}}} p_t \right)
```

This is the continuity equation for the **probability flow ODE**:

```math
\frac{d\mathbf{x}}{dt}= h_t \mathbf{x} - \tfrac12 g_t^2\,\Sigma\, \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
```

The score $s_t = \nabla \log p_t$ appears as the correction term accounting for the "missing" stochasticity. This enables:
1. **Deterministic sampling** (no noise needed)
2. **Exact likelihood evaluation** (via instantaneous change of variables)
3. **Latent manipulation** (invertible data ↔ noise mapping)

---

### Summary: Forward SDE → Reverse SDE → ODE

A unified view of the three processes that all share the **same marginal densities** $p_t(\mathbf{x})$.

---

#### 1. Forward SDE (data → noise)

```math
d\mathbf{x} = \underbrace{h_t \mathbf{x}}_{\text{drift}} dt + \underbrace{g_t \mathbf{R}}_{\text{diffusion}} d\mathbf{w}
```

- **Direction:** $t: 0 \to 1$ (clean data → pure noise)
- **Drift:** $h_t = \frac{d}{dt}\log\alpha_t < 0$ shrinks signal toward origin
- **Diffusion:** $g_t \mathbf{R}\, d\mathbf{w}$ injects noise with covariance $g_t^2 \Sigma$
- **Marginals:** $p_t$ evolves from data $p_0$ to prior $p_1 \approx \mathcal{N}(0, \Sigma)$

---

#### 2. Reverse SDE (noise → data)

**Anderson's time-reversal theorem (1982):** Any SDE can be reversed in time. For forward SDE with drift $\mathbf{f}$ and diffusion $g$:

```math
d\mathbf{x} = \left[\mathbf{f} - g^2 \Sigma \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g\, \mathbf{R}\, d\bar{\mathbf{w}}
```

For our linear case $\mathbf{f} = h_t \mathbf{x}$:

```math
d\mathbf{x} = \big(h_t \mathbf{x} - g_t^2\,\Sigma\, \nabla_{\mathbf{x}}\log p_t(\mathbf{x})\big)\,dt + g_t\,\mathbf{R}\,d\bar{\mathbf{w}}
```

- **Direction:** $t: 1 \to 0$ (noise → data), $d\bar{\mathbf{w}}$ is reverse Brownian motion
- **Key insight:** The score $\nabla \log p_t$ "steers" particles back toward data
- **Same marginals:** Running this backward recovers exactly $p_t$ at each time

**Where does the $-g^2 \Sigma \nabla \log p_t$ term come from?**

The Fokker-Planck equation for forward time is:

```math
\frac{\partial p}{\partial t} = -\nabla \cdot (\mathbf{f} p) + \frac{g^2}{2} \nabla \cdot (\Sigma \nabla p)
```

For reverse time $\tau = 1-t$, we need $\frac{\partial p}{\partial \tau} = -\frac{\partial p}{\partial t}$. The **diffusion term stays the same** (symmetric in time), but we must modify the drift to flip the sign of the FPE. Setting $\mathbf{f}_{\text{rev}} = \mathbf{f} - g^2 \Sigma \nabla \log p$ achieves this.

---

#### 3. Probability Flow ODE (deterministic equivalent)

**Key observation:** The diffusion term in the FPE can be rewritten using the score:

```math
\frac{g^2}{2} \nabla \cdot (\Sigma \nabla p) = \frac{g^2}{2} \nabla \cdot (p \, \Sigma \nabla \log p)
```

This lets us absorb half the diffusion into an effective drift:

```math
\frac{\partial p}{\partial t} = -\nabla \cdot \left(\underbrace{\mathbf{f} - \frac{g^2}{2} \Sigma \nabla \log p}_{\mathbf{v}_{\text{ODE}}}\right) p
```

This is a **continuity equation** (no diffusion term!), corresponding to the ODE:

```math
\frac{d\mathbf{x}}{dt} = h_t \mathbf{x} - \frac{g_t^2}{2}\,\Sigma\, \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
```

- **Direction:** Works both ways ($0 \to 1$ or $1 \to 0$)
- **Same marginals:** Identical $p_t$ as the SDEs
- **No stochasticity:** Deterministic trajectories, invertible mapping

---

#### Side-by-side comparison

| | **Forward SDE** | **Reverse SDE** | **Probability Flow ODE** |
|---|---|---|---|
| **Equation** | $d\mathbf{x} = h_t \mathbf{x}\, dt + g_t \mathbf{R}\, d\mathbf{w}$ | $d\mathbf{x} = (h_t \mathbf{x} - g_t^2 \Sigma s_t)\, dt + g_t \mathbf{R}\, d\bar{\mathbf{w}}$ | $\frac{d\mathbf{x}}{dt} = h_t \mathbf{x} - \frac{g_t^2}{2} \Sigma s_t$ |
| **Time** | $0 \to 1$ | $1 \to 0$ | Either |
| **Stochastic?** | Yes | Yes | No |
| **Score needed?** | No | Yes | Yes |
| **Marginals** | $p_t$ | $p_t$ | $p_t$ |
| **Use case** | Training (add noise) | Sampling | Sampling + likelihood |

**Intuition:** The reverse SDE and ODE both use the score to "undo" the forward diffusion. The SDE adds noise then corrects with $g^2 \Sigma s_t$; the ODE uses half the correction ($\frac{g^2}{2} \Sigma s_t$) with no noise, achieving the same marginal evolution deterministically.
