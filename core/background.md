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
d\mathbf{x} = \big(h_t \mathbf{x} - g_t^2\,\Sigma\,s_t(\mathbf{x})\big)\,dt + g_t\,\mathbf{R}\,d\bar{\mathbf{w}}
```

Denoiser-to-score identity (common parameterization):

```math
s_t(\mathbf{x})=(\sigma_t^2\Sigma)^{-1}\big(\alpha_t\,\hat{\mathbf{x}}_\theta(\mathbf{x},t)-\mathbf{x}\big)
```

### Probability flow ODE (deterministic)
ODE with the same marginals $p_t$:

```math
\frac{d\mathbf{x}}{dt}= h_t \mathbf{x} - \tfrac12 g_t^2\,\Sigma\,s_t(\mathbf{x})
```
