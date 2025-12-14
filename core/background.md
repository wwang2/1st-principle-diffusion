## Diffusion (forward + reverse) — minimal notes

### Defi.
- Data: $x_0$ (clean sample), typically $x_0\in\mathbb{R}^{N\times d}$ (often $d=3$)
- Time: $t\in[0,1]$ (0=data, 1=prior/noise)
- Schedules: $\alpha_t$ (signal scale), $\sigma_t$ (noise scale). VP: $\sigma_t^2=1-\alpha_t^2$
- Covariance: $\Sigma = RR^\top$ (isotropic special case: $\Sigma=I$)
- Noise: $z\sim\mathcal N(0,I)$ so $Rz\sim\mathcal N(0,RR^\top)$
- Marginal: $p_t(x)$ (noise-convolved marginal)
- Score: $s_t(x)=\nabla_x\log p_t(x)$
- Denoiser: $\hat x_\theta(x_t,t)\approx\mathbb E[x_0\mid x_t]$

### Forward diffusion (kernel + marginal)
Forward kernel:

$$
q(x_t\mid x_0)=\mathcal N\big(x_t;\,\alpha_t x_0,\,\sigma_t^2\Sigma\big)
$$

Sampling form:

$$
x_t = \alpha_t x_0 + \sigma_t R z,\qquad z\sim\mathcal N(0,I)
$$

Noise-convolved marginal:

$$
p_t(x_t)=\int q(x_t\mid x_0)\,p_0(x_0)\,dx_0
$$

### Forward SDE (OU form)
A linear SDE that yields the same Gaussian kernels:

$$
dx = h_t\,x\,dt + g_t\,R\,dw
$$

with

$$
h_t=\frac{d}{dt}\log\alpha_t,\qquad \mathrm{SNR}_t=\frac{\alpha_t^2}{\sigma_t^2},\qquad g_t=\sqrt{-\sigma_t^2\,\frac{d}{dt}\log\mathrm{SNR}_t}
$$

(VP special case: $dx=-\tfrac12\beta_t x\,dt+\sqrt{\beta_t}\,R\,dw$.)

### Reverse-time SDE (sampling)
Reverse SDE (maps $p_1\to p_0$):

$$
dx = \big(h_t x - g_t^2\,\Sigma\,s_t(x)\big)\,dt + g_t\,R\,d\bar w
$$

Denoiser-to-score identity (common parameterization):

$$
s_t(x)=(\sigma_t^2\Sigma)^{-1}\big(\alpha_t\,\hat x_\theta(x,t)-x\big)
$$

### Probability flow ODE (deterministic)
ODE with the same marginals $p_t$:

$$
\frac{dx}{dt}= h_t x - \tfrac12 g_t^2\,\Sigma\,s_t(x)
$$
