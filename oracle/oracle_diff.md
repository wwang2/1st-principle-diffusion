# Oracle: Gaussian Mixture Prior under Linear-Gaussian Forward Kernel

---

## Bayes' Rule

The central object of interest is the **posterior** $p(\mathbf x_0 \mid \mathbf x_t)$, which tells us what we can infer about the clean data $\mathbf x_0$ given a noisy observation $\mathbf x_t$. By Bayes' rule:

```math
\boxed{
p(\mathbf x_0 \mid \mathbf x_t) 
= \frac{q(\mathbf x_t \mid \mathbf x_0)\, p_0(\mathbf x_0)}{\displaystyle\int q(\mathbf x_t \mid \mathbf x_0)\, p_0(\mathbf x_0)\, d\mathbf x_0}
= \frac{q(\mathbf x_t \mid \mathbf x_0)\, p_0(\mathbf x_0)}{p_t(\mathbf x_t)}
}
```

where:
- $q(\mathbf x_t \mid \mathbf x_0)$ is the **forward diffusion kernel** (how noise corrupts clean data)
- $p_0(\mathbf x_0)$ is the **data prior** (distribution of clean data)
- $p_t(\mathbf x_t) = \int q(\mathbf x_t \mid \mathbf x_0)\, p_0(\mathbf x_0)\, d\mathbf x_0$ is the **marginal** at time $t$

The posterior is generally intractable for complex priors, but becomes **analytically computable** when both the forward kernel and prior have special structure—specifically, when both are Gaussian (or mixtures thereof).

---

## Setup

Assume the (possibly anisotropic / correlated) linear-Gaussian diffusion kernel:

```math
q(\mathbf x_t\mid \mathbf x_0)=\mathcal N\!\big(\mathbf x_t;\,\alpha_t\,\mathbf x_0,\;\sigma_t^2\,\Sigma\big),\qquad \Sigma\succ 0,
```

where $\mathbf x_0,\mathbf x_t\in\mathbb R^D$ (e.g. flatten $N\times d\to D$). Equivalently,

```math
\mathbf x_t=\alpha_t\,\mathbf x_0+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma_t^2\Sigma),\ \varepsilon\perp \mathbf x_0.
```

Assume the data distribution is a $K$-component Gaussian mixture:

```math
p_0(\mathbf x_0)=\sum_{k=1}^K \pi_k\,\mathcal N(\mathbf x_0;\,\mu_k,\Lambda_k),
\qquad \pi_k\ge 0,\ \sum_{k=1}^K\pi_k=1,\qquad \Lambda_k\succ 0.
```

---

## Marginal Distribution

**Key observation (forward marginal stays a GMM):**

Conditioned on component $k$, $\mathbf x_0\mid k\sim\mathcal N(\mu_k,\Lambda_k)$. An affine transform of a Gaussian plus independent Gaussian noise is Gaussian, so:

```math
\mathbf x_t\mid k\sim\mathcal N\!\Big(\alpha_t\mu_k,\ \alpha_t^2\Lambda_k+\sigma_t^2\Sigma\Big).
```

Therefore the marginal is analytic and remains a GMM:

```math
\boxed{
 p_t(\mathbf x_t)
 :=\int q(\mathbf x_t\mid \mathbf x_0)\,p_0(\mathbf x_0)\,d\mathbf x_0
 =\sum_{k=1}^K \pi_k\,\mathcal N\!\Big(\mathbf x_t;\ \alpha_t\mu_k,\ C_{t,k}\Big)
}
\qquad C_{t,k}:=\alpha_t^2\Lambda_k+\sigma_t^2\Sigma.
```

So: mixture weights unchanged, means scaled by $\alpha_t$, and covariances become “scaled + diffusion noise”.

---

## Posterior Responsibility

Target: `p(k | x_t)`

By Bayes’ rule,

```math
\boxed{
 r_{t,k}(\mathbf x_t)\ :=\ p(k\mid \mathbf x_t)
 =\frac{\pi_k\,\mathcal N\!\big(\mathbf x_t;\alpha_t\mu_k,\ C_{t,k}\big)}{\sum_{j=1}^K \pi_j\,\mathcal N\!\big(\mathbf x_t;\alpha_t\mu_j,\ C_{t,j}\big)}
}.
```

---

## Component-Conditional Posterior

Target: `p(x_0 | x_t, k)`

For a fixed $k$, this is a standard linear-Gaussian model:

```math
\mathbf x_0\mid k\sim\mathcal N(\mu_k,\Lambda_k),\qquad
\mathbf x_t\mid \mathbf x_0\sim\mathcal N(\alpha_t\mathbf x_0,\sigma_t^2\Sigma).
```

Hence $\mathbf x_0\mid \mathbf x_t,k$ is Gaussian:

```math
\boxed{
 p(\mathbf x_0\mid \mathbf x_t,k)=\mathcal N\!\big(\mathbf x_0;\ m_{t,k}(\mathbf x_t),\ V_{t,k}\big)
}
```

with

```math
\boxed{
 V_{t,k}:=\Big(\Lambda_k^{-1}+\alpha_t^2(\sigma_t^2\Sigma)^{-1}\Big)^{-1},
 \qquad
 m_{t,k}(\mathbf x_t):=V_{t,k}\Big(\Lambda_k^{-1}\mu_k+\alpha_t(\sigma_t^2\Sigma)^{-1}\mathbf x_t\Big).
}
```

---

## Exact MMSE Denoiser (Oracle)

Target: `E[x_0 | x_t]`

The full posterior is a mixture of the component posteriors:

```math
p(\mathbf x_0\mid \mathbf x_t)=\sum_{k=1}^K r_{t,k}(\mathbf x_t)\,\mathcal N\!\big(\mathbf x_0;\ m_{t,k}(\mathbf x_t),\ V_{t,k}\big).
```

Therefore the exact MMSE denoiser is:

```math
\boxed{
 \hat{\mathbf x}_0(\mathbf x_t)\ :=\ \mathbb E[\mathbf x_0\mid \mathbf x_t]
 =\sum_{k=1}^K r_{t,k}(\mathbf x_t)\,m_{t,k}(\mathbf x_t).
}
```

Since $\varepsilon=\mathbf x_t-\alpha_t\mathbf x_0$, the conditional mean noise is:

```math
\boxed{
 \mathbb E[\varepsilon\mid \mathbf x_t]
 =\mathbf x_t-\alpha_t\,\mathbb E[\mathbf x_0\mid \mathbf x_t]
 =\mathbf x_t-\alpha_t\sum_{k=1}^K r_{t,k}(\mathbf x_t)\,m_{t,k}(\mathbf x_t).
}
```

---

## Oracle Score

Target: `∇_{x_t} log p_t(x_t)`

### Key Identity: Marginal Score as Posterior Expectation

The marginal score can always be written as a **posterior expectation** of the conditional score:

```math
\boxed{
\nabla_{\mathbf x_t} \log p_t(\mathbf x_t) = \mathbb{E}_{p(\mathbf x_0|\mathbf x_t)}\left[\nabla_{\mathbf x_t} \log q(\mathbf x_t|\mathbf x_0)\right]
}
```

This follows directly from:
```math
\nabla_{\mathbf x_t} \log p_t(\mathbf x_t) 
= \frac{\nabla_{\mathbf x_t} p_t(\mathbf x_t)}{p_t(\mathbf x_t)}
= \frac{\int \nabla_{\mathbf x_t} q(\mathbf x_t|\mathbf x_0)\, p_0(\mathbf x_0)\, d\mathbf x_0}{p_t(\mathbf x_t)}
= \int \nabla_{\mathbf x_t} \log q(\mathbf x_t|\mathbf x_0)\, p(\mathbf x_0|\mathbf x_t)\, d\mathbf x_0
```

This identity is fundamental—it connects the **marginal score** (what diffusion models learn) to the **conditional score** (which has a simple closed form for Gaussian kernels). See [force=score.md](../force/force=score.md) for the derivation showing this equals the expected data-space force.

### Explicit Formula for GMM

For the Gaussian kernel, the conditional score is:
```math
\nabla_{\mathbf x_t}\log q(\mathbf x_t|\mathbf x_0)
= -(\sigma_t^2\Sigma)^{-1}(\mathbf x_t - \alpha_t\mathbf x_0)
```

Since $p_t$ is a GMM with component covariances $C_{t,k}=\alpha_t^2\Lambda_k+\sigma_t^2\Sigma$, we can also compute directly:

```math
\nabla_{\mathbf x_t}\log \mathcal N(\mathbf x_t;\alpha_t\mu_k,C_{t,k})
=-C_{t,k}^{-1}(\mathbf x_t-\alpha_t\mu_k).
```

Weighting by responsibilities gives:

```math
\boxed{
 \nabla_{\mathbf x_t}\log p_t(\mathbf x_t)
 =\sum_{k=1}^K r_{t,k}(\mathbf x_t)\,\big[-C_{t,k}^{-1}(\mathbf x_t-\alpha_t\mu_k)\big].
}
```
