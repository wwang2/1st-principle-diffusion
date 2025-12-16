# Marginal Score from Data-Space Forces

The marginal score $\nabla_{x_t} \log p_t(x_t)$ can be expressed entirely in terms of the **data-space score** $\nabla_{x_0} \log p_0(x_0)$.

---

## Setup

- Forward kernel: $q(x_t \mid x_0) = \mathcal{N}(x_t; \alpha_t x_0, \sigma_t^2 I)$
- Data distribution: $p_0(x_0)$
- Marginal: $p_t(x_t) = \int q(x_t \mid x_0) p_0(x_0) \, dx_0$

---

## Gradient Relationship

For the Gaussian kernel:

$$
\log q(x_t \mid x_0) = -\frac{\|x_t - \alpha_t x_0\|^2}{2\sigma_t^2} + \text{const}
$$

Taking gradients:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2}
$$

$$
\nabla_{x_0} \log q(x_t \mid x_0) = +\frac{\alpha_t(x_t - \alpha_t x_0)}{\sigma_t^2}
$$

Therefore:

$$
\boxed{\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{1}{\alpha_t} \nabla_{x_0} \log q(x_t \mid x_0)}
$$

---

## Bayes Rule Decomposition

From Bayes' rule $p(x_0 \mid x_t) \propto q(x_t \mid x_0) p_0(x_0)$:

$$
\nabla_{x_0} \log q(x_t \mid x_0) = \nabla_{x_0} \log p(x_0 \mid x_t) - \nabla_{x_0} \log p_0(x_0)
$$

---

## Plug the defintions back

The marginal score is:

$$
\nabla_{x_t} \log p_t(x_t) = \mathbb{E}_{p(x_0 \mid x_t)}\left[ \nabla_{x_t} \log q(x_t \mid x_0) \right]
$$

Substituting the gradient relationship and Bayes decomposition:

$$
= -\frac{1}{\alpha_t} \mathbb{E}_{p(x_0 \mid x_t)}\left[ \nabla_{x_0} \log p(x_0 \mid x_t) - \nabla_{x_0} \log p_0(x_0) \right]
$$

The first term vanishes by the score identity $\mathbb{E}_p[\nabla \log p] = 0$:

$$
\boxed{\nabla_{x_t} \log p_t(x_t) = \frac{1}{\alpha_t} \mathbb{E}_{p(x_0 \mid x_t)}\left[ \nabla_{x_0} \log p_0(x_0) \right]}
$$

---

## Physical Interpretation: Newtonian Forces

In physics, systems at thermal equilibrium follow the **Boltzmann distribution**:

$$
p_0(x_0) = \frac{1}{Z} \exp\left(-\frac{U(x_0)}{k_B T}\right)
$$

where $U(x_0)$ is the potential energy. Setting $k_B T = 1$:

$$
\nabla_{x_0} \log p_0(x_0) = -\nabla_{x_0} U(x_0) = \mathbf{F}(x_0)
$$

This is precisely **Newton's force**: the negative gradient of potential energy.

### Marginal Score as Expected Force

The main result becomes:

$$
\boxed{\nabla_{x_t} \log p_t(x_t) = \frac{1}{\alpha_t} \mathbb{E}_{p(x_0 \mid x_t)}\left[ \mathbf{F}(x_0) \right]}
$$

**The marginal score at noise level $t$ is the posterior-averaged physical force, scaled by $1/\alpha_t$.**

---

## Score Matching with Forces

### Standard Denoising Score Matching

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \varepsilon}\left[ \left\| s_\theta(x_t, t) + \frac{\varepsilon}{\sigma_t} \right\|^2 \right]
$$

### Force-Based Score Matching

$$
\boxed{\mathcal{L}_{\text{Force}} = \mathbb{E}_{t, x_0, x_t}\left[ \left\| s_\theta(x_t, t) - \frac{\mathbf{F}(x_0)}{\alpha_t} \right\|^2 \right]}
$$

Both objectives are **equivalent** at the population level—they have the same minimizer.


### Single-Sample Approximation

Using the denoiser $\hat{x}_0 = \mathbb{E}[x_0 \mid x_t]$:

$$
\nabla_{x_t} \log p_t(x_t) \approx \frac{\mathbf{F}(\hat{x}_0)}{\alpha_t}
$$

---

## Summary

| Quantity | Expression |
|----------|------------|
| Data-space score | $\nabla_{x_0} \log p_0(x_0)$ |
| Physical force | $\mathbf{F}(x_0) = -\nabla_{x_0} U(x_0)$ |
| Marginal score | $\frac{1}{\alpha_t} \mathbb{E}_{p(x_0 \mid x_t)}[\mathbf{F}(x_0)]$ |

> **Key insight**: Diffusion score matching can be performed using only data-space forces $\mathbf{F}(x_0) = -\nabla U(x_0)$, without computing anything in the noisy space.
