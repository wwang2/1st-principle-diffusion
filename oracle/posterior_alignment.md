## Posterior Alignment: Optimal Denoising = Posterior Mean

**Key insight:** Both GMM and symmetry-augmented diffusion share the same structure—the optimal denoiser is a **posterior expectation**.

---

### Unified Framework

| | **GMM Oracle** ([oracle_diff.md](./oracle_diff.md)) | **SO(3) Alignment** |
|---|---|---|
| **Prior** | $p_0(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mu_k, \Lambda_k)$ | $\text{Aug}[p_0] = \int_{SO(3)} p_0(\mathbf{R}^{-1} \circ \mathbf{x})\, d\mathbf{R}$ |
| **Posterior** | Mixture of Gaussians | **Matrix Fisher** over $SO(3)$ |
| **Optimal denoiser** | $\displaystyle\sum_k r_k(\mathbf{x}_t)\, m_k(\mathbf{x}_t)$ | $\mathbb{E}_{\mathbf{R} \sim p(\mathbf{R}\|\mathbf{y},\mathbf{x}_0)}[\mathbf{R}] \circ \mathbf{x}_0$ |
| **Computation** | Closed-form (weighted sum) | Mode + corrections (Kabsch) |
| **Perfect denoiser?** | ✓ Yes (Bayes-optimal) | ✗ No (irreducible ambiguity) |

---

### The Matrix Fisher Posterior

Given noisy observation $\mathbf{y} = \mathbf{R}_{\text{aug}} \circ (\mathbf{x}_0 + \sigma\boldsymbol{\eta})$, the posterior over rotations is:

```math
\boxed{p(\mathbf{R} \mid \mathbf{y}, \mathbf{x}_0, \sigma) = \text{MF}\left(\mathbf{R};\, \frac{\mathbf{y}^\top \mathbf{x}_0}{\sigma^2}\right) \propto \exp\left(\text{Tr}\left[\frac{\mathbf{y}^\top \mathbf{x}_0}{\sigma^2} \mathbf{R}\right]\right)}
```

**Key properties:**
- Unimodal (for generic point clouds)
- Concentrates as $\sigma \to 0$
- Mode via SVD: $\mathbf{y}^\top \mathbf{x}_0 = USV^\top \Rightarrow \mathbf{R}^* = UV^\top$ (Kabsch alignment)

---

### Alignment as Posterior Mode Approximation

```
              Posterior Mean (exact)                 Mode (alignment)
                     ↓                                      ↓
    D*(y; x₀) = E_R[R] ∘ x₀     ───σ→0───▶      D₀*(y; x₀) = R* ∘ x₀
                     │                                      │
                     │          Laplace expansion           │
                     └──────────────────────────────────────┘
                           D* ≈ (R* + σ²B₁ + σ⁴B₂) ∘ x₀
```

**First-order correction** (from SVD singular values $s_1 \geq s_2 \geq |s_3|$):

```math
C_1(S) = -\frac{1}{2}\text{diag}\left[\frac{1}{s_1+s_2} + \frac{1}{s_1+s_3},\; \frac{1}{s_2+s_1} + \frac{1}{s_2+s_3},\; \frac{1}{s_3+s_1} + \frac{1}{s_3+s_2}\right]
```

→ No extra cost: $U, V$ already computed in Kabsch!

---

### Practical Takeaway

> **Alignment is the low-noise limit of optimal Bayesian denoising.**

| Noise level | Recommendation |
|-------------|----------------|
| Small $\sigma$ | Kabsch alignment ≈ optimal |
| Large $\sigma$ | Higher-order corrections can diverge |
| Any $\sigma$ | Equivariant architectures bypass alignment entirely |

**Ref:** [Daigavane et al. 2024](https://arxiv.org/abs/2510.03335) — *Matching the Optimal Denoiser in Point Cloud Diffusion*
