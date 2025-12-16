# Constrained Sampling via Coordinate Transformation

## 1. Problem Setup

Consider sampling from a target distribution $p(x)$ where $x \in \mathcal{X} \subseteq \mathbb{R}^n$ is constrained to lie on some manifold or constrained region. Rather than sampling directly in the constrained space, we work in an **unconstrained coordinate system** $\tilde{x} \in \mathbb{R}^m$ and define a differentiable map:

```math
f: \mathbb{R}^m \to \mathcal{X}, \quad x = f(\tilde{x})
```

The map $f$ serves as a **projection** (or conditioning) operator that transforms unconstrained coordinates to the constrained space.

---

## 2. Langevin Dynamics in Unconstrained Coordinates

### 2.1 Standard Langevin SDE

In unconstrained coordinates, the overdamped Langevin dynamics for sampling from a target $\pi(\tilde{x})$ is:

```math
d\tilde{x} = \frac{1}{2} \nabla_{\tilde{x}} \log \pi(\tilde{x}) \, dt + dw
```

where $w$ is standard Brownian motion. At equilibrium, this samples from $\pi(\tilde{x})$.

### 2.2 Pull-back Score Through the Projection

To sample from $p(x)$ in the constrained space, we **pull back** the score through the projection $f$:

```math
\nabla_{\tilde{x}} \log p(f(\tilde{x}))
```

By the chain rule:

```math
\nabla_{\tilde{x}} \log p(f(\tilde{x})) = J^\top \nabla_x \log p(x) \Big|_{x = f(\tilde{x})}
```

where $J = \frac{\partial f}{\partial \tilde{x}} \in \mathbb{R}^{n \times m}$ is the Jacobian of the projection map.

---

## 3. The Constrained Langevin SDE

### 3.1 General Form

Running Langevin dynamics in unconstrained coordinates with the pulled-back score:

```math
d\tilde{x} = -\frac{\beta_t}{2} \lambda_t R R^\top \nabla_{\tilde{x}} U(f(\tilde{x})) \, dt + \sqrt{\beta_t} R \, d\bar{w}
```

where:
- $U(x) = -\log p(x)$ is the potential energy
- $R \in \mathbb{R}^{m \times k}$ is a noise injection matrix
- $\beta_t$ is a time-dependent diffusion coefficient
- $\lambda_t$ is the inverse temperature (controls the strength of the drift toward low-energy regions)
- $\bar{w}$ is $k$-dimensional Brownian motion

### 3.2 Induced Dynamics in Constrained Coordinates

Applying the map $x = f(\tilde{x})$ and using Itô's lemma, the dynamics in $x$-coordinates become:

```math
dx = A \, d\tilde{x}
```

where $A = J = \frac{\partial f}{\partial \tilde{x}}$. Substituting:

```math
dx = A \left[ -\frac{\beta_t}{2} \lambda_t R R^\top \nabla_{\tilde{x}} U \, dt + \sqrt{\beta_t} R \, d\bar{w} \right]
```

Using $\nabla_{\tilde{x}} U = A^\top \nabla_x U$:

```math
dx = -\frac{\beta_t}{2} \lambda_t \, A R R^\top A^\top \nabla_x U \, dt + \sqrt{\beta_t} \, A R \, d\bar{w}
```

---

## 4. Mass Matrix Interpretation

### 4.1 Effective Diffusion Tensor

The induced SDE in $x$-space has the form of **preconditioned Langevin dynamics**:

```math
dx = -\frac{\beta_t}{2} \lambda_t \, M^{-1} \nabla_x U \, dt + \sqrt{\beta_t} \, \Sigma^{1/2} \, dw
```

where the **effective mass matrix** (or inverse preconditioner) is:

```math
M^{-1} = A R R^\top A^\top
```

and the **noise covariance** is:

```math
\Sigma = A R R^\top A^\top = M^{-1}
```

The equality $\Sigma = M^{-1}$ is the **fluctuation-dissipation relation**, ensuring that the dynamics samples the correct equilibrium distribution.

### 4.2 Physical Interpretation

The mass matrix $M = (A R R^\top A^\top)^{-1}$ encodes **anisotropic inertia** in the constrained space:

- **High mass** (large eigenvalues of $M$) → slow dynamics in that direction
- **Low mass** (small eigenvalues of $M$) → fast dynamics in that direction

The projection $A = J$ shapes which directions are easy or hard to explore.

---

## 5. The Linear Case: Explicit Analysis

### 5.1 Setup

Consider a linear projection:

```math
f(\tilde{x}) = A \tilde{x}, \quad A \in \mathbb{R}^{n \times m}
```

For simplicity, let $R = I$ (identity noise injection). The Jacobian is constant: $J = A$.

### 5.2 Unconstrained Dynamics

```math
d\tilde{x} = -\frac{\beta_t}{2} A^\top \nabla_x U(A\tilde{x}) \, dt + \sqrt{\beta_t} \, dw
```

### 5.3 Induced Dynamics in $x$-Space

```math
dx = A \, d\tilde{x} = -\frac{\beta_t}{2} A A^\top \nabla_x U \, dt + \sqrt{\beta_t} \, A \, dw
```

The effective mass matrix is:

```math
M^{-1} = A A^\top
```

### 5.4 Rank and Degeneracy

**Key observation**: If $m < n$ (dimension reduction), then:

```math
\text{rank}(A A^\top) \leq m < n
```

This means $M^{-1}$ is **rank-deficient**, and the mass matrix $M = (A A^\top)^{-1}$ is **ill-defined** (has infinite eigenvalues in the null space).

**Interpretation**: The dynamics can only explore directions in the range of $A$. The null space of $A^\top$ is frozen—there is zero diffusion in these directions, consistent with the constraint that $x$ must lie in $\text{range}(A)$.

### 5.5 Example: Projection onto a Line

Let $A = \begin{pmatrix} 1 \\ -1 \end{pmatrix}^\top / \sqrt{2}$ (normalized), mapping $\mathbb{R}^1 \to \mathbb{R}^2$ onto the line $x_2 = -x_1$.

Then:

```math
A A^\top = \frac{1}{2} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}
```

This is a rank-1 projection matrix onto the line. The induced dynamics:

```math
dx = -\frac{\beta_t}{4} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} \nabla_x U \, dt + \frac{\sqrt{\beta_t}}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix} dw
```

The gradient is projected onto the constraint manifold, and noise is injected only along the manifold—exactly what we want for constrained sampling.

---

## 6. Many-to-One Mappings and Rank Implications

### 6.1 Surjective Projections

When $f: \mathbb{R}^m \to \mathcal{X}$ is **many-to-one** (not injective), multiple unconstrained coordinates map to the same constrained point. This is common for:

- Projections onto lower-dimensional manifolds (e.g., unit circle)
- Radial projections (e.g., normalization $x / \|x\|$)
- Quotient structures (e.g., angles modulo $2\pi$)

### 6.2 Jacobian Rank

For a many-to-one map with $n$-dimensional input and $m$-dimensional output ($m < n$):

```math
\text{rank}(J) \leq m
```

The induced mass matrix $M^{-1} = J J^\top$ inherits this rank bound.

### 6.3 Implications for Sampling

1. **Degeneracy**: The effective covariance $\Sigma = J J^\top$ is degenerate in the normal directions to the image of $f$.

2. **Volume Distortion**: Different unconstrained regions map to the same constrained point with different "weights." This requires a **Jacobian correction** to sample the correct density.

3. **Correct Sampling**: To properly account for volume distortion, one must add a correction term:

```math
d\tilde{x} = \frac{1}{2} \left[ \nabla_{\tilde{x}} \log p(f(\tilde{x})) + \nabla_{\tilde{x}} \log |\det J| \right] dt + dw
```

Without this correction, the sampling is biased toward regions where $|\det J|$ is large (i.e., where the map is locally "expanding").

---

## 7. Does This Sample the Correct Posterior?

### 7.1 Bijective Case ($m = n$, invertible $J$)

If $f$ is a diffeomorphism (smooth bijection with smooth inverse), then the change of variables formula gives:

```math
p_{\tilde{x}}(\tilde{x}) = p(f(\tilde{x})) \cdot |\det J|
```

Running standard Langevin on the pulled-back density $\log p(f(\tilde{x}))$ samples:

```math
\pi_{\text{naive}}(\tilde{x}) \propto p(f(\tilde{x}))
```

This is **not** the correct pushforward unless we add the Jacobian correction:

```math
\pi_{\text{correct}}(\tilde{x}) \propto p(f(\tilde{x})) \cdot |\det J|
```

**Corrected dynamics**:

```math
d\tilde{x} = \frac{1}{2} \nabla_{\tilde{x}} \left[ \log p(f(\tilde{x})) + \log |\det J| \right] dt + dw
```

### 7.2 Non-Bijective Case ($m > n$, rank-deficient $J$)

When the projection is not injective:

1. The unconstrained space has **redundant degrees of freedom**
2. The dynamics explores fibers $f^{-1}(x)$ for each $x \in \mathcal{X}$
3. The marginal over $x$ depends on how the dynamics weights different fibers

**Key result**: Without Jacobian correction, the induced density on $\mathcal{X}$ is:

```math
\tilde{p}(x) \propto p(x) \cdot \int_{f^{-1}(x)} d\mu(\tilde{x})
```

where $\mu$ is the measure induced by the unconstrained equilibrium. This integral is the "volume of the fiber," leading to bias.

### 7.3 When is Naive Sampling Correct?

The naive approach (no Jacobian correction) samples the correct marginal when:

1. **$f$ is linear**: $J$ is constant, so $\nabla \log |\det J| = 0$
2. **$f$ is isometric**: The map preserves volumes locally (conformal maps)
3. **Target is pre-adjusted**: The target $p(x)$ already incorporates the volume factor

---

## 8. Summary

| Aspect | Formula |
|--------|---------|
| Unconstrained dynamics | $d\tilde{x} = \frac{1}{2} \nabla_{\tilde{x}} \log p(f(\tilde{x})) \, dt + dw$ |
| Induced $x$-dynamics | $dx = -\frac{\beta_t}{2} \lambda_t J J^\top \nabla_x U \, dt + \sqrt{\beta_t} J \, dw$ |
| Effective mass matrix | $M = (J J^\top)^{-1}$ |
| Noise covariance | $\Sigma = J J^\top$ |
| Jacobian correction | $+\frac{1}{2} \nabla_{\tilde{x}} \log |\det J| \, dt$ |
| Correct equilibrium | $\pi(\tilde{x}) \propto p(f(\tilde{x})) \cdot |\det J|$ |

### Key Takeaways

1. **Coordinate transformation** converts constrained sampling to unconstrained dynamics
2. The **mass matrix** $M = (J J^\top)^{-1}$ determines anisotropic exploration
3. **Rank deficiency** in $J$ reflects dimensional reduction; null-space directions are frozen
4. **Many-to-one maps** require Jacobian correction for unbiased sampling
5. The **fluctuation-dissipation relation** $\Sigma = M^{-1}$ ensures correct equilibrium

---

## 9. Symmetry Constraints as Linear Coordinate Transformations

Symmetry conditioning is arguably the cleanest real-world example of the constrained sampling framework: it is a **hard constraint**, uses a **linear transformation**, results in a **rank-deficient mass matrix**, and requires **no Jacobian correction**.

### 9.1 Symmetry as a Hard Constraint

Let $G = \{g_1, \dots, g_K\}$ be a finite symmetry group (e.g., cyclic $C_n$, dihedral $D_n$) acting on coordinates $x \in \mathbb{R}^{n \times 3}$. Each $g \in G$ acts via an SE(3) transform:

```math
g \cdot x = O_g x + t_g
```

A structure is **symmetric** if:

```math
x = g \cdot x \quad \forall g \in G
```

Rather than enforcing this via an energy penalty (soft constraint), symmetry can be imposed by **reparameterizing the state space** (hard constraint).

### 9.2 Symmetric Coordinate Map

Introduce unconstrained coordinates $\tilde{x}$ for a single **asymmetric unit** and define:

```math
x = f(\tilde{x}) =
\begin{pmatrix}
g_1 \cdot \tilde{x} \\
g_2 \cdot \tilde{x} \\
\vdots \\
g_K \cdot \tilde{x}
\end{pmatrix}
```

This map is linear (affine if translations are included) and enforces symmetry **by construction**:
- Langevin / reverse-SDE sampling runs only on $\tilde{x}$
- Full coordinates are reconstructed deterministically by $f$
- Noise and drift are injected only along symmetric degrees of freedom

This avoids symmetry breaking by design.

### 9.3 Jacobian and Mass Matrix

Let $\tilde{x} \in \mathbb{R}^m$ (one asymmetric unit) and $x \in \mathbb{R}^{Km}$ ($K$ symmetric copies).

The Jacobian is constant:

```math
J =
\begin{pmatrix}
O_{g_1} \\
O_{g_2} \\
\vdots \\
O_{g_K}
\end{pmatrix}
```

The induced inverse mass matrix is:

```math
M^{-1} = J J^\top =
\begin{pmatrix}
O_{g_1}O_{g_1}^\top & O_{g_1}O_{g_2}^\top & \cdots \\
O_{g_2}O_{g_1}^\top & O_{g_2}O_{g_2}^\top & \cdots \\
\vdots & \vdots & \ddots \\
O_{g_K}O_{g_1}^\top & \cdots & O_{g_K}O_{g_K}^\top
\end{pmatrix}
```

**Properties**:
- $\text{rank}(M^{-1}) = \dim(\tilde{x}) = m$ (rank-deficient when $K > 1$)
- Diffusion is restricted to symmetric directions
- Antisymmetric modes are frozen (infinite effective mass)

### 9.4 Induced Langevin Dynamics

Sampling in $\tilde{x}$-space induces the constrained dynamics:

```math
dx = -\frac{\beta_t}{2} \lambda_t J J^\top \nabla_x U(x) \, dt + \sqrt{\beta_t} \, J \, dw
```

**Interpretation**:
- The gradient $\nabla_x U$ is **group-averaged** by $J J^\top$
- Noise is **shared** across all symmetric copies via $J$
- Antisymmetric modes have **zero diffusion**

This is degenerate preconditioned Langevin dynamics that samples exactly from the symmetric manifold.

### 9.5 No Jacobian Correction Required

Because $f$ is linear and $J$ is constant:

```math
\nabla_{\tilde{x}} \log |\det J| = 0
```

**No Jacobian correction is required**. Symmetry conditioning is therefore an example of **exact constrained sampling** via linear transformation, not approximate posterior reweighting.

### 9.6 Summary

| Property | Symmetry Conditioning |
|----------|----------------------|
| Constraint type | Hard (by construction) |
| Map $f$ | Linear / affine |
| Jacobian $J$ | Constant |
| Mass matrix rank | $\dim(\tilde{x}) < \dim(x)$ |
| Jacobian correction | Not needed |
| Equilibrium | Exact on symmetric subspace |

---

## 10. References

- Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations.
- Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods.
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo.
