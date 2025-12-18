import numpy as np
from pathlib import Path

import torch
import torch.autograd as autograd
import torch.distributions as D
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

torch.manual_seed(42)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def default_out_dir() -> Path:
    # Keep generated outputs out of the repo.
    out = Path.home() / "temp_scripts" / "1st-principle-diffusion" / "manifold"
    out.mkdir(parents=True, exist_ok=True)
    return out

def linear_projec(X_unc: torch.Tensor) -> torch.Tensor:
    vec = [0.5, -0.5]
    P = torch.Tensor([vec, vec]).to(X_unc.device)
    return X_unc @ P


def map_to_unit_circle(x: torch.Tensor, r: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    # Project points onto a circle of radius r, preserving direction.
    norms = torch.norm(x, dim=1, keepdim=True).clamp_min(eps)
    return r * x / norms


def build_distribution(device=DEVICE, k=20):
    mix = D.Categorical(torch.ones(k, ).to(device))
    R = torch.randn(k, 2, 2).to(device)
    mu = 2.0 * torch.randn(k, 2).to(device)
    cov = 0.5 * torch.einsum("bij,bkj->bik", R, R) + 0.1 * torch.eye(2).to(device)
    comp = D.Independent(D.MultivariateNormal(mu, cov), 0)
    gmm = D.MixtureSameFamily(mix, comp)
    return gmm

@torch.no_grad()
def score_function(X, distribution, conditioner=None):

    with torch.enable_grad():
        _X = X.clone()
        _X.requires_grad_(True)
        X_input = _X 

        if conditioner is not None:
            X_input = conditioner(X_input)

        logp = distribution.log_prob(X_input).sum()
        dUdX = autograd.grad(logp, _X)[0]

    return dUdX

def langevin_dynamics(distribution, conditioner=None,
                        num_steps=1000, num_batch=10000, step_size=0.01, grad_correction=None, device=DEVICE):
    
    X_unc_t = torch.randn([num_batch, 2], device=device)

    for t in range(num_steps):
        dUdX = score_function(X_unc_t, distribution, conditioner)

        if grad_correction is not None:

            X_unc_t = X_unc_t.clone().detach()
            X_unc_t.requires_grad_(True)

            dlogDetJacobian_grad = grad_correction(X_unc_t)[1]
            dUdX = dUdX + dlogDetJacobian_grad 

            X_unc_t = X_unc_t.detach()

        X_unc_t = X_unc_t + 0.5 * dUdX * step_size + np.sqrt(step_size) * torch.randn_like(X_unc_t)

    X_t = X_unc_t if conditioner is None else conditioner(X_unc_t)

    return X_t, X_unc_t


@torch.no_grad()
def langevin_dynamics_with_trace(
    distribution,
    conditioner=None,
    *,
    num_steps: int = 1000,
    num_batch: int = 10000,
    step_size: float = 0.01,
    device: str = DEVICE,
    trace_batch: int = 128,
    trace_every: int = 50,
    record_all_for_marginal: bool = False,
):
    """
    Same dynamics as `langevin_dynamics`, but records a small subset of particles
    over time so we can visualize the integration → projection story.
    
    Args:
        record_all_for_marginal: If True, also record ALL samples (X_all) at each
            trace step for computing accurate marginal densities. The traced subset
            (X_unc, X) is still recorded for trajectory visualization.
    """
    X_unc_t = torch.randn([num_batch, 2], device=device)
    trace_ix = torch.randperm(num_batch, device=device)[: min(trace_batch, num_batch)]

    trace = {
        "t": [],
        "X_unc": [],  # list[Tensor(trace_batch, 2)] on CPU - for trajectory viz
        "X": [],      # list[Tensor(trace_batch, 2)] on CPU - for trajectory viz
    }
    if record_all_for_marginal:
        trace["X_all"] = []  # list[Tensor(num_batch, 2)] on CPU - for marginal

    def _record(t: int):
        u = X_unc_t[trace_ix].detach().cpu()
        x = u if conditioner is None else conditioner(X_unc_t[trace_ix]).detach().cpu()
        trace["t"].append(int(t))
        trace["X_unc"].append(u)
        trace["X"].append(x)
        if record_all_for_marginal:
            x_all = X_unc_t if conditioner is None else conditioner(X_unc_t)
            trace["X_all"].append(x_all.detach().cpu())

    _record(0)
    for t in range(1, num_steps + 1):
        dUdX = score_function(X_unc_t, distribution, conditioner)
        X_unc_t = X_unc_t + 0.5 * dUdX * step_size + np.sqrt(step_size) * torch.randn_like(X_unc_t)

        if (t % trace_every == 0) or (t == num_steps):
            _record(t)

    X_t = X_unc_t if conditioner is None else conditioner(X_unc_t)
    return X_t, X_unc_t, trace


def soft_plus(_x, R=2):
    r = torch.norm(_x, dim=1)[:, None].clamp_min(1e-12)
    v = _x / r 
    # Smooth radial "soft" projection towards radius R.
    # Use numerically-stable softplus: softplus(z) = log(1 + exp(z)).
    # WARNING: Has singularity at origin due to v = x/||x||.
    return v * (R - F.softplus(R - r))


def algebraic_disk(x: torch.Tensor, R: float = 2.0) -> torch.Tensor:
    """
    Smooth bijection from R^2 -> Open Disk of radius R.
    
    f(x) = x * R / sqrt(R^2 + |x|^2)
    
    Properties:
    - Smooth everywhere (C^∞), no singularity at origin
    - f(0) = 0, J(0) = I (identity-like near origin)
    - |f(x)| -> R as |x| -> ∞
    - Jacobian is well-conditioned everywhere
    
    Args:
        x: Tensor of shape (B, 2)
        R: Radius of the target disk
    
    Returns:
        Tensor of shape (B, 2) with |output| < R
    """
    r2 = torch.sum(x**2, dim=1, keepdim=True)
    scale = R / torch.sqrt(R**2 + r2)
    return x * scale


def tanh_disk(x: torch.Tensor, R: float = 2.0, scale: float = 1.0) -> torch.Tensor:
    """
    Smooth bijection from R^2 -> Open Disk of radius R using tanh.
    
    f(x) = x * R * tanh(|x| / scale) / (|x| + ε)
    
    To avoid singularity at origin, we use a smooth approximation:
    f(x) = x * R * tanh(|x| / scale) / sqrt(|x|^2 + ε^2)
    
    Properties:
    - Smooth everywhere (C^∞)
    - f(0) = 0
    - |f(x)| -> R as |x| -> ∞
    - Steeper saturation than algebraic (controlled by scale parameter)
    
    Args:
        x: Tensor of shape (B, 2)
        R: Radius of the target disk
        scale: Controls steepness of saturation (smaller = faster saturation)
    
    Returns:
        Tensor of shape (B, 2) with |output| < R
    """
    eps = 1e-6
    r2 = torch.sum(x**2, dim=1, keepdim=True)
    r = torch.sqrt(r2 + eps**2)  # Soft norm, smooth at origin
    
    # tanh(r/scale) goes from 0 to 1
    # Multiply by R to get radius, divide by r to get unit direction
    saturation = torch.tanh(r / scale)
    return x * R * saturation / r


def sigmoid_disk(x: torch.Tensor, R: float = 2.0, scale: float = 1.0) -> torch.Tensor:
    """
    Smooth bijection from R^2 -> Open Disk of radius R using sigmoid.
    
    f(x) = x * R * 2 * (sigmoid(|x| / scale) - 0.5) / sqrt(|x|^2 + ε^2)
    
    This maps: |x|=0 -> |f|=0, |x|->∞ -> |f|->R
    
    Properties:
    - Smooth everywhere (C^∞)
    - f(0) = 0 (since sigmoid(0)-0.5 = 0)
    - |f(x)| -> R as |x| -> ∞
    - Different saturation curve than tanh
    
    Args:
        x: Tensor of shape (B, 2)
        R: Radius of the target disk
        scale: Controls steepness of saturation
    
    Returns:
        Tensor of shape (B, 2) with |output| < R
    """
    eps = 1e-6
    r2 = torch.sum(x**2, dim=1, keepdim=True)
    r = torch.sqrt(r2 + eps**2)  # Soft norm
    
    # 2*(sigmoid(r/scale) - 0.5) goes from 0 to 1 as r goes from 0 to ∞
    saturation = 2 * (torch.sigmoid(r / scale) - 0.5)
    return x * R * saturation / r


def star_shape(x: torch.Tensor, R: float = 2.0, n_points: int = 5, amplitude: float = 0.3) -> torch.Tensor:
    """
    Smooth bijection from R^2 -> Star-shaped region.
    
    The boundary radius varies as: r(θ) = R * (1 + amplitude * cos(n*θ))
    We use angle-preserving algebraic projection with varying radius.
    
    This is smooth because we express cos(n*θ) in terms of (x, y) directly
    using complex arithmetic, avoiding arctan2.
    
    Properties:
    - Smooth everywhere (C^∞)
    - f(0) = 0
    - Maps to interior of n-pointed star
    
    Args:
        x: Tensor of shape (B, 2)
        R: Base radius of the star
        n_points: Number of star points (e.g., 5 for 5-pointed star)
        amplitude: Modulation amplitude (0 = circle, larger = more pointy)
    
    Returns:
        Tensor of shape (B, 2) inside star region
    """
    eps = 1e-12
    r2 = torch.sum(x**2, dim=1, keepdim=True).clamp_min(eps)
    r = torch.sqrt(r2)
    
    # Compute cos(n*θ) using complex arithmetic: Re[(x + iy)^n] / r^n
    z = torch.complex(x[:, 0], x[:, 1])
    z_n = z ** n_points
    cos_n_theta = z_n.real / (r[:, 0] ** n_points + eps)
    
    # Modulated radius at each angle
    R_theta = R * (1 + amplitude * cos_n_theta.unsqueeze(1))
    
    # Algebraic projection to this varying radius
    scale = R_theta / torch.sqrt(R_theta**2 + r2)
    return x * scale


def grad_correction(f, _x, eps: float = 1e-12):
    """
    Compute log volume change (log|det J_f|) and its gradient wrt x for a batched map f.
    
    SLOW VERSION - uses full Jacobian computation. Use grad_correction_fast for 2D.

    Args:
        f: function mapping Tensor[B,2] -> Tensor[B,2]
        _x: Tensor[B,2]
        eps: numerical stability for log(|det| + eps)

    Returns:
        dvol: Tensor[B]          (log|det J| per sample)
        grad: Tensor[B,2]        (∇_x sum_i dvol_i)
    """
    original_device = _x.device
    # MPS doesn't support linalg_lu_solve needed for Jacobian backward.
    # Fall back to CPU for this computation.
    if _x.device.type == "mps":
        _x = _x.cpu()

    if not _x.requires_grad:
        _x = _x.clone().detach().requires_grad_(True)

    jac = torch.autograd.functional.jacobian(f, _x, create_graph=True, vectorize=True)
    # Shape: (B, 2, B, 2) -> take per-sample Jacobians: (B, 2, 2)
    jac = jac[torch.arange(_x.shape[0]), :, torch.arange(_x.shape[0]), :]

    det = torch.linalg.det(jac)
    dvol = torch.log(det.abs() + eps)
    grad = autograd.grad(dvol.sum(), _x, retain_graph=False, create_graph=False)[0]
    
    # Move back to original device
    return dvol.to(original_device), grad.to(original_device)


def grad_correction_2d(f, _x, eps: float = 1e-12):
    """
    FAST: Compute log|det J| and its gradient for 2D->2D maps.
    
    Uses efficient per-element gradient computation instead of full Jacobian.
    O(B) memory instead of O(B²).
    
    For f: R² -> R², the Jacobian is:
        J = [[∂f₀/∂x₀, ∂f₀/∂x₁],
             [∂f₁/∂x₀, ∂f₁/∂x₁]]
    
    det(J) = J₀₀·J₁₁ - J₀₁·J₁₀
    
    Args:
        f: function mapping Tensor[B,2] -> Tensor[B,2]
        _x: Tensor[B,2]
        eps: numerical stability
    
    Returns:
        dvol: Tensor[B]    (log|det J| per sample)
        grad: Tensor[B,2]  (∇_x log|det J|)
    """
    if not _x.requires_grad:
        _x = _x.clone().detach().requires_grad_(True)
    
    # Forward pass
    y = f(_x)  # (B, 2)
    
    # Compute Jacobian columns using vector-Jacobian products
    # Column j of J = ∂f/∂x_j
    # We use grad with grad_outputs to get rows, then transpose mentally
    
    B = _x.shape[0]
    
    # ∂f₀/∂x (gives [∂f₀/∂x₀, ∂f₀/∂x₁] for all batch)
    grad_f0 = autograd.grad(y[:, 0].sum(), _x, create_graph=True)[0]  # (B, 2)
    # ∂f₁/∂x (gives [∂f₁/∂x₀, ∂f₁/∂x₁] for all batch)  
    grad_f1 = autograd.grad(y[:, 1].sum(), _x, create_graph=True)[0]  # (B, 2)
    
    # Jacobian elements
    J00 = grad_f0[:, 0]  # ∂f₀/∂x₀
    J01 = grad_f0[:, 1]  # ∂f₀/∂x₁
    J10 = grad_f1[:, 0]  # ∂f₁/∂x₀
    J11 = grad_f1[:, 1]  # ∂f₁/∂x₁
    
    # Determinant
    det = J00 * J11 - J01 * J10
    dvol = torch.log(det.abs() + eps)
    
    # Gradient of log|det| w.r.t. x
    grad_dvol = autograd.grad(dvol.sum(), _x, retain_graph=False)[0]
    
    return dvol.detach(), grad_dvol.detach()


# ---- Plotting Functions and Helpers (Made Importable) ----

def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _hist_line(ax, values: np.ndarray, *, bins: np.ndarray, label: str, color, alpha: float = 0.55, lw: float = 1.1):
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, hist, color=color, alpha=alpha, lw=lw, label=label)
    return centers, hist

def plot_algorithm_flow_marginal_recovery(
    *,
    distribution,
    conditioner,
    trace: dict,
    X_final: torch.Tensor,
    X_final_unc: torch.Tensor,
    out_name: str,
    out_dir: Path | None = None,
    lim: float = 5.0,
    N: int = 150,
    manifold_kind: str | None = None,  # e.g. "circle" or "line"
    circle_r: float = 1.0,
    marginal_kind: str = "theta",  # "theta" or "x1"
    ref_curve: tuple[np.ndarray, np.ndarray] | None = None,
    n_snapshots: int = 5,
) -> Path:
    """
    One figure that reads left→right as:
      pull-back/update/project (top, text) and what happens in practice (bottom)

    Bottom row:
      - u-space trace subset
      - projection mapping u(T) -> x(T) with a small inset showing x(T) on density
      - projected marginal over time snapshots vs a reference curve
    """
    out_dir = default_out_dir() if out_dir is None else out_dir

    # Background density grid (for the target in x-space).
    x = np.linspace(-lim, lim, N)
    Xg, Yg = np.meshgrid(x, x)
    XY = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32, device=DEVICE)
    P_grid = torch.exp(distribution.log_prob(XY)).reshape(N, N).detach().cpu().numpy()

    fig = plt.figure(figsize=(12.8, 6.2), dpi=320, constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.62, 1.0], width_ratios=[1.05, 1.05, 1.10])

    ax_t0 = fig.add_subplot(gs[0, 0])
    ax_t1 = fig.add_subplot(gs[0, 1])
    ax_t2 = fig.add_subplot(gs[0, 2])
    ax_u = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])
    ax_marg = fig.add_subplot(gs[1, 2])

    # ---- Top row: concept-first (no axes) ----
    for ax in (ax_t0, ax_t1, ax_t2):
        ax.set_axis_off()

    ax_t0.text(
        0.02,
        0.72,
        "A. Pull-back score through the projection\n"
        r"evaluate $\log p(x)$ at $x=\Pi(u)$ and pull back to $u$:\n"
        r"$\nabla_u \log p(\Pi(u))$",
        va="top",
        ha="left",
        fontsize=10.2,
    )
    ax_t1.text(
        0.02,
        0.72,
        "B. Langevin update in unconstrained coordinates (u)\n"
        r"$u_{t+1}=u_t+\frac{\eta}{2}\nabla_u\log p(\Pi(u_t))+\sqrt{\eta}\,\xi_t$",
        va="top",
        ha="left",
        fontsize=10.2,
    )
    ax_t2.text(
        0.02,
        0.72,
        "C. Project to the constrained space and compare marginals\n"
        r"$x_t=\Pi(u_t)$ ; check that projected marginal matches reference",
        va="top",
        ha="left",
        fontsize=10.2,
    )
    ax_t0.annotate(
        "",
        xy=(1.02, 0.50),
        xytext=(0.86, 0.50),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.2, color="0.2"),
    )
    ax_t1.annotate(
        "",
        xy=(1.02, 0.50),
        xytext=(0.86, 0.50),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.2, color="0.2"),
    )

    # ---- Bottom left: u-space integration trace ----
    # Trace arrays are recorded for a subset of particles: shape (Trec, B, 2).
    U = np.stack([u.numpy() for u in trace["X_unc"]], axis=0)  # (Trec, B, 2)
    t_rec = np.array(trace["t"], dtype=np.float32)
    B = U.shape[1]
    # Consistent particle-identity colors across panels.
    id_cmap = plt.get_cmap("tab20")
    id_colors = np.stack([id_cmap((i % 20) / 20.0) for i in range(B)], axis=0)

    # Light trajectories per particle (identity-colored), plus start/end markers.
    for b in range(B):
        ax_u.plot(U[:, b, 0], U[:, b, 1], color=id_colors[b], lw=0.9, alpha=0.35)
    ax_u.scatter(U[0, :, 0], U[0, :, 1], s=18, c=id_colors, alpha=0.85, linewidths=0.0, label="u(0)")
    ax_u.scatter(U[-1, :, 0], U[-1, :, 1], s=28, c=id_colors, alpha=0.95, linewidths=0.0, label="u(T)")
    ax_u.set_title("B. Integrate in u-space (trace subset, identity-colored)")
    ax_u.set_xlabel("u1")
    ax_u.set_ylabel("u2")
    ax_u.legend(loc="upper right", fontsize=9)
    u_lim = float(np.max(np.abs(X_final_unc.detach().cpu().numpy())) * 1.05 + 1e-9)
    ax_u.set_xlim(-u_lim, u_lim)
    ax_u.set_ylim(-u_lim, u_lim)
    ax_u.set_aspect("equal", adjustable="box")
    ax_u.set_box_aspect(1)

    # ---- Bottom middle: projection mapping u(T) -> x(T) + inset x(T) on target density ----
    X_unc_np = X_final_unc.detach().cpu().numpy()
    X_np = X_final.detach().cpu().numpy()
    # Use the traced subset (identity-colored) for an easy-to-follow mapping story.
    U_T = U[-1]  # (B, 2)
    X_T = trace["X"][-1].numpy()  # (B, 2)
    for b in range(B):
        ax_map.plot([U_T[b, 0], X_T[b, 0]], [U_T[b, 1], X_T[b, 1]], color=id_colors[b], alpha=0.22, lw=1.0)
    ax_map.scatter(U_T[:, 0], U_T[:, 1], s=22, c=id_colors, alpha=0.85, linewidths=0.0, label="u(T)")
    ax_map.scatter(X_T[:, 0], X_T[:, 1], s=26, c=id_colors, alpha=0.95, linewidths=0.0, label=r"$x(T)=\Pi(u(T))$")
    ax_map.set_title("C. Project: u(T) → x(T) (same particles)")
    ax_map.set_xlabel("coord 1")
    ax_map.set_ylabel("coord 2")
    ax_map.legend(loc="upper right", fontsize=9)
    lim_map = float(max(np.max(np.abs(U_T)), np.max(np.abs(X_T))) * 1.15 + 1e-9)
    ax_map.set_xlim(-lim_map, lim_map)
    ax_map.set_ylim(-lim_map, lim_map)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_box_aspect(1)

    ax_in = ax_map.inset_axes([0.57, 0.05, 0.40, 0.40])
    cf = ax_in.contourf(Xg, Yg, P_grid, levels=28, cmap=plt.get_cmap("GnBu"), alpha=0.92)
    # Show all projected samples in the inset for density context.
    ax_in.scatter(X_np[:, 0], X_np[:, 1], c="black", s=0.5, linewidths=0, alpha=0.70)
    if manifold_kind == "circle":
        th = np.linspace(-np.pi, np.pi, 400)
        ax_in.plot(circle_r * np.cos(th), circle_r * np.sin(th), color="white", lw=1.1, alpha=0.9)
    elif manifold_kind == "line":
        xx = np.linspace(-lim, lim, 200)
        ax_in.plot(xx, -xx, color="white", lw=1.1, alpha=0.9)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_xlim(-lim, lim)
    ax_in.set_ylim(-lim, lim)
    ax_in.set_title("x(T) on target", fontsize=8.8, pad=2.0)
    cax_in = ax_in.inset_axes([0.12, 0.92, 0.76, 0.10])
    cbx = fig.colorbar(cf, cax=cax_in, orientation="horizontal")
    cbx.ax.xaxis.set_ticks_position("top")
    cbx.ax.tick_params(labelsize=7)

    # ---- Bottom right: projected marginal over time vs reference ----
    if ref_curve is None:
        raise ValueError("ref_curve is required to show marginal recovery.")
    grid, ref_pdf = ref_curve
    # Use dark color
    dark = plt.get_cmap("GnBu")(0.95)
    ax_marg.plot(grid, ref_pdf, color=dark, lw=2.0, label="reference", zorder=3)

    # choose snapshot indices spaced through trace
    Trec = len(trace["t"])
    if Trec <= n_snapshots:
        snap_ix = list(range(Trec))
    else:
        snap_ix = np.linspace(0, Trec - 1, n_snapshots).round().astype(int).tolist()
        snap_ix = sorted(set(snap_ix))

    # Precompute a simple convergence metric: L1 distance between histogram and reference.
    conv_t = []
    conv_l1 = []

    if marginal_kind == "theta":
        bins = np.linspace(-np.pi, np.pi, 121)
        for j, i in enumerate(snap_ix):
            Xs = trace["X"][i].numpy()
            theta = np.arctan2(Xs[:, 1], Xs[:, 0])
            centers, hist = _hist_line(
                ax_marg,
                theta,
                bins=bins,
                label=f"t={trace['t'][i]}",
                color=plt.get_cmap("viridis")(0.20 + 0.70 * (j / max(1, len(snap_ix) - 1))),
                alpha=0.70,
                lw=1.1,
            )
            ref_interp = np.interp(centers, grid, ref_pdf)
            dx = float(centers[1] - centers[0])
            conv_t.append(float(trace["t"][i]))
            conv_l1.append(float(np.sum(np.abs(hist - ref_interp)) * dx))
        ax_marg.set_xlim(-np.pi, np.pi)
        ax_marg.set_xlabel(r"$\theta$")
        ax_marg.set_title(r"Projected marginal over time (density in $\theta$)")
        ax_marg.xaxis.set_major_locator(mticker.FixedLocator([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]))
        ax_marg.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    elif marginal_kind == "x1":
        bins = np.linspace(float(grid.min()), float(grid.max()), 121)
        for j, i in enumerate(snap_ix):
            Xs = trace["X"][i].numpy()
            x1 = Xs[:, 0]
            centers, hist = _hist_line(
                ax_marg,
                x1,
                bins=bins,
                label=f"t={trace['t'][i]}",
                color=plt.get_cmap("viridis")(0.20 + 0.70 * (j / max(1, len(snap_ix) - 1))),
                alpha=0.70,
                lw=1.1,
            )
            ref_interp = np.interp(centers, grid, ref_pdf)
            dx = float(centers[1] - centers[0])
            conv_t.append(float(trace["t"][i]))
            conv_l1.append(float(np.sum(np.abs(hist - ref_interp)) * dx))
        ax_marg.set_xlabel("x1")
        ax_marg.set_title("Projected marginal over time (density in x1)")
    else:
        raise ValueError(f"Unknown marginal_kind={marginal_kind!r}")

    ax_marg.set_ylabel("density")
    ax_marg.legend(loc="upper right", fontsize=8.4, ncol=1)

    # Inset: convergence metric over time (L1 distance between histogram and reference).
    ax_err = ax_marg.inset_axes([0.14, 0.10, 0.50, 0.30])
    if len(conv_t) >= 2:
        ax_err.plot(conv_t, conv_l1, color="0.2", lw=1.2)
        ax_err.scatter([conv_t[-1]], [conv_l1[-1]], color="0.2", s=10)
    ax_err.set_title(r"$\int | \hat p_t - p_{\mathrm{ref}} |$", fontsize=8.0, pad=1.0)
    ax_err.set_xlabel("t", fontsize=7.6, labelpad=1.0)
    ax_err.set_ylabel("L1", fontsize=7.6, labelpad=1.0)
    ax_err.tick_params(axis="both", labelsize=7)

    # Ticks / grid feel consistency
    for ax in (ax_u, ax_map, ax_marg):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.set_axisbelow(True)

    out_path = out_dir / out_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
    return out_path

def plot_algorithm_overview_single_figure(
    *,
    distribution,
    conditioner,
    trace: dict,
    X_final: torch.Tensor,
    X_final_unc: torch.Tensor,
    out_name: str,
    out_dir: Path | None = None,
    lim: float = 5.0,
    N: int = 150,
    manifold_kind: str | None = None,  # "circle" or "line"
    circle_r: float = 1.0,
    marginal_kind: str = "theta",  # "theta" or "x1"
    ref_curve: tuple[np.ndarray, np.ndarray] | None = None,
    n_snapshots: int = 6,
    connect_n: int = 80,
) -> Path:
    """
    Single 2x3 overview figure designed to be readable without any text in-plot:
      Row 1: u-space trajectories | projection map u(T)->x(T) | x-space density + samples
      Row 2: u(T) cloud           | x(T) cloud on manifold     | marginal snapshots + error inset
    All axes suppress titles/labels/legends/tick labels.
    """
    if ref_curve is None:
        raise ValueError("ref_curve is required for the marginal panel.")

    out_dir = default_out_dir() if out_dir is None else out_dir

    # Background density grid (x-space).
    x = np.linspace(-lim, lim, N)
    Xg, Yg = np.meshgrid(x, x)
    XY = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32, device=DEVICE)
    P_grid = torch.exp(distribution.log_prob(XY)).reshape(N, N).detach().cpu().numpy()

    fig = plt.figure(figsize=(13.0, 7.2), dpi=320, constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.15])
    ax_u_traj = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_x = fig.add_subplot(gs[0, 2])
    ax_uT = fig.add_subplot(gs[1, 0])
    ax_xT = fig.add_subplot(gs[1, 1])
    ax_marg = fig.add_subplot(gs[1, 2])

    def _mute(ax):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_alpha(0.35)

    for ax in (ax_u_traj, ax_map, ax_x, ax_uT, ax_xT, ax_marg):
        _mute(ax)

    # ---- u-space trajectories (time-colored) ----
    U = np.stack([u.numpy() for u in trace["X_unc"]], axis=0)  # (Trec, B, 2) on CPU already
    t_rec = np.asarray(trace["t"], dtype=np.float32)
    Trec, B, _ = U.shape
    # Use a time-based colormap; no colorbar.
    time_cmap = plt.get_cmap("viridis")
    t_norm = (t_rec - t_rec.min()) / (t_rec.max() - t_rec.min() + 1e-12)

    # Draw each particle as a LineCollection across time.
    for b in range(B):
        pts = U[:, b, :]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (Trec-1, 2, 2)
        lc = LineCollection(segs, cmap=time_cmap, linewidths=0.9, alpha=0.55)
        lc.set_array(t_norm[1:])  # color by segment end time
        ax_u_traj.add_collection(lc)
    # Start/end markers (no legend)
    ax_u_traj.scatter(U[0, :, 0], U[0, :, 1], s=12, c=time_cmap(0.05), alpha=0.65, linewidths=0.0)
    ax_u_traj.scatter(U[-1, :, 0], U[-1, :, 1], s=18, c=time_cmap(0.95), alpha=0.85, linewidths=0.0)
    u_lim = float(np.max(np.abs(X_final_unc.detach().cpu().numpy())) * 1.05 + 1e-9)
    ax_u_traj.set_xlim(-u_lim, u_lim)
    ax_u_traj.set_ylim(-u_lim, u_lim)
    ax_u_traj.set_aspect("equal", adjustable="box")
    ax_u_traj.set_box_aspect(1)

    # ---- Projection map u(T) -> x(T) (same subset) ----
    UT = U[-1]  # (B, 2)
    XT = trace["X"][-1].numpy()  # (B, 2)
    # Light connectors
    if connect_n > 0:
        m = min(B, connect_n)
        sel = np.linspace(0, B - 1, m).round().astype(int)
        for i in sel:
            ax_map.plot([UT[i, 0], XT[i, 0]], [UT[i, 1], XT[i, 1]], color="0.2", alpha=0.18, lw=1.0)
    ax_map.scatter(UT[:, 0], UT[:, 1], s=16, c=time_cmap(0.90), alpha=0.65, linewidths=0.0)
    ax_map.scatter(XT[:, 0], XT[:, 1], s=18, c=time_cmap(0.25), alpha=0.85, linewidths=0.0)
    lim_map = float(max(np.max(np.abs(UT)), np.max(np.abs(XT))) * 1.15 + 1e-9)
    ax_map.set_xlim(-lim_map, lim_map)
    ax_map.set_ylim(-lim_map, lim_map)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_box_aspect(1)

    # ---- x-space density + all projected samples ----
    X_np = X_final.detach().cpu().numpy()
    ax_x.contourf(Xg, Yg, P_grid, levels=32, cmap=plt.get_cmap("GnBu"), alpha=0.92)
    ax_x.scatter(X_np[:, 0], X_np[:, 1], c="black", s=0.35, linewidths=0, alpha=0.70)
    if manifold_kind == "circle":
        th = np.linspace(-np.pi, np.pi, 500)
        ax_x.plot(circle_r * np.cos(th), circle_r * np.sin(th), color="white", lw=1.1, alpha=0.9)
    elif manifold_kind == "line":
        xx = np.linspace(-lim, lim, 300)
        ax_x.plot(xx, -xx, color="white", lw=1.1, alpha=0.9)
    ax_x.set_xlim(-lim, lim)
    ax_x.set_ylim(-lim, lim)
    ax_x.set_aspect("equal", adjustable="box")
    ax_x.set_box_aspect(1)

    # ---- u(T) cloud ----
    ax_uT.scatter(UT[:, 0], UT[:, 1], c="black", s=10, linewidths=0.0, alpha=0.75)
    ax_uT.set_xlim(-lim_map, lim_map)
    ax_uT.set_ylim(-lim_map, lim_map)
    ax_uT.set_aspect("equal", adjustable="box")
    ax_uT.set_box_aspect(1)

    # ---- x(T) cloud on manifold (trace subset) ----
    ax_xT.contourf(Xg, Yg, P_grid, levels=32, cmap=plt.get_cmap("GnBu"), alpha=0.92)
    ax_xT.scatter(XT[:, 0], XT[:, 1], c="black", s=10, linewidths=0.0, alpha=0.75)
    if manifold_kind == "circle":
        th = np.linspace(-np.pi, np.pi, 500)
        ax_xT.plot(circle_r * np.cos(th), circle_r * np.sin(th), color="white", lw=1.1, alpha=0.9)
    elif manifold_kind == "line":
        xx = np.linspace(-lim, lim, 300)
        ax_xT.plot(xx, -xx, color="white", lw=1.1, alpha=0.9)
    ax_xT.set_xlim(-lim, lim)
    ax_xT.set_ylim(-lim, lim)
    ax_xT.set_aspect("equal", adjustable="box")
    ax_xT.set_box_aspect(1)

    # ---- Marginal snapshots + error inset ----
    grid, ref_pdf = ref_curve
    ax_marg.plot(grid, ref_pdf, color="0.15", lw=2.0, alpha=0.95)

    # Choose snapshots (indices in trace recordings)
    if Trec <= n_snapshots:
        snap_ix = list(range(Trec))
    else:
        snap_ix = np.linspace(0, Trec - 1, n_snapshots).round().astype(int).tolist()
        snap_ix = sorted(set(snap_ix))

    conv_t = []
    conv_l1 = []

    if marginal_kind == "theta":
        bins = np.linspace(-np.pi, np.pi, 121)
        centers = 0.5 * (bins[:-1] + bins[1:])
        dx = float(centers[1] - centers[0])
        for j, i in enumerate(snap_ix):
            Xs = trace["X"][i].numpy()
            theta = np.arctan2(Xs[:, 1], Xs[:, 0])
            hist, _ = np.histogram(theta, bins=bins, density=True)
            color = time_cmap(0.15 + 0.75 * (j / max(1, len(snap_ix) - 1)))
            ax_marg.plot(centers, hist, color=color, lw=1.2, alpha=0.75)
            ref_interp = np.interp(centers, grid, ref_pdf)
            conv_t.append(float(trace["t"][i]))
            conv_l1.append(float(np.sum(np.abs(hist - ref_interp)) * dx))
        ax_marg.set_xlim(-np.pi, np.pi)
    elif marginal_kind == "x1":
        bins = np.linspace(float(grid.min()), float(grid.max()), 121)
        centers = 0.5 * (bins[:-1] + bins[1:])
        dx = float(centers[1] - centers[0])
        for j, i in enumerate(snap_ix):
            Xs = trace["X"][i].numpy()
            x1 = Xs[:, 0]
            hist, _ = np.histogram(x1, bins=bins, density=True)
            color = time_cmap(0.15 + 0.75 * (j / max(1, len(snap_ix) - 1)))
            ax_marg.plot(centers, hist, color=color, lw=1.2, alpha=0.75)
            ref_interp = np.interp(centers, grid, ref_pdf)
            conv_t.append(float(trace["t"][i]))
            conv_l1.append(float(np.sum(np.abs(hist - ref_interp)) * dx))
        ax_marg.set_xlim(float(grid.min()), float(grid.max()))
    else:
        raise ValueError(f"Unknown marginal_kind={marginal_kind!r}")

    # No tick labels, but keep a tight view on y for visibility.
    ax_marg.relim()
    ax_marg.autoscale_view()

    # Error inset: curve only, no text/ticks.
    ax_err = ax_marg.inset_axes([0.58, 0.08, 0.38, 0.32])
    _mute(ax_err)
    if len(conv_t) >= 2:
        ax_err.plot(conv_t, conv_l1, color="0.15", lw=1.4, alpha=0.9)
        ax_err.scatter([conv_t[-1]], [conv_l1[-1]], color="0.15", s=10, alpha=0.9)

    out_path = out_dir / out_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
    return out_path

def plot_triptych(
    *,
    X_samples: torch.Tensor,
    X_samples_unc: torch.Tensor,
    out_name: str,
    out_dir: Path | None = None,
    marginal_kind: str = "x1",
    theta_ref_curve: tuple[np.ndarray, np.ndarray] | None = None,
    lim: float = 5.0,
    N: int = 150,
) -> Path:
    # Background density grid (for the original unconstrained target).
    x = np.linspace(-lim, lim, N)
    X, Y = np.meshgrid(x, x)
    XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=DEVICE)
    # Rebuild distribution for plotting since it's not passed in
    gmm = build_distribution()
    P_grid = torch.exp(gmm.log_prob(XY)).reshape(N, N).detach().cpu().numpy()

    # Use a GridSpec; keep axes aligned and place a horizontal colorbar
    # as an inset on the first subplot (so the axis size isn't changed).
    fig = plt.figure(figsize=(11.2, 3.6), dpi=400, constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    cmap = plt.get_cmap("GnBu")
    accent = cmap(0.75)
    dark = cmap(0.95)

    # (1) Conditioned samples on original density background
    cf = ax0.contourf(X, Y, P_grid, levels=40, cmap=cmap, alpha=0.92)
    ax0.scatter(
        to_np(X_samples[:, 0]),
        to_np(X_samples[:, 1]),
        c="black",
        s=0.6,
        linewidths=0,
        label="samples",
        alpha=0.9,
    )
    ax0.set_title("Projected samples")
    ax0.set_xlim(-lim, lim)
    ax0.set_ylim(-lim, lim)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_box_aspect(1)
    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    # Horizontal colorbar inset at the top of subplot 1 (inside the axes,
    # below the title, so it doesn't distort subplot alignment).
    cax_in = ax0.inset_axes([0.16, 0.90, 0.68, 0.035])  # [x0, y0, w, h] in axes fraction
    cb = fig.colorbar(cf, cax=cax_in, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("density", labelpad=2.0, fontsize=11)
    cb.ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    # (2) Unconstrained coordinates
    X_unc_np = to_np(X_samples_unc)
    u_lim = float(np.max(np.abs(X_unc_np)) * 1.05 + 1e-9)
    ax1.scatter(X_unc_np[:, 0], X_unc_np[:, 1], c="black", s=0.6, linewidths=0, alpha=0.9)
    ax1.set_title("Unconstrained coordinates")
    ax1.set_xlim(-u_lim, u_lim)
    ax1.set_ylim(-u_lim, u_lim)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_box_aspect(1)
    ax1.set_xlabel("u1")
    ax1.set_ylabel("u2")

    # (3) Marginal
    if marginal_kind == "x1":
        xs = to_np(X_samples[:, 0])
        ax2.hist(xs, density=True, bins=80, color=accent, alpha=0.35, label="samples")

        P = torch.exp(gmm.log_prob(X_samples)).detach().cpu().numpy()
        Z = np.trapz(P, x=xs)
        ax2.plot(xs, P / (Z + 1e-12), color=dark, linewidth=1.4, label="exact density")
        ax2.set_title("Marginal (x1)")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("density")
    elif marginal_kind == "theta":
        X_np = to_np(X_samples)
        theta = np.arctan2(X_np[:, 1], X_np[:, 0])  # [-pi, pi]
        bins = np.linspace(-np.pi, np.pi, 121)
        ax2.hist(theta, bins=bins, density=True, color=accent, alpha=0.35, label="samples")

        if theta_ref_curve is not None:
            theta_grid, pdf_theta = theta_ref_curve
            ax2.plot(theta_grid, pdf_theta, color=dark, linewidth=1.6, label="ref density")

        ax2.set_title(r"Marginal ($\theta$)")
        ax2.set_xlabel(r"$\theta$ (rad)")
        ax2.set_ylabel("density")
        ax2.set_xlim(-np.pi, np.pi)
        ax2.xaxis.set_major_locator(mticker.FixedLocator([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]))
        ax2.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    else:
        raise ValueError(f"Unknown marginal_kind={marginal_kind!r}")

    ax2.set_box_aspect(1)
    ax2.legend(loc="upper right")

    # Make tick density consistent across panels for a cleaner grid feel.
    for ax in (ax0, ax1):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.set_axisbelow(True)
    if marginal_kind != "theta":
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax2.set_axisbelow(True)

    out_dir = default_out_dir() if out_dir is None else out_dir
    out_path = out_dir / out_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
    return out_path

def create_trajectory_gif(
    *,
    distribution,
    conditioner,
    trace: dict,
    out_name: str,
    out_dir: Path | None = None,
    lim: float = 5.0,
    N: int = 150,
    manifold_kind: str | None = None,  # "circle", "line", "disk", "star"
    circle_r: float = 1.0,
    disk_r: float = 2.0,
    star_params: dict | None = None,  # {"R": 2.0, "n_points": 5, "amplitude": 0.35}
    marginal_kind: str = "theta",  # "theta" or "x1"
    ref_curve: tuple[np.ndarray, np.ndarray] | None = None,
    fps: int = 8,
    dpi: int = 150,
    trail_length: int = 5,  # Number of past frames to show as trail
    display_particles: int = 32,  # Number of particles to show in trajectory panels
) -> Path:
    """
    Create an animated GIF showing trajectory evolution in three panels:
      1. Unconstrained space (u-space) - selected particle trajectories
      2. Constrained space (x-space) - selected projected samples on density background
      3. Marginal density evolution over time (using ALL samples if available)
    
    Args:
        distribution: Target distribution for density background
        conditioner: Projection function u -> x
        trace: Dictionary from langevin_dynamics_with_trace containing 't', 'X_unc', 'X',
               and optionally 'X_all' for full samples (used for marginal)
        out_name: Output filename (should end in .gif)
        out_dir: Output directory (defaults to temp_scripts)
        lim: Axis limits for plotting
        N: Grid resolution for density background
        manifold_kind: Type of manifold ("circle", "line", "disk", "star")
        circle_r: Radius for circle manifold
        disk_r: Radius for disk manifold
        star_params: Parameters for star manifold
        marginal_kind: Type of marginal ("theta" for angular, "x1" for first coordinate)
        ref_curve: Reference curve for marginal (grid, pdf) tuple
        fps: Frames per second for GIF
        dpi: Resolution for GIF
        trail_length: Number of past positions to show as fading trail
        display_particles: Number of particles to display in trajectory panels
    
    Returns:
        Path to saved GIF
    """
    out_dir = default_out_dir() if out_dir is None else out_dir
    
    # Background density grid
    x = np.linspace(-lim, lim, N)
    Xg, Yg = np.meshgrid(x, x)
    XY = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32, device=DEVICE)
    P_grid = torch.exp(distribution.log_prob(XY)).reshape(N, N).detach().cpu().numpy()
    
    # Extract trace data for trajectory visualization (subset)
    U_trace = np.stack([u.numpy() for u in trace["X_unc"]], axis=0)  # (Trec, trace_batch, 2)
    X_trace = np.stack([x.numpy() for x in trace["X"]], axis=0)  # (Trec, trace_batch, 2)
    t_all = np.array(trace["t"])
    Trec, B_trace, _ = U_trace.shape
    
    # Use X_all for marginal if available (more samples = smoother histogram)
    has_all_samples = "X_all" in trace and len(trace["X_all"]) > 0
    if has_all_samples:
        X_marginal = np.stack([x.numpy() for x in trace["X_all"]], axis=0)  # (Trec, num_batch, 2)
        n_marginal_samples = X_marginal.shape[1]
        print(f"  Using {n_marginal_samples} samples for marginal density")
    else:
        X_marginal = X_trace  # Fall back to traced subset
        n_marginal_samples = B_trace
        print(f"  Using {n_marginal_samples} traced samples for marginal (no X_all available)")
    
    # Select particles to display (subset of traced particles)
    n_display = min(display_particles, B_trace)
    display_idx = np.linspace(0, B_trace - 1, n_display).astype(int)
    U_display = U_trace[:, display_idx, :]  # (Trec, n_display, 2)
    X_display = X_trace[:, display_idx, :]  # (Trec, n_display, 2)
    
    # Particle colors (consistent identity across frames)
    id_cmap = plt.get_cmap("tab20")
    id_colors = np.array([id_cmap((i % 20) / 20.0) for i in range(n_display)])
    
    # Set up figure
    fig = plt.figure(figsize=(14, 4.5), dpi=dpi, constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.15])
    ax_u = fig.add_subplot(gs[0, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_marg = fig.add_subplot(gs[0, 2])
    
    # Compute axis limits for u-space (use max across all frames)
    u_lim = float(np.max(np.abs(U_trace)) * 1.1 + 1e-9)
    
    # Draw manifold boundary helper
    def draw_manifold(ax, color='white', lw=1.2, alpha=0.9):
        if manifold_kind == "circle":
            th = np.linspace(-np.pi, np.pi, 400)
            ax.plot(circle_r * np.cos(th), circle_r * np.sin(th), color=color, lw=lw, alpha=alpha)
        elif manifold_kind == "line":
            xx = np.linspace(-lim, lim, 200)
            ax.plot(xx, -xx, color=color, lw=lw, alpha=alpha)
        elif manifold_kind == "disk":
            th = np.linspace(-np.pi, np.pi, 400)
            ax.plot(disk_r * np.cos(th), disk_r * np.sin(th), color=color, lw=lw, alpha=alpha, ls='--')
        elif manifold_kind == "star" and star_params is not None:
            th = np.linspace(-np.pi, np.pi, 500)
            R = star_params.get("R", 2.0)
            n_pts = star_params.get("n_points", 5)
            amp = star_params.get("amplitude", 0.35)
            r_star = R * (1 + amp * np.cos(n_pts * th))
            ax.plot(r_star * np.cos(th), r_star * np.sin(th), color=color, lw=lw, alpha=alpha, ls='--')
    
    # Marginal bins and reference
    if marginal_kind == "theta":
        bins = np.linspace(-np.pi, np.pi, 81)
        centers = 0.5 * (bins[:-1] + bins[1:])
    elif marginal_kind == "x1":
        if ref_curve is not None:
            grid_min, grid_max = ref_curve[0].min(), ref_curve[0].max()
        else:
            grid_min, grid_max = -lim, lim
        bins = np.linspace(grid_min, grid_max, 81)
        centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Animation update function
    def update(frame_idx):
        ax_u.clear()
        ax_x.clear()
        ax_marg.clear()
        
        t = t_all[frame_idx]
        U_t = U_display[frame_idx]  # (n_display, 2) - selected particles
        X_t = X_display[frame_idx]  # (n_display, 2) - selected particles
        X_t_marginal = X_marginal[frame_idx]  # (num_batch, 2) - all samples for marginal
        
        # --- Panel 1: Unconstrained space (u-space) ---
        # Draw trails (past positions fading)
        start_trail = max(0, frame_idx - trail_length)
        for trail_i in range(start_trail, frame_idx):
            alpha_trail = 0.15 + 0.3 * ((trail_i - start_trail) / max(1, frame_idx - start_trail))
            U_trail = U_display[trail_i]
            ax_u.scatter(U_trail[:, 0], U_trail[:, 1], s=8, c=id_colors, alpha=alpha_trail, linewidths=0.0)
        
        # Current positions
        ax_u.scatter(U_t[:, 0], U_t[:, 1], s=35, c=id_colors, alpha=0.95, linewidths=0.5, edgecolors='black')
        
        ax_u.set_xlim(-u_lim, u_lim)
        ax_u.set_ylim(-u_lim, u_lim)
        ax_u.set_aspect('equal', adjustable='box')
        ax_u.set_xlabel("u₁")
        ax_u.set_ylabel("u₂")
        ax_u.set_title(f"Unconstrained (u-space)  t={t}  [{n_display} particles]")
        ax_u.grid(True, alpha=0.3)
        
        # --- Panel 2: Constrained space (x-space) ---
        ax_x.contourf(Xg, Yg, P_grid, levels=32, cmap=plt.get_cmap("GnBu"), alpha=0.92)
        draw_manifold(ax_x)
        
        # Draw trails in x-space too
        for trail_i in range(start_trail, frame_idx):
            alpha_trail = 0.15 + 0.3 * ((trail_i - start_trail) / max(1, frame_idx - start_trail))
            X_trail = X_display[trail_i]
            ax_x.scatter(X_trail[:, 0], X_trail[:, 1], s=8, c=id_colors, alpha=alpha_trail, linewidths=0.0)
        
        # Current positions
        ax_x.scatter(X_t[:, 0], X_t[:, 1], s=35, c=id_colors, alpha=0.95, linewidths=0.5, edgecolors='white')
        
        ax_x.set_xlim(-lim, lim)
        ax_x.set_ylim(-lim, lim)
        ax_x.set_aspect('equal', adjustable='box')
        ax_x.set_xlabel("x₁")
        ax_x.set_ylabel("x₂")
        ax_x.set_title(f"Constrained (x-space)  t={t}")
        
        # --- Panel 3: Marginal density (using ALL samples) ---
        # Reference curve
        if ref_curve is not None:
            grid_ref, pdf_ref = ref_curve
            ax_marg.fill_between(grid_ref, pdf_ref, alpha=0.15, color='0.2', label='reference')
            ax_marg.plot(grid_ref, pdf_ref, color='0.2', lw=2.0, alpha=0.8)
        
        # Compute current marginal histogram from ALL samples
        if marginal_kind == "theta":
            vals = np.arctan2(X_t_marginal[:, 1], X_t_marginal[:, 0])
            ax_marg.set_xlim(-np.pi, np.pi)
            ax_marg.set_xlabel(r"$\theta$ (radians)")
            ax_marg.xaxis.set_major_locator(mticker.FixedLocator([-np.pi, -np.pi/2, 0, np.pi/2, np.pi]))
            ax_marg.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
        elif marginal_kind == "x1":
            vals = X_t_marginal[:, 0]
            ax_marg.set_xlim(bins[0], bins[-1])
            ax_marg.set_xlabel("x₁")
        
        hist, _ = np.histogram(vals, bins=bins, density=True)
        time_color = plt.get_cmap("viridis")(0.3 + 0.6 * (frame_idx / max(1, Trec - 1)))
        ax_marg.bar(centers, hist, width=centers[1] - centers[0], alpha=0.65, color=time_color, edgecolor='none', label=f't={t}')
        ax_marg.plot(centers, hist, color=time_color, lw=1.5, alpha=0.9)
        
        ax_marg.set_ylabel("density")
        ax_marg.set_title(f"Marginal (n={n_marginal_samples})  t={t}")
        ax_marg.legend(loc='upper right', fontsize=9)
        ax_marg.grid(True, alpha=0.3, axis='y')
        
        # Auto-scale y-axis for marginal
        if ref_curve is not None:
            y_max = max(np.max(pdf_ref) * 1.15, np.max(hist) * 1.15)
        else:
            y_max = np.max(hist) * 1.15 if np.max(hist) > 0 else 1.0
        ax_marg.set_ylim(0, y_max)
        
        return ax_u, ax_x, ax_marg
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=Trec, interval=1000 // fps, blit=False)
    
    # Save GIF
    out_path = out_dir / out_name
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF to {out_path}")
    return out_path


def plot_integration_projection_layout(
    *,
    distribution,
    conditioner,
    trace: dict,
    X_final: torch.Tensor,
    X_final_unc: torch.Tensor,
    out_name: str,
    out_dir: Path | None = None,
    lim: float = 5.0,
    N: int = 150,
    manifold_kind: str | None = None,  # e.g. "circle"
    circle_r: float = 1.0,
    connect_n: int = 80,
) -> Path:
    """
    2x3 layout:
      Top row: schematic flow + equations
      Bottom row: (u-space integration trace) → (projection mapping) → (x-space result on density)
    """
    out_dir = default_out_dir() if out_dir is None else out_dir

    # Background density grid (for the original target in x-space).
    x = np.linspace(-lim, lim, N)
    Xg, Yg = np.meshgrid(x, x)
    XY = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32, device=DEVICE)
    P_grid = torch.exp(distribution.log_prob(XY)).reshape(N, N).detach().cpu().numpy()

    fig = plt.figure(figsize=(12.0, 6.0), dpi=320, constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.55, 1.0], width_ratios=[1.0, 1.0, 1.0])

    ax_t0 = fig.add_subplot(gs[0, 0])
    ax_t1 = fig.add_subplot(gs[0, 1])
    ax_t2 = fig.add_subplot(gs[0, 2])
    ax_u = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[1, 2])

    # ---- Top row: schematic (layout-first) ----
    for ax in (ax_t0, ax_t1, ax_t2):
        ax.set_axis_off()

    ax_t0.text(
        0.02,
        0.65,
        "A. Integrate in unconstrained coords (u)\n"
        r"$u_{t+1} = u_t + \frac{\eta}{2}\nabla_u \log p(\Pi(u_t)) + \sqrt{\eta}\,\xi_t$",
        va="top",
        ha="left",
        fontsize=10.5,
    )
    ax_t1.text(
        0.02,
        0.65,
        "B. Projection / conditioning\n"
        r"$x_t = \Pi(u_t)$",
        va="top",
        ha="left",
        fontsize=10.5,
    )
    ax_t2.text(
        0.02,
        0.65,
        "C. Look at results on constrained space\n"
        r"$x_T$ (and its marginals on the manifold)",
        va="top",
        ha="left",
        fontsize=10.5,
    )
    # Flow arrows between top panels
    ax_t0.annotate("", xy=(1.02, 0.50), xytext=(0.86, 0.50), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="->", lw=1.2, color="0.2"))
    ax_t1.annotate("", xy=(1.02, 0.50), xytext=(0.86, 0.50), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="->", lw=1.2, color="0.2"))

    # ---- Bottom left: integration trace in u-space (identity-colored) ----
    U = np.stack([u.numpy() for u in trace["X_unc"]], axis=0)  # (Trec, B, 2)
    B = U.shape[1]
    id_cmap = plt.get_cmap("tab20")
    id_colors = np.stack([id_cmap((i % 20) / 20.0) for i in range(B)], axis=0)

    for b in range(B):
        ax_u.plot(U[:, b, 0], U[:, b, 1], color=id_colors[b], lw=0.9, alpha=0.35)
    U0 = U[0]
    UT = U[-1]
    ax_u.scatter(U0[:, 0], U0[:, 1], s=18, c=id_colors, alpha=0.85, linewidths=0.0, label="u(0)")
    ax_u.scatter(UT[:, 0], UT[:, 1], s=28, c=id_colors, alpha=0.95, linewidths=0.0, label="u(T)")
    ax_u.set_title("A. Integrate in u-space (trace subset, identity-colored)")
    ax_u.set_xlabel("u1")
    ax_u.set_ylabel("u2")
    ax_u.legend(loc="upper right", fontsize=9)
    u_lim = float(np.max(np.abs(X_final_unc.detach().cpu().numpy())) * 1.05 + 1e-9)
    ax_u.set_xlim(-u_lim, u_lim)
    ax_u.set_ylim(-u_lim, u_lim)
    ax_u.set_aspect("equal", adjustable="box")
    ax_u.set_box_aspect(1)

    # ---- Bottom middle: projection mapping (connect a few points) ----
    # Use the traced subset for a consistent mapping view.
    X_T = trace["X"][-1].numpy()
    for b in range(B):
        ax_map.plot([UT[b, 0], X_T[b, 0]], [UT[b, 1], X_T[b, 1]], color=id_colors[b], alpha=0.22, lw=1.0)
    ax_map.scatter(UT[:, 0], UT[:, 1], s=22, c=id_colors, alpha=0.85, linewidths=0.0, label="u(T)")
    ax_map.scatter(X_T[:, 0], X_T[:, 1], s=26, c=id_colors, alpha=0.95, linewidths=0.0, label=r"$x(T)=\Pi(u(T))$")
    ax_map.set_title("B. Projection step (same particles)")
    ax_map.set_xlabel("coord 1")
    ax_map.set_ylabel("coord 2")
    ax_map.legend(loc="upper right", fontsize=9)
    lim_map = float(max(np.max(np.abs(UT)), np.max(np.abs(X_T))) * 1.15 + 1e-9)
    ax_map.set_xlim(-lim_map, lim_map)
    ax_map.set_ylim(-lim_map, lim_map)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_box_aspect(1)

    # ---- Bottom right: resulting x samples on density background ----
    X_np = X_final.detach().cpu().numpy()
    cf = ax_x.contourf(Xg, Yg, P_grid, levels=40, cmap=plt.get_cmap("GnBu"), alpha=0.92)
    ax_x.scatter(X_np[:, 0], X_np[:, 1], c="black", s=0.6, linewidths=0, alpha=0.85)
    if manifold_kind == "circle":
        th = np.linspace(-np.pi, np.pi, 400)
        ax_x.plot(circle_r * np.cos(th), circle_r * np.sin(th), color="white", lw=1.2, alpha=0.9)
    ax_x.set_title("C. Result in x-space (projected)")
    ax_x.set_xlabel("x1")
    ax_x.set_ylabel("x2")
    ax_x.set_xlim(-lim, lim)
    ax_x.set_ylim(-lim, lim)
    ax_x.set_aspect("equal", adjustable="box")
    ax_x.set_box_aspect(1)

    # Small inset colorbar inside the x-panel (doesn't change layout).
    cax_in = ax_x.inset_axes([0.14, 0.90, 0.72, 0.04])
    cbx = fig.colorbar(cf, cax=cax_in, orientation="horizontal")
    cbx.ax.xaxis.set_ticks_position("top")
    cbx.ax.xaxis.set_label_position("top")
    cbx.set_label("density", labelpad=1.5, fontsize=10)
    cbx.ax.xaxis.set_major_locator(mticker.MaxNLocator(4))

    # Ticks / grid feel consistency
    for ax in (ax_u, ax_map, ax_x):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.set_axisbelow(True)

    out_path = out_dir / out_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Langevin sampling + projection visualizations.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a smaller/shorter simulation to quickly validate plots.",
    )
    args = parser.parse_args()

    gmm = build_distribution()
    print(gmm.sample((10,)))

    # Langevin sampling controls
    if args.fast:
        NUM_STEPS = 600
        NUM_BATCH = 3000
        STEP_SIZE = 0.015
        TRACE_BATCH = 96
        TRACE_EVERY = 30
    else:
        NUM_STEPS = 3000  # was 1000
        NUM_BATCH = 10000
        STEP_SIZE = 0.01
        TRACE_BATCH = 96
        TRACE_EVERY = 60

    # ---- Plot styling: monospace + clean grid layout ----
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 8.0,
            "axes.labelpad": 4.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )

    # --- Run 1: linear projection + x1 marginal ---
    X_samples, X_samples_unc = langevin_dynamics(
        gmm,
        conditioner=linear_projec,
        num_steps=NUM_STEPS,
        num_batch=NUM_BATCH,
        step_size=STEP_SIZE,
    )
    _, ix = torch.sort(X_samples, dim=0, descending=False)
    X_samples = X_samples[ix[:, 0], :]
    out_dir = default_out_dir()
    plot_triptych(
        X_samples=X_samples,
        X_samples_unc=X_samples_unc,
        out_name="linear.png",
        out_dir=out_dir,
        marginal_kind="x1",
    )

    # Layout figure (integration + projection)
    X_final, X_final_unc, trace = langevin_dynamics_with_trace(
        gmm,
        conditioner=linear_projec,
        num_steps=NUM_STEPS,
        num_batch=NUM_BATCH,
        step_size=STEP_SIZE,
        trace_batch=TRACE_BATCH,
        trace_every=TRACE_EVERY,
    )
    plot_integration_projection_layout(
        distribution=gmm,
        conditioner=linear_projec,
        trace=trace,
        X_final=X_final,
        X_final_unc=X_final_unc,
        out_name="linear_layout.png",
        out_dir=out_dir,
        manifold_kind=None,
    )

    # Reference x1 density along the line manifold x2=-x1:
    # pdf(x1) ∝ p(x1, -x1), normalized over x1.
    x1_grid = np.linspace(-5.0, 5.0, 2048, dtype=np.float32)
    line = np.stack([x1_grid, -x1_grid], axis=1)
    line_t = torch.tensor(line, device=DEVICE)
    pdf_unnorm = torch.exp(gmm.log_prob(line_t)).detach().cpu().numpy()
    Z = np.trapz(pdf_unnorm, x=x1_grid)
    pdf_x1 = pdf_unnorm / (Z + 1e-12)
    plot_algorithm_flow_marginal_recovery(
        distribution=gmm,
        conditioner=linear_projec,
        trace=trace,
        X_final=X_final,
        X_final_unc=X_final_unc,
        out_name="linear_marginal.png",
        out_dir=out_dir,
        manifold_kind="line",
        marginal_kind="x1",
        ref_curve=(x1_grid, pdf_x1),
    )
    plot_algorithm_overview_single_figure(
        distribution=gmm,
        conditioner=linear_projec,
        trace=trace,
        X_final=X_final,
        X_final_unc=X_final_unc,
        out_name="linear_overview.png",
        out_dir=out_dir,
        manifold_kind="line",
        marginal_kind="x1",
        ref_curve=(x1_grid, pdf_x1),
    )
    # Create trajectory GIF for linear projection (10x longer sampling)
    print("Creating trajectory GIF for linear projection (10x longer)...")
    GIF_STEPS_LINEAR = NUM_STEPS * 10  # 10x longer
    GIF_TRACE_EVERY_LINEAR = TRACE_EVERY * 10  # Keep ~same number of frames
    GIF_BATCH_LINEAR = NUM_BATCH * 2  # More samples for smoother marginal
    _, _, trace_gif_linear = langevin_dynamics_with_trace(
        gmm,
        conditioner=linear_projec,
        num_steps=GIF_STEPS_LINEAR,
        num_batch=GIF_BATCH_LINEAR,
        step_size=STEP_SIZE,
        trace_batch=TRACE_BATCH,
        trace_every=GIF_TRACE_EVERY_LINEAR,
        record_all_for_marginal=True,  # Record all samples for smooth marginal
    )
    create_trajectory_gif(
        distribution=gmm,
        conditioner=linear_projec,
        trace=trace_gif_linear,
        out_name="linear_trajectory.gif",
        out_dir=out_dir,
        manifold_kind="line",
        marginal_kind="x1",
        ref_curve=(x1_grid, pdf_x1),
        fps=6,
        trail_length=4,
        display_particles=32,
    )

    # --- Run 2: map-to-circle projection + theta marginal ---
    X_samples_c, X_samples_unc_c = langevin_dynamics(
        gmm,
        conditioner=map_to_unit_circle,
        num_steps=NUM_STEPS,
        num_batch=NUM_BATCH,
        step_size=STEP_SIZE,
    )

    # Reference theta density on the radius-r circle:
    # pdf_theta(θ) ∝ p( r cosθ, r sinθ ), normalized over θ ∈ [-π, π].
    # (Since arc length is r dθ and r is constant, normalization over θ is enough.)
    r = 1.0
    theta_grid = np.linspace(-np.pi, np.pi, 2048)
    circle = np.stack([r * np.cos(theta_grid), r * np.sin(theta_grid)], axis=1).astype(np.float32)
    circle_t = torch.tensor(circle, device=DEVICE)
    pdf_unnorm = torch.exp(gmm.log_prob(circle_t)).detach().cpu().numpy()
    Z = np.trapz(pdf_unnorm, x=theta_grid)
    pdf_theta = pdf_unnorm / (Z + 1e-12)

    plot_triptych(
        X_samples=X_samples_c,
        X_samples_unc=X_samples_unc_c,
        out_name="circle.png",
        out_dir=out_dir,
        marginal_kind="theta",
        theta_ref_curve=(theta_grid, pdf_theta),
    )

    # Layout figure (integration + projection) for circle projection
    X_final_c, X_final_unc_c, trace_c = langevin_dynamics_with_trace(
        gmm,
        conditioner=map_to_unit_circle,
        num_steps=NUM_STEPS,
        num_batch=NUM_BATCH,
        step_size=STEP_SIZE,
        trace_batch=TRACE_BATCH,
        trace_every=TRACE_EVERY,
    )
    plot_integration_projection_layout(
        distribution=gmm,
        conditioner=map_to_unit_circle,
        trace=trace_c,
        X_final=X_final_c,
        X_final_unc=X_final_unc_c,
        out_name="circle_layout.png",
        out_dir=out_dir,
        manifold_kind="circle",
        circle_r=1.0,
    )

    plot_algorithm_flow_marginal_recovery(
        distribution=gmm,
        conditioner=map_to_unit_circle,
        trace=trace_c,
        X_final=X_final_c,
        X_final_unc=X_final_unc_c,
        out_name="circle_marginal.png",
        out_dir=out_dir,
        manifold_kind="circle",
        circle_r=1.0,
        marginal_kind="theta",
        ref_curve=(theta_grid, pdf_theta),
    )
    plot_algorithm_overview_single_figure(
        distribution=gmm,
        conditioner=map_to_unit_circle,
        trace=trace_c,
        X_final=X_final_c,
        X_final_unc=X_final_unc_c,
        out_name="circle_overview.png",
        out_dir=out_dir,
        manifold_kind="circle",
        circle_r=1.0,
        marginal_kind="theta",
        ref_curve=(theta_grid, pdf_theta),
    )
    # Create trajectory GIF for circle projection (10x longer sampling)
    print("Creating trajectory GIF for circle projection (10x longer)...")
    GIF_STEPS_CIRCLE = NUM_STEPS * 10  # 10x longer
    GIF_TRACE_EVERY_CIRCLE = TRACE_EVERY * 10  # Keep ~same number of frames
    GIF_BATCH_CIRCLE = NUM_BATCH * 2  # More samples for smoother marginal
    _, _, trace_gif_circle = langevin_dynamics_with_trace(
        gmm,
        conditioner=map_to_unit_circle,
        num_steps=GIF_STEPS_CIRCLE,
        num_batch=GIF_BATCH_CIRCLE,
        step_size=STEP_SIZE,
        trace_batch=TRACE_BATCH,
        trace_every=GIF_TRACE_EVERY_CIRCLE,
        record_all_for_marginal=True,  # Record all samples for smooth marginal
    )
    create_trajectory_gif(
        distribution=gmm,
        conditioner=map_to_unit_circle,
        trace=trace_gif_circle,
        out_name="circle_trajectory.gif",
        out_dir=out_dir,
        manifold_kind="circle",
        circle_r=1.0,
        marginal_kind="theta",
        ref_curve=(theta_grid, pdf_theta),
        fps=6,
        trail_length=4,
        display_particles=32,
    )

    # --- Run 3: algebraic_disk bijection with volume correction ---
    # This demonstrates sampling on a disk using a smooth bijective map
    # with the Jacobian correction term for proper density estimation.
    print("\n" + "=" * 60)
    print("Running algebraic_disk bijection demo with volume correction...")
    print("=" * 60)

    DISK_R = 2.0
    
    # Choose projection type: "algebraic", "tanh", or "sigmoid"
    PROJECTION_TYPE = "tanh"
    PROJECTION_SCALE = 1.0  # For tanh/sigmoid: controls steepness
    
    # Create a curried version for the conditioner
    if PROJECTION_TYPE == "algebraic":
        def disk_conditioner(x):
            return algebraic_disk(x, R=DISK_R)
        proj_name = "algebraic"
    elif PROJECTION_TYPE == "tanh":
        def disk_conditioner(x):
            return tanh_disk(x, R=DISK_R, scale=PROJECTION_SCALE)
        proj_name = f"tanh (scale={PROJECTION_SCALE})"
    elif PROJECTION_TYPE == "sigmoid":
        def disk_conditioner(x):
            return sigmoid_disk(x, R=DISK_R, scale=PROJECTION_SCALE)
        proj_name = f"sigmoid (scale={PROJECTION_SCALE})"
    else:
        raise ValueError(f"Unknown projection type: {PROJECTION_TYPE}")
    
    print(f"Using {proj_name} projection to disk R={DISK_R}")

    # Create a curried grad_correction for the disk (use fast 2D version)
    def disk_grad_correction(x):
        return grad_correction_2d(disk_conditioner, x)

    # Run Langevin dynamics - compare WITH and WITHOUT volume correction
    DISK_STEPS = NUM_STEPS * 3  # 3x longer for better samples
    DISK_BATCH = NUM_BATCH * 5
    DISK_STEP_SIZE = STEP_SIZE * 0.1
    
    print(f"Running {DISK_STEPS} steps for disk sampling...")
    
    # WITH volume correction
    print("  [1/2] With grad log|det J| correction...")
    X_with_corr, X_unc_with_corr = langevin_dynamics(
        gmm,
        conditioner=disk_conditioner,
        num_steps=DISK_STEPS,
        num_batch=DISK_BATCH,
        step_size=DISK_STEP_SIZE,
        grad_correction=disk_grad_correction,
    )
    
    # WITHOUT volume correction
    print("  [2/2] unbiased (naive)...")
    X_no_corr, X_unc_no_corr = langevin_dynamics(
        gmm,
        conditioner=disk_conditioner,
        num_steps=DISK_STEPS,
        num_batch=DISK_BATCH,
        step_size=DISK_STEP_SIZE,
        grad_correction=None,  # No correction!
    )

    # Side-by-side comparison visualization
    from scipy.stats import gaussian_kde
    
    # Compute log|det J| field and gradient on a grid
    print("Computing log|det J| field for disk...")
    n_field = 30  # Coarser grid for vector field
    x_field = np.linspace(-3, 3, n_field)
    Xf, Yf = np.meshgrid(x_field, x_field)
    xy_field = torch.tensor(np.stack([Xf.ravel(), Yf.ravel()], axis=1), dtype=torch.float32)
    
    dvol_disk, grad_dvol_disk = grad_correction_2d(disk_conditioner, xy_field)
    logdetJ_disk = dvol_disk.numpy().reshape(n_field, n_field)
    grad_u_disk = grad_dvol_disk[:, 0].numpy().reshape(n_field, n_field)
    grad_v_disk = grad_dvol_disk[:, 1].numpy().reshape(n_field, n_field)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300, constrained_layout=True)

    # Grid for density plots
    x_plot = np.linspace(-3, 3, 150)
    Xg, Yg = np.meshgrid(x_plot, x_plot)
    XY_plot = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32, device=DEVICE)
    P_grid = torch.exp(gmm.log_prob(XY_plot)).reshape(150, 150).detach().cpu().numpy()
    
    th = np.linspace(-np.pi, np.pi, 400)
    
    # --- Top row: volume corrected ---
    X_with_np = X_with_corr.detach().cpu().numpy()
    
    ax = axes[0, 0]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.95)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("Target Density", fontsize=11)
    ax.set_ylabel("volume corrected", fontsize=12, fontweight='bold')
    
    ax = axes[0, 1]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.9)
    ax.scatter(X_with_np[:, 0], X_with_np[:, 1], c="black", s=0.6, alpha=0.7)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("Samples", fontsize=11)
    
    ax = axes[0, 2]
    print("Computing KDE (volume corrected)...")
    kde_with = gaussian_kde(X_with_np.T, bw_method=0.15)
    KDE_with = kde_with(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(150, 150)
    ax.contourf(Xg, Yg, KDE_with, levels=40, cmap="GnBu", alpha=0.95)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("KDE of Samples", fontsize=11)
    
    # Log|det J| with gradient field (top right)
    ax = axes[0, 3]
    vmax = np.max(np.abs(logdetJ_disk))
    cf = ax.contourf(Xf, Yf, logdetJ_disk, levels=30, cmap="RdBu_r", alpha=0.9, vmin=-vmax, vmax=vmax)
    ax.quiver(Xf, Yf, grad_u_disk, grad_v_disk, color='black', alpha=0.7, scale=30)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'k--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("log|det J| + ∇log|det J|", fontsize=11)
    plt.colorbar(cf, ax=ax, shrink=0.8)
    
    # --- Bottom row: unbiased ---
    X_no_np = X_no_corr.detach().cpu().numpy()
    
    ax = axes[1, 0]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.95)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_ylabel("unbiased", fontsize=12, fontweight='bold', color='darkred')
    ax.set_xlabel('x₁')
    
    ax = axes[1, 1]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.9)
    ax.scatter(X_no_np[:, 0], X_no_np[:, 1], c="darkred", s=0.6, alpha=0.7)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁')
    
    ax = axes[1, 2]
    print("Computing KDE (unbiased)...")
    kde_no = gaussian_kde(X_no_np.T, bw_method=0.15)
    KDE_no = kde_no(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(150, 150)
    ax.contourf(Xg, Yg, KDE_no, levels=40, cmap="OrRd", alpha=0.95)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁')
    
    # Empty or duplicate info for bottom right (show the correction term being ignored)
    ax = axes[1, 3]
    ax.contourf(Xf, Yf, logdetJ_disk, levels=30, cmap="RdBu_r", alpha=0.3, vmin=-vmax, vmax=vmax)
    ax.plot(DISK_R * np.cos(th), DISK_R * np.sin(th), 'k--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("(ignored)", fontsize=11, color='darkred')
    ax.set_xlabel('x₁')
    ax.text(0, 0, "unbiased", ha='center', va='center', fontsize=16, color='darkred', fontweight='bold', alpha=0.7)
    
    fig.suptitle(f"Disk Sampling: Effect of ∇log|det J| Correction ({proj_name})", fontsize=13, fontweight='bold')

    # Save to project manifold directory
    manifold_dir = Path(__file__).parent
    out_path = manifold_dir / "disk_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")

    # Create trajectory GIF for disk sampling (unbiased version to show evolution)
    # 10x longer sampling, proportionally larger trace_every to keep frame count manageable
    print("Creating trajectory GIF for disk sampling (10x longer)...")
    DISK_GIF_STEPS = DISK_STEPS * 10  # 10x longer (was min(DISK_STEPS, 2000))
    DISK_GIF_BATCH = 20000  # Many samples for smooth marginal
    DISK_GIF_TRACE_EVERY = 400  # 10x larger to keep ~same frame count
    _, _, trace_disk = langevin_dynamics_with_trace(
        gmm,
        conditioner=disk_conditioner,
        num_steps=DISK_GIF_STEPS,
        num_batch=DISK_GIF_BATCH,
        step_size=DISK_STEP_SIZE,
        trace_batch=96,
        trace_every=DISK_GIF_TRACE_EVERY,
        record_all_for_marginal=True,  # Record all samples for smooth marginal
    )
    # Reference for disk: radial marginal at angle theta=0 (just use x1 marginal inside disk)
    x1_disk_grid = np.linspace(-DISK_R * 0.99, DISK_R * 0.99, 512, dtype=np.float32)
    # Approximate: sample along a horizontal line through disk center
    line_disk = np.stack([x1_disk_grid, np.zeros_like(x1_disk_grid)], axis=1)
    line_disk_t = torch.tensor(line_disk, device=DEVICE)
    pdf_disk_unnorm = torch.exp(gmm.log_prob(line_disk_t)).detach().cpu().numpy()
    Z_disk = np.trapz(pdf_disk_unnorm, x=x1_disk_grid)
    pdf_disk_x1 = pdf_disk_unnorm / (Z_disk + 1e-12)
    
    create_trajectory_gif(
        distribution=gmm,
        conditioner=disk_conditioner,
        trace=trace_disk,
        out_name="disk_trajectory.gif",
        out_dir=out_dir,
        manifold_kind="disk",
        disk_r=DISK_R,
        marginal_kind="x1",
        ref_curve=(x1_disk_grid, pdf_disk_x1),
        fps=8,
        trail_length=5,
        lim=3.5,
        display_particles=32,
    )

    # --- Run 4: Star-shaped manifold with volume correction ---
    print("\n" + "=" * 60)
    print("Running star-shaped manifold demo with volume correction...")
    print("=" * 60)

    STAR_R = 2.0
    STAR_N_POINTS = 5
    STAR_AMPLITUDE = 0.35
    
    def star_conditioner(x):
        return star_shape(x, R=STAR_R, n_points=STAR_N_POINTS, amplitude=STAR_AMPLITUDE)
    
    def star_grad_correction(x):
        return grad_correction_2d(star_conditioner, x)
    
    STAR_STEPS = NUM_STEPS * 3
    STAR_BATCH = NUM_BATCH * 5
    STAR_STEP_SIZE = STEP_SIZE * 0.1
    
    print(f"Star shape: R={STAR_R}, {STAR_N_POINTS} points, amplitude={STAR_AMPLITUDE}")
    print(f"Running {STAR_STEPS} steps...")
    
    # WITH volume correction
    print("  [1/2] With grad log|det J| correction...")
    X_star_with, _ = langevin_dynamics(
        gmm,
        conditioner=star_conditioner,
        num_steps=STAR_STEPS,
        num_batch=STAR_BATCH,
        step_size=STAR_STEP_SIZE,
        grad_correction=star_grad_correction,
    )
    
    # WITHOUT volume correction
    print("  [2/2] unbiased (naive)...")
    X_star_no, _ = langevin_dynamics(
        gmm,
        conditioner=star_conditioner,
        num_steps=STAR_STEPS,
        num_batch=STAR_BATCH,
        step_size=STAR_STEP_SIZE,
        grad_correction=None,
    )

    # Compute log|det J| field and gradient for star
    print("Computing log|det J| field for star...")
    dvol_star, grad_dvol_star = grad_correction_2d(star_conditioner, xy_field)
    logdetJ_star = dvol_star.numpy().reshape(n_field, n_field)
    grad_u_star = grad_dvol_star[:, 0].numpy().reshape(n_field, n_field)
    grad_v_star = grad_dvol_star[:, 1].numpy().reshape(n_field, n_field)
    
    # Side-by-side comparison for star
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300, constrained_layout=True)
    
    # Star boundary
    th_star = np.linspace(-np.pi, np.pi, 500)
    r_star = STAR_R * (1 + STAR_AMPLITUDE * np.cos(STAR_N_POINTS * th_star))
    
    # --- Top row: volume corrected ---
    X_star_with_np = X_star_with.detach().cpu().numpy()
    
    ax = axes[0, 0]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.95)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("Target Density", fontsize=11)
    ax.set_ylabel("volume corrected", fontsize=12, fontweight='bold')
    
    ax = axes[0, 1]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.9)
    ax.scatter(X_star_with_np[:, 0], X_star_with_np[:, 1], c="black", s=0.6, alpha=0.7)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("Samples", fontsize=11)
    
    ax = axes[0, 2]
    print("Computing KDE (star, volume corrected)...")
    kde_star_with = gaussian_kde(X_star_with_np.T, bw_method=0.15)
    KDE_star_with = kde_star_with(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(150, 150)
    ax.contourf(Xg, Yg, KDE_star_with, levels=40, cmap="GnBu", alpha=0.95)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("KDE of Samples", fontsize=11)
    
    # Log|det J| with gradient field (top right)
    ax = axes[0, 3]
    vmax_star = np.max(np.abs(logdetJ_star))
    cf_star = ax.contourf(Xf, Yf, logdetJ_star, levels=30, cmap="RdBu_r", alpha=0.9, vmin=-vmax_star, vmax=vmax_star)
    ax.quiver(Xf, Yf, grad_u_star, grad_v_star, color='black', alpha=0.7, scale=30)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'k--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("log|det J| + ∇log|det J|", fontsize=11)
    plt.colorbar(cf_star, ax=ax, shrink=0.8)
    
    # --- Bottom row: unbiased ---
    X_star_no_np = X_star_no.detach().cpu().numpy()
    
    ax = axes[1, 0]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.95)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_ylabel("unbiased", fontsize=12, fontweight='bold', color='darkred')
    ax.set_xlabel('x₁')
    
    ax = axes[1, 1]
    ax.contourf(Xg, Yg, P_grid, levels=40, cmap="GnBu", alpha=0.9)
    ax.scatter(X_star_no_np[:, 0], X_star_no_np[:, 1], c="darkred", s=0.6, alpha=0.7)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁')
    
    ax = axes[1, 2]
    print("Computing KDE (star, unbiased)...")
    kde_star_no = gaussian_kde(X_star_no_np.T, bw_method=0.15)
    KDE_star_no = kde_star_no(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(150, 150)
    ax.contourf(Xg, Yg, KDE_star_no, levels=40, cmap="OrRd", alpha=0.95)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'w--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁')
    
    # Show ignored correction (bottom right)
    ax = axes[1, 3]
    ax.contourf(Xf, Yf, logdetJ_star, levels=30, cmap="RdBu_r", alpha=0.3, vmin=-vmax_star, vmax=vmax_star)
    ax.plot(r_star * np.cos(th_star), r_star * np.sin(th_star), 'k--', lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("(ignored)", fontsize=11, color='darkred')
    ax.set_xlabel('x₁')
    ax.text(0, 0, "unbiased", ha='center', va='center', fontsize=12, color='darkred', fontweight='bold', alpha=0.7)
    
    fig.suptitle(f"Star Sampling: Effect of ∇log|det J| Correction ({STAR_N_POINTS}-point star)", fontsize=13, fontweight='bold')

    # Save to project manifold directory
    out_path = manifold_dir / "star_shape.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")

    # Create trajectory GIF for star sampling (10x longer)
    print("Creating trajectory GIF for star sampling (10x longer)...")
    STAR_GIF_STEPS = STAR_STEPS * 10  # 10x longer (was min(STAR_STEPS, 2000))
    STAR_GIF_BATCH = 20000  # Many samples for smooth marginal
    STAR_GIF_TRACE_EVERY = 400  # 10x larger to keep ~same frame count
    _, _, trace_star = langevin_dynamics_with_trace(
        gmm,
        conditioner=star_conditioner,
        num_steps=STAR_GIF_STEPS,
        num_batch=STAR_GIF_BATCH,
        step_size=STAR_STEP_SIZE,
        trace_batch=96,
        trace_every=STAR_GIF_TRACE_EVERY,
        record_all_for_marginal=True,  # Record all samples for smooth marginal
    )
    # Reference for star: similar to disk, use x1 marginal
    x1_star_grid = np.linspace(-STAR_R * 1.3 * 0.99, STAR_R * 1.3 * 0.99, 512, dtype=np.float32)
    line_star = np.stack([x1_star_grid, np.zeros_like(x1_star_grid)], axis=1)
    line_star_t = torch.tensor(line_star, device=DEVICE)
    pdf_star_unnorm = torch.exp(gmm.log_prob(line_star_t)).detach().cpu().numpy()
    Z_star = np.trapz(pdf_star_unnorm, x=x1_star_grid)
    pdf_star_x1 = pdf_star_unnorm / (Z_star + 1e-12)
    
    create_trajectory_gif(
        distribution=gmm,
        conditioner=star_conditioner,
        trace=trace_star,
        out_name="star_trajectory.gif",
        out_dir=out_dir,
        manifold_kind="star",
        star_params={"R": STAR_R, "n_points": STAR_N_POINTS, "amplitude": STAR_AMPLITUDE},
        marginal_kind="x1",
        ref_curve=(x1_star_grid, pdf_star_x1),
        fps=8,
        trail_length=5,
        lim=3.5,
        display_particles=32,
    )

    print("\n✓ All demos completed.")
    print(f"  Comparison plots saved to: {manifold_dir}")
    print(f"  Other plots saved to: {out_dir}")
