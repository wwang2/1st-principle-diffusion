"""
Unified visualization script for Force Matching (Data Score) vs DSM (Noise Score).
Consolidates logic from:
- dsm_vs_force.py (Intuition, vector fields)
- data_score_bias_variance.py (Bias/Variance analysis)
- data_vs_noise_score.py (Variance comparison)

Outputs all relevant plots and animations to ~/temp_scripts/1st-principle-diffusion/force/
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from pathlib import Path
import shutil
import subprocess

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cpu"  # CPU is sufficient for these 2D demos and often easier for plotting

def default_out_dir() -> Path:
    out = Path(__file__).resolve().parent.parent / "assets" / "force"
    out.mkdir(parents=True, exist_ok=True)
    return out

# =============================================================================
# Core Logic: GMM & Diffusion Utilities
# =============================================================================

def alpha(t): return torch.sqrt(1 - t)
def sigma(t): return torch.sqrt(t)

class GMM:
    """Vectorized 2D Gaussian Mixture Model."""
    def __init__(self, means, covs, weights=None, device=DEVICE):
        self.device = device
        self.K = len(means)
        self.means = torch.tensor(means, dtype=torch.float32, device=device)
        self.covs = torch.tensor(covs, dtype=torch.float32, device=device)
        self.weights = torch.tensor(weights if weights else [1/self.K]*self.K, device=device)
        self.weights = self.weights / self.weights.sum()
        self.precs = torch.linalg.inv(self.covs)
        self.log_dets = torch.linalg.slogdet(self.covs)[1]
    
    def log_prob(self, x):
        """Log probability density log p(x)."""
        # x: [B, 2] -> diff: [B, K, 2]
        diff = x[:, None, :] - self.means[None, :, :]
        mahal = torch.einsum('bkd,kde,bke->bk', diff, self.precs, diff)
        log_comp = -0.5 * mahal - 0.5 * self.log_dets[None, :] - np.log(2*np.pi)
        log_comp = log_comp + torch.log(self.weights)[None, :]
        return torch.logsumexp(log_comp, dim=1)
    
    def score(self, x):
        """Score function ∇_x log p(x)."""
        diff = x[:, None, :] - self.means[None, :, :]
        mahal = torch.einsum('bkd,kde,bke->bk', diff, self.precs, diff)
        log_resp = -0.5 * mahal - 0.5 * self.log_dets[None, :] + torch.log(self.weights)[None, :]
        resp = torch.softmax(log_resp, dim=1)
        score_k = -torch.einsum('kde,bke->bkd', self.precs, diff)
        return torch.einsum('bk,bkd->bd', resp, score_k)
    
    def sample(self, n):
        """Sample x ~ p(x)."""
        idx = torch.multinomial(self.weights, n, replacement=True)
        L = torch.linalg.cholesky(self.covs)
        z = torch.randn(n, 2, device=self.device)
        return self.means[idx] + torch.einsum('kde,be->bd', L[idx], z)

def noised_gmm_density(XX, YY, t, gmm):
    """Compute marginal density p_t(x_t) on a grid (XX, YY)."""
    t_tensor = torch.as_tensor(t, device=gmm.device).float()
    a, s = alpha(t_tensor), sigma(t_tensor)
    
    noised_means = a * gmm.means
    noised_covs = (a**2) * gmm.covs + (s**2) * torch.eye(2, device=gmm.device)
    noised_precs = torch.linalg.inv(noised_covs)
    noised_log_dets = torch.linalg.slogdet(noised_covs)[1]
    
    XY = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32, device=gmm.device)
    diff = XY[:, None, :] - noised_means[None, :, :]
    mahal = torch.einsum('bkd,kde,bke->bk', diff, noised_precs, diff)
    log_comp = -0.5 * mahal - 0.5 * noised_log_dets[None, :] - np.log(2*np.pi)
    log_comp = log_comp + torch.log(gmm.weights)[None, :]
    log_p = torch.logsumexp(log_comp, dim=1)
    return torch.exp(log_p).reshape(XX.shape).cpu().numpy()

def noised_gmm_score(x_t, t, gmm):
    """Compute exact marginal score ∇_x_t log p_t(x_t)."""
    t_tensor = torch.as_tensor(t, device=gmm.device).float()
    a, s = alpha(t_tensor), sigma(t_tensor)
    
    noised_means = a * gmm.means
    noised_covs = (a**2) * gmm.covs + (s**2) * torch.eye(2, device=gmm.device)
    noised_precs = torch.linalg.inv(noised_covs)
    noised_log_dets = torch.linalg.slogdet(noised_covs)[1]
    
    diff = x_t[:, None, :] - noised_means[None, :, :]
    mahal = torch.einsum('bkd,kde,bke->bk', diff, noised_precs, diff)
    log_resp = -0.5 * mahal - 0.5 * noised_log_dets[None, :] + torch.log(gmm.weights)[None, :]
    resp = torch.softmax(log_resp, dim=1)
    score_k = -torch.einsum('kde,bke->bkd', noised_precs, diff)
    return torch.einsum('bk,bkd->bd', resp, score_k)

def sample_posterior(x_t, t, gmm, n_samples=1):
    """Sample x_0 ~ p(x_0 | x_t). Returns [B, n_samples, 2] or [n_samples, 2] if B=1 squeezed."""
    t_tensor = torch.as_tensor(t, device=gmm.device).float()
    a, s = alpha(t_tensor), sigma(t_tensor)
    
    x_t = x_t if x_t.ndim == 2 else x_t.unsqueeze(0) # Ensure [B, 2]
    B = x_t.shape[0]
    
    # Posterior Precision & Covariance (same for all x_t)
    lam_lik = (a**2) / (s**2)
    lam_post = gmm.precs + lam_lik * torch.eye(2, device=gmm.device)
    sig_post = torch.linalg.inv(lam_post)
    L_post = torch.linalg.cholesky(sig_post)
    
    # Posterior Means (depend on x_t)
    term1 = torch.einsum('kde,ke->kd', gmm.precs, gmm.means)
    term2 = (a / s**2) * x_t # [B, 2]
    # [K, 2, 2] @ ([K, 2] + [B, 2]) -> need broadcast
    # term1: [1, K, 2], term2: [B, 1, 2]
    mu_post = torch.einsum('kde,bke->bkd', sig_post, term1[None, :, :] + term2[:, None, :])
    
    # Posterior Weights
    noised_means = a * gmm.means
    noised_covs = (a**2) * gmm.covs + (s**2) * torch.eye(2, device=gmm.device)
    noised_precs = torch.linalg.inv(noised_covs)
    noised_log_dets = torch.linalg.slogdet(noised_covs)[1]
    
    diff = x_t[:, None, :] - noised_means[None, :, :]
    mahal = torch.einsum('bkd,kde,bke->bk', diff, noised_precs, diff)
    log_w = -0.5 * mahal - 0.5 * noised_log_dets[None, :] + torch.log(gmm.weights)[None, :]
    w_post = torch.softmax(log_w, dim=1) # [B, K]
    
    # Sampling
    comp_idx = torch.multinomial(w_post, n_samples, replacement=True) # [B, n_samples]
    z = torch.randn(B, n_samples, 2, device=gmm.device)
    
    # Gather means/covs
    # mu_post: [B, K, 2] -> gather -> [B, n_samples, 2]
    b_idx = torch.arange(B, device=gmm.device)[:, None]
    mu_gathered = mu_post[b_idx, comp_idx, :]
    L_gathered = L_post[comp_idx] # [B, n_samples, 2, 2]
    
    samples = mu_gathered + torch.einsum('bsde,bse->bsd', L_gathered, z)
    
    return samples

# =============================================================================
# Plotting Helpers
# =============================================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "monospace",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })

# =============================================================================
# 1. Intuition Plots (Arrows)
# =============================================================================

def plot_intuition_comparison(gmm, out_dir):
    """Visual comparison of DSM (Noise) vs Force (Data) vectors."""
    print("Generating Intuition Comparison plot...")
    setup_style()
    
    noise_levels = [0.1, 0.5, 0.9]
    n_samples = 8
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=150)
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    c_x0, c_xt = '#2ecc71', '#e74c3c'
    c_dsm, c_force = '#3498db', '#9b59b6'
    
    # Background grid
    lim = 4.5
    xx = np.linspace(-lim, lim, 100)
    XX, YY = np.meshgrid(xx, xx)
    XY = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32)
    P0 = torch.exp(gmm.log_prob(XY)).reshape(100, 100).numpy()
    
    for row, t in enumerate(noise_levels):
        t_tensor = torch.tensor(t)
        a, s = alpha(t_tensor), sigma(t_tensor)
        
        torch.manual_seed(42 + row)
        x_0 = gmm.sample(n_samples)
        eps = torch.randn_like(x_0)
        x_t = a * x_0 + s * eps
        
        dsm_target = -eps / s
        force_target = gmm.score(x_0) / a
        
        arrow_scale = 0.4
        
        # Col 0: Setup
        ax = axes[row, 0]
        ax.contourf(XX, YY, P0, levels=15, cmap='Greens', alpha=0.3)
        ax.scatter(x_0[:,0], x_0[:,1], c=c_x0, s=40, edgecolors='w', label=r'$x_0$')
        ax.scatter(x_t[:,0], x_t[:,1], c=c_xt, s=40, edgecolors='w', label=r'$x_t$')
        for i in range(n_samples):
            ax.plot([x_0[i,0], x_t[i,0]], [x_0[i,1], x_t[i,1]], 'k--', alpha=0.3, lw=0.8)
        
        ax.set_ylabel(f't = {t}\nσ={s:.2f}', fontweight='bold')
        if row==0: ax.set_title(r'Setup: $x_t = \alpha_t x_0 + \sigma_t \epsilon$')
        
        # Col 1: DSM
        ax = axes[row, 1]
        ax.contourf(XX, YY, P0, levels=15, cmap='Greens', alpha=0.2)
        ax.scatter(x_t[:,0], x_t[:,1], c=c_xt, s=40, edgecolors='w')
        for i in range(n_samples):
            ax.arrow(x_t[i,0], x_t[i,1], arrow_scale*dsm_target[i,0], arrow_scale*dsm_target[i,1],
                     width=0.05, color=c_dsm, alpha=0.8)
        if row==0: ax.set_title(r'DSM: $-\epsilon / \sigma_t$')
        
        # Col 2: Force
        ax = axes[row, 2]
        ax.contourf(XX, YY, P0, levels=15, cmap='Greens', alpha=0.2)
        ax.scatter(x_t[:,0], x_t[:,1], c=c_xt, s=40, edgecolors='w')
        for i in range(n_samples):
            ax.arrow(x_t[i,0], x_t[i,1], arrow_scale*force_target[i,0], arrow_scale*force_target[i,1],
                     width=0.05, color=c_force, alpha=0.8)
        if row==0: ax.set_title(r'Force: $\nabla \log p_0(x_0) / \alpha_t$')
        
        # Col 3: Comparison
        ax = axes[row, 3]
        ax.contourf(XX, YY, P0, levels=15, cmap='Greens', alpha=0.2)
        ax.scatter(x_t[:,0], x_t[:,1], c=c_xt, s=40, edgecolors='w')
        offset = 0.08
        for i in range(n_samples):
            # DSM (blue)
            ax.arrow(x_t[i,0]-offset, x_t[i,1], arrow_scale*dsm_target[i,0], arrow_scale*dsm_target[i,1],
                     width=0.04, color=c_dsm, alpha=0.7)
            # Force (purple)
            ax.arrow(x_t[i,0]+offset, x_t[i,1], arrow_scale*force_target[i,0], arrow_scale*force_target[i,1],
                     width=0.04, color=c_force, alpha=0.7)
        if row==0: ax.set_title('Comparison')

        for ax_row in axes[row]:
            ax_row.set_xlim(-lim, lim)
            ax_row.set_ylim(-lim, lim)
            ax_row.set_xticks([])
            ax_row.set_yticks([])
            ax_row.set_aspect('equal')

    fig.suptitle('DSM vs Force Matching Estimators', fontsize=16, fontweight='bold')
    out_path = out_dir / "dsm_vs_force_intuition.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def plot_single_detailed_example(gmm, out_dir):
    """Detailed view of a single point to show vector alignment."""
    print("Generating Single Example plot...")
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    
    # Grid for density
    lim = 4.5
    xx = np.linspace(-lim, lim, 100)
    XX, YY = np.meshgrid(xx, xx)
    
    # Specific point
    x_0 = torch.tensor([[-1.8, -1.9]])
    torch.manual_seed(456)
    eps = torch.randn_like(x_0)
    
    c_x0, c_xt = '#27ae60', '#c0392b'
    c_dsm, c_force = '#2980b9', '#8e44ad'
    
    for i, t in enumerate([0.1, 0.5, 0.9]):
        t_tensor = torch.tensor(t)
        a, s = alpha(t_tensor), sigma(t_tensor)
        
        # Marginal Density p_t
        P_t = noised_gmm_density(XX, YY, t, gmm)
        
        x_t = a * x_0 + s * eps
        dsm_vec = -eps / s
        force_vec = gmm.score(x_0) / a
        
        ax = axes[i]
        ax.contourf(XX, YY, P_t, levels=20, cmap='Blues', alpha=0.5)
        
        # Connection
        ax.plot([x_0[0,0], x_t[0,0]], [x_0[0,1], x_t[0,1]], 'k--', alpha=0.4)
        
        # Points
        ax.scatter(x_0[:,0], x_0[:,1], c=c_x0, s=150, edgecolors='w', zorder=10, label=r'$x_0$')
        ax.scatter(x_t[:,0], x_t[:,1], c=c_xt, s=150, edgecolors='w', zorder=10, label=r'$x_t$')
        
        # Arrows (normalized for visibility)
        scale = 0.5
        ax.arrow(x_t[0,0], x_t[0,1], scale*dsm_vec[0,0], scale*dsm_vec[0,1],
                 width=0.08, color=c_dsm, label='DSM', zorder=5)
        ax.arrow(x_t[0,0], x_t[0,1], scale*force_vec[0,0], scale*force_vec[0,1],
                 width=0.05, color=c_force, label='Force', zorder=6) # Force on top slightly thinner
        
        ax.set_title(f't={t} (noise={s:.2f})')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        if i==0: ax.legend()

    out_path = out_dir / "single_example_vectors.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# 2. Quantitative Analysis (Bias/Variance)
# =============================================================================

def compute_stats(gmm, t_vals, n_test=500, K_post=20, n_runs=50):
    """Compute Bias/Variance stats across noise levels."""
    print(f"Computing stats for {len(t_vals)} noise levels...")
    
    stats = {'t': [], 'noise_var': [], 'data_var': [], 'bias': [], 'mse': []}
    
    for t in t_vals:
        t_tensor = torch.tensor(t)
        a, s = alpha(t_tensor), sigma(t_tensor)
        
        # Sample test set x_t
        x_0 = gmm.sample(n_test)
        eps_base = torch.randn_like(x_0)
        x_t = a * x_0 + s * eps_base
        
        # Ground Truth Score
        gt_score = noised_gmm_score(x_t, t, gmm)
        
        # Noise Score (Analytic: -eps/sigma)
        # Note: We need the specific eps that generated x_t to evaluate the estimator 
        # But variance is over eps ~ p(eps).
        # Here we compute variance of the Data Score estimator which uses MC sampling of posterior.
        # The "Noise Score" estimator has variance 1/sigma^2 * I analytically.
        
        noise_var_analytic = (1 / s**2).item()
        
        # Data Score Estimator Statistics (Monte Carlo over n_runs)
        # We estimate E[hat{s}] and Var[hat{s}] where hat{s} uses K posterior samples
        estimates = []
        for _ in range(n_runs):
            # Sample posterior
            x0_post = sample_posterior(x_t, t, gmm, n_samples=K_post) # [n_test, K, 2]
            # Average score
            score_vals = gmm.score(x0_post.reshape(-1, 2)).reshape(n_test, K_post, 2)
            est = score_vals.mean(dim=1) / a # [n_test, 2]
            estimates.append(est)
            
        estimates = torch.stack(estimates) # [n_runs, n_test, 2]
        
        est_mean = estimates.mean(dim=0)
        est_var = estimates.var(dim=0).sum(dim=1).mean().item() # Trace of cov, averaged over x_t
        bias_sq = (est_mean - gt_score).pow(2).sum(dim=1).mean().item()
        mse = (estimates - gt_score.unsqueeze(0)).pow(2).sum(dim=2).mean().item()
        
        stats['t'].append(t)
        stats['noise_var'].append(2 * noise_var_analytic) # Trace is 2 * 1/sigma^2
        stats['data_var'].append(est_var)
        stats['bias'].append(bias_sq) # Bias squared
        stats['mse'].append(mse)
        
    return stats

def plot_bias_variance_analysis(gmm, out_dir):
    """Plot Bias-Variance curves."""
    print("Generating Bias-Variance analysis...")
    setup_style()
    
    ts = np.linspace(0.05, 0.95, 15)
    stats = compute_stats(gmm, ts, n_test=400, K_post=10, n_runs=40)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    
    # 1. Variance Comparison
    ax = axes[0]
    ax.semilogy(stats['t'], stats['noise_var'], 'o-', label='Noise Score (DSM)')
    ax.semilogy(stats['t'], stats['data_var'], 's-', label='Data Score (Force)')
    ax.set_xlabel('Noise level t')
    ax.set_ylabel('Variance (log)')
    ax.set_title('Variance Profile')
    ax.legend()
    
    # 2. Bias of Data Score (should be near zero)
    ax = axes[1]
    ax.plot(stats['t'], stats['bias'], 'o-', color='tab:green')
    ax.set_xlabel('Noise level t')
    ax.set_title('Bias of Data Score Estimator\n(Should be ≈ 0)')
    ax.set_ylim(bottom=0)
    
    # 3. MSE Comparison (if we assumed Noise Score was just an estimator)
    # Note: Noise score is exact given eps. But as an estimator of score from x_t only, 
    # its MSE is high at low t.
    ax = axes[2]
    ax.semilogy(stats['t'], stats['noise_var'], 'o-', label='Noise Score MSE', alpha=0.5)
    ax.semilogy(stats['t'], stats['mse'], 's-', label='Data Score MSE')
    ax.set_xlabel('Noise level t')
    ax.set_title('Total Error (MSE)')
    ax.legend()
    
    fig.suptitle('Quantitative Comparison', fontsize=16)
    fig.tight_layout()
    out_path = out_dir / "bias_variance_analysis.png"
    fig.savefig(out_path)
    plt.close(fig)

# =============================================================================
# 3. Animations
# =============================================================================

def create_convergence_gif(gmm, out_dir):
    """Animation showing Monte Carlo convergence of Data Score."""
    print("Generating Convergence GIF...")
    frames_dir = out_dir / "convergence_frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Fix x_t
    x_t = torch.tensor([[-1.0, -0.5]])
    
    t_vals = [0.2, 0.5, 0.8]
    n_max = 150
    sample_steps = list(range(1, 20)) + list(range(20, 50, 2)) + list(range(50, n_max, 5))
    
    # Precompute samples
    all_samples = {}
    gt_scores = {}
    
    for t in t_vals:
        t_ten = torch.tensor(t)
        a = alpha(t_ten)
        all_samples[t] = sample_posterior(x_t, t, gmm, n_samples=n_max) # [1, N, 2]
        gt_scores[t] = noised_gmm_score(x_t, t, gmm)[0]
    
    lim = 4.5
    xx = np.linspace(-lim, lim, 100)
    XX, YY = np.meshgrid(xx, xx)
    
    for i, K in enumerate(sample_steps):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        
        for j, t in enumerate(t_vals):
            ax = axes[j]
            t_ten = torch.tensor(t)
            a = alpha(t_ten)
            
            # Density background
            P_t = noised_gmm_density(XX, YY, t, gmm)
            ax.contourf(XX, YY, P_t, levels=10, cmap='Greys', alpha=0.2)
            
            # Samples
            samps = all_samples[t][0, :K]
            ax.scatter(samps[:,0], samps[:,1], c='tab:purple', s=10, alpha=0.5, label='Samples')
            
            # Running average score
            # Score est = (1/K) sum score(x_0) / a
            scores = gmm.score(samps)
            curr_est = scores.mean(dim=0) / a
            gt = gt_scores[t]
            
            # Vectors at x_t
            origin = x_t[0].numpy()
            scale = 0.5
            ax.arrow(origin[0], origin[1], scale*gt[0], scale*gt[1], 
                     width=0.08, color='k', alpha=0.3, label='GT')
            ax.arrow(origin[0], origin[1], scale*curr_est[0], scale*curr_est[1], 
                     width=0.05, color='tab:purple', label='Est')
            
            ax.set_title(f't={t} (K={K})')
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            if j==0: ax.legend(loc='upper left')
            
        fig.suptitle(f'Data Score Convergence: MC Estimation of Posterior Expectation', fontsize=14)
        fig.savefig(frames_dir / f"frame_{i:03d}.png")
        plt.close(fig)

    # Create GIF
    gif_path = out_dir / "score_convergence.gif"
    try:
        subprocess.run([
            "convert", "-delay", "10", "-loop", "0", 
            str(frames_dir / "frame_*.png"), 
            str(gif_path)
        ], check=True)
        print(f"Saved GIF: {gif_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ImageMagick 'convert' not found. Skipping GIF creation.")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    out_dir = default_out_dir()
    print(f"Output directory: {out_dir}")
    
    # Define GMM
    gmm = GMM(
        means=[[-2, -2], [-2, 2], [2, -2], [2, 2]],
        covs=[[[0.5, 0.2], [0.2, 0.5]],
              [[0.8, -0.3], [-0.3, 0.4]],
              [[0.3, 0.0], [0.0, 0.7]],
              [[0.6, 0.4], [0.4, 0.6]]],
        weights=[0.3, 0.2, 0.25, 0.25]
    )
    
    # Run Visualizations
    plot_intuition_comparison(gmm, out_dir)
    plot_single_detailed_example(gmm, out_dir)
    plot_bias_variance_analysis(gmm, out_dir)
    create_convergence_gif(gmm, out_dir)
    
    print("Done.")

