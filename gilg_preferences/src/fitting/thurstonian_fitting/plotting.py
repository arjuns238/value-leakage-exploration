"""Plotting utilities for Thurstonian model diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def normalize_mu_for_comparison(
    true_mu: np.ndarray,
    fitted_mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize fitted μ to match the scale of true μ for fair comparison.

    The Thurstonian model fixes μ_0 = 0 and utility scale is arbitrary.
    This shifts true_mu to have μ_0 = 0 and scales fitted_mu to match.

    Returns:
        shifted_true: true_mu shifted so first element is 0
        scaled_fitted: fitted_mu scaled to match true_mu range
        scale: the scaling factor applied
    """
    shifted_true = true_mu - true_mu[0]
    true_range = shifted_true.max() - shifted_true.min()
    fitted_range = fitted_mu.max() - fitted_mu.min()
    scale = true_range / fitted_range if fitted_range > 0 else 1.0
    scaled_fitted = fitted_mu * scale
    return shifted_true, scaled_fitted, scale


def plot_mu_recovery(
    ax: Axes,
    true_mu: np.ndarray,
    fitted_mu: np.ndarray,
    label: str | None = None,
    alpha: float = 0.6,
    s: int = 25,
    add_diagonal: bool = True,
) -> float:
    """Plot true vs fitted μ with proper normalization.

    Returns the Pearson correlation coefficient.
    """
    shifted_true, scaled_fitted, _ = normalize_mu_for_comparison(true_mu, fitted_mu)

    ax.scatter(shifted_true, scaled_fitted, alpha=alpha, label=label, s=s)

    if add_diagonal:
        lims = [
            min(shifted_true.min(), scaled_fitted.min()),
            max(shifted_true.max(), scaled_fitted.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim(lims[0] - 0.5, lims[1] + 0.5)
        ax.set_ylim(lims[0] - 0.5, lims[1] + 0.5)

    ax.set_xlabel("True μ (shifted)")
    ax.set_ylabel("Fitted μ (scaled)")

    corr = np.corrcoef(true_mu, fitted_mu)[0, 1]
    return corr


def plot_sigma_recovery(
    ax: Axes,
    true_sigma: np.ndarray,
    fitted_sigma: np.ndarray,
    true_mu: np.ndarray,
    fitted_mu: np.ndarray,
    add_diagonal: bool = True,
) -> float:
    """Plot true vs fitted σ, scaled consistently with μ scaling.

    Returns the Pearson correlation coefficient.
    """
    _, _, scale = normalize_mu_for_comparison(true_mu, fitted_mu)
    scaled_fitted_sigma = fitted_sigma * scale

    ax.scatter(true_sigma, scaled_fitted_sigma, alpha=0.6)

    if add_diagonal:
        sigma_lims = [0, max(true_sigma.max(), scaled_fitted_sigma.max()) * 1.1]
        ax.plot(sigma_lims, sigma_lims, 'k--', alpha=0.5)

    ax.set_xlabel("True σ")
    ax.set_ylabel("Fitted σ (scaled)")

    corr = np.corrcoef(true_sigma, fitted_sigma)[0, 1] if true_sigma.std() > 0 else float('nan')
    return corr


def plot_ranking_matrix(
    ax: Axes,
    rank_corr: np.ndarray,
    labels: list[str],
    title: str = "Spearman Rank Correlation",
    vmin: float = 0.85,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
):
    """Plot a rank correlation matrix with values annotated."""
    im = ax.imshow(rank_corr, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = rank_corr[i, j]
            color = 'white' if val < 0.92 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    ax.set_title(title, fontsize=11, pad=10)
    return im
