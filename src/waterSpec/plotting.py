import os
import re
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .interpreter import _format_period


def _is_fit_successful(fit_results):
    """Checks if a model fit was successful based on the presence of valid results."""
    is_standard_success = "beta" in fit_results and np.isfinite(fit_results.get("beta"))
    is_segmented_success = (
        "betas" in fit_results
        and len(fit_results.get("betas", [])) > 0
        and np.isfinite(fit_results["betas"][0])
    )
    return is_standard_success or is_segmented_success


def _plot_standard_model(ax, fit_results, color, label_prefix, plot_ci, log_freq):
    """Helper to plot the standard model fit."""
    beta = fit_results.get("beta")
    intercept = fit_results.get("intercept")
    fit_line = 10 ** (intercept - beta * log_freq)

    # Determine label
    label = f"{label_prefix} Fit (β ≈ {beta:.2f})".strip()

    ax.loglog(
        10**log_freq,
        fit_line,
        "-",
        color=color,
        linewidth=2.5,
        label=label,
    )

    if plot_ci:
        beta_ci_lower = fit_results.get("beta_ci_lower")
        beta_ci_upper = fit_results.get("beta_ci_upper")
        # Plot the confidence interval if available
        if beta_ci_lower is not None and beta_ci_upper is not None:
            lower_bound = 10 ** (intercept - beta_ci_upper * log_freq)
            upper_bound = 10 ** (intercept - beta_ci_lower * log_freq)
            ax.fill_between(
                10**log_freq,
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.2,
                label="95% CI on β",
            )


def _plot_segmented_model(ax, fit_results, color, label_prefix, plot_ci, log_freq, plot_breakpoints, use_segment_colors):
    """Helper to plot the segmented model fit."""
    n_breakpoints = fit_results.get("n_breakpoints", 0)
    log_power_fit = fit_results.get("fitted_log_power")

    if use_segment_colors:
        # Detailed plotting
        log_bps = [np.log10(bp) for bp in fit_results["breakpoints"]]
        colors = ["r", "m", "g"]

        # Plot the confidence interval for the entire fit if available
        if plot_ci:
            fit_ci_lower = fit_results.get("fit_ci_lower")
            fit_ci_upper = fit_results.get("fit_ci_upper")
            if fit_ci_lower is not None and fit_ci_upper is not None:
                ax.fill_between(
                    10**log_freq,
                    10**fit_ci_lower,
                    10**fit_ci_upper,
                    color="gray",
                    alpha=0.3,
                    label="95% CI on Fit",
                )

        # Plot each segment
        for i in range(n_breakpoints + 1):
            if i == 0:
                mask = log_freq <= log_bps[0]
                label = f"Low-Freq (β1≈{fit_results['betas'][0]:.2f})"
            elif i == n_breakpoints:
                mask = log_freq > log_bps[i - 1]
                label = f"High-Freq (β{i+1}≈{fit_results['betas'][i]:.2f})"
            else:
                mask = (log_freq > log_bps[i - 1]) & (log_freq <= log_bps[i])
                label = f"Mid-Freq (β{i+1}≈{fit_results['betas'][i]:.2f})"

            ax.loglog(
                10 ** log_freq[mask],
                10 ** log_power_fit[mask],
                color=colors[i % len(colors)],
                linestyle="-",
                linewidth=2.5,
                label=label,
            )
    else:
        # Simple plotting
        betas = fit_results.get("betas", [])
        beta_str = ", ".join([f"β{i+1}≈{b:.2f}" for i, b in enumerate(betas)])
        label = f"{label_prefix} Fit ({beta_str})".strip()

        ax.loglog(
            10**log_freq,
            10**log_power_fit,
            "-",
            color=color,
            linewidth=2.5,
            label=label,
        )

    # Plot breakpoint vertical lines
    if plot_breakpoints and "breakpoints" in fit_results:
        linestyles = ["--", ":", "-."]
        for i, bp_freq in enumerate(fit_results["breakpoints"]):
            ax.axvline(
                x=bp_freq,
                color="k",
                linestyle=linestyles[i % len(linestyles)],
                alpha=0.8,
                label=f"BP {i+1} ≈ {_format_period(bp_freq)}",
            )


def _plot_fit_line(
    ax,
    fit_results,
    color,
    label_prefix="",
    plot_ci=False,
    plot_breakpoints=False,
    use_segment_colors=False,
):
    """A helper to plot just the fit line and CI on a given axis."""
    if not _is_fit_successful(fit_results):
        return

    analysis_type = fit_results.get("chosen_model_type")
    log_freq = fit_results.get("log_freq")

    if analysis_type == "standard":
        _plot_standard_model(ax, fit_results, color, label_prefix, plot_ci, log_freq)
    elif analysis_type == "segmented":
        _plot_segmented_model(ax, fit_results, color, label_prefix, plot_ci, log_freq, plot_breakpoints, use_segment_colors)


def _plot_single_spectrum(ax, frequency, power, fit_results, title=""):
    """
    Plots a single power spectrum and its fit on a given matplotlib Axes object.
    """
    # Plot the raw power spectrum
    ax.loglog(frequency, power, "o", markersize=5, alpha=0.6, label="Raw Periodogram")

    if _is_fit_successful(fit_results):
        _plot_fit_line(
            ax,
            fit_results,
            color="r",
            plot_ci=True,
            plot_breakpoints=True,
            use_segment_colors=True,
        )
    else:
        ax.text(
            0.5, 0.5, "Fit Failed", ha="center", va="center", transform=ax.transAxes
        )

    # Plot FAP level and peaks
    if "fap_level" in fit_results:
        ax.axhline(
            fit_results["fap_level"], ls="--", color="k", alpha=0.8, label="FAP Level"
        )
    for i, peak in enumerate(fit_results.get("significant_peaks", [])):
        ax.annotate(
            f"{_format_period(peak['frequency'])}",
            xy=(peak["frequency"], peak["power"]),
            xytext=(peak["frequency"], peak["power"] * (1.5 if i % 2 == 0 else 2.5)),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
            ha="center",
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()


def plot_spectrum(
    frequency,
    power,
    fit_results,
    output_path=None,
    param_name="Parameter",
    show: bool = False,
):
    """
    Generates, saves, or shows a plot of the power spectrum and its fit.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_single_spectrum(
        ax, frequency, power, fit_results, title=f"Power Spectrum for {param_name}"
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_changepoint_analysis(
    results: Dict, output_dir: str, param_name: str, plot_style: str = "separate"
):
    """
    Creates a comparison plot for a changepoint analysis.

    Args:
        results (Dict): The results dictionary from the changepoint analysis.
        output_dir (str): The directory to save the plot.
        param_name (str): The name of the parameter being analyzed.
        plot_style (str): The style of plot, either 'separate' or 'combined'.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    before_seg = results["segment_before"]
    after_seg = results["segment_after"]
    cp_time_str = results["changepoint_time"]
    sanitized_name = re.sub(
        r"(?u)[^-\w.]", "", str(param_name).strip().replace(" ", "_")
    )

    if plot_style == "separate":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        _plot_single_spectrum(
            ax1,
            before_seg["frequency"],
            before_seg["power"],
            before_seg,
            title=f"Before Changepoint (~{cp_time_str})",
        )
        _plot_single_spectrum(
            ax2,
            after_seg["frequency"],
            after_seg["power"],
            after_seg,
            title=f"After Changepoint (~{cp_time_str})",
        )
        fig.suptitle(f"Changepoint Analysis for {param_name}", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"{sanitized_name}_changepoint_separate.png"

    elif plot_style == "combined":
        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot "Before" segment
        ax.loglog(
            before_seg["frequency"],
            before_seg["power"],
            "o",
            color="blue",
            alpha=0.5,
            label=f"Before ~{cp_time_str}",
        )
        # Plot "After" segment
        ax.loglog(
            after_seg["frequency"],
            after_seg["power"],
            "s",
            color="green",
            alpha=0.5,
            markersize=5,
            label=f"After ~{cp_time_str}",
        )
        # Manually plot fit lines with distinct colors
        _plot_fit_line(ax, before_seg, color="darkblue", label_prefix="Before")
        _plot_fit_line(ax, after_seg, color="darkgreen", label_prefix="After")
        ax.set_title(f"Changepoint Analysis for {param_name}", fontsize=16)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        filename = f"{sanitized_name}_changepoint_combined.png"

    else:
        raise ValueError("plot_style must be 'separate' or 'combined'.")

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return fig


def plot_site_comparison(
    results: Dict, output_dir: str, plot_style: str = "separate"
):
    """
    Creates a comparison plot for a two-site analysis.

    Args:
        results (Dict): The results dictionary from the site comparison analysis.
        output_dir (str): The directory to save the plot.
        plot_style (str): The style of plot, either 'separate' or 'overlaid'.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    site1_results = results["site1"]
    site2_results = results["site2"]
    site1_name = site1_results["site_name"]
    site2_name = site2_results["site_name"]
    comparison_name = results["comparison_name"]
    sanitized_name = re.sub(
        r"(?u)[^-\w.]", "", str(comparison_name).strip().replace(" ", "_")
    )

    if plot_style == "separate":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        _plot_single_spectrum(
            ax1,
            site1_results["frequency"],
            site1_results["power"],
            site1_results,
            title=f"Spectrum for {site1_name}",
        )
        _plot_single_spectrum(
            ax2,
            site2_results["frequency"],
            site2_results["power"],
            site2_results,
            title=f"Spectrum for {site2_name}",
        )
        fig.suptitle(f"Site Comparison: {site1_name} vs. {site2_name}", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"{sanitized_name}_comparison_separate.png"

    elif plot_style == "overlaid":
        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot Site 1
        ax.loglog(
            site1_results["frequency"],
            site1_results["power"],
            "o",
            color="blue",
            alpha=0.5,
            label=f"Periodogram for {site1_name}",
        )
        # Plot Site 2
        ax.loglog(
            site2_results["frequency"],
            site2_results["power"],
            "s",
            color="green",
            alpha=0.5,
            markersize=5,
            label=f"Periodogram for {site2_name}",
        )
        # Plot fit lines
        _plot_fit_line(ax, site1_results, color="darkblue", label_prefix=site1_name)
        _plot_fit_line(ax, site2_results, color="darkgreen", label_prefix=site2_name)
        ax.set_title(f"Site Comparison: {site1_name} vs. {site2_name}", fontsize=16)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        filename = f"{sanitized_name}_comparison_overlaid.png"

    else:
        raise ValueError("plot_style must be 'separate' or 'overlaid'.")

    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return fig