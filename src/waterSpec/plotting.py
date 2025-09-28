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


def plot_spectrum(
    frequency,
    power,
    fit_results,
    output_path=None,
    param_name="Parameter",
):
    """
    Generates and saves a plot of the power spectrum and its fit.
    The plot type is determined from the `fit_results` dictionary.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        fit_results (dict): The dictionary of results from the workflow.
        output_path (str, optional): The path to save the plot image. If None,
            the plot is displayed. Defaults to None.
        param_name (str, optional): The name of the parameter being plotted.
            Defaults to "Parameter".
    """
    plt.figure(figsize=(10, 6))

    # Plot the raw power spectrum
    plt.loglog(frequency, power, "o", markersize=5, alpha=0.6, label="Raw Periodogram")

    if _is_fit_successful(fit_results):
        analysis_type = fit_results.get("chosen_model_type")
        log_freq = fit_results.get("log_freq")

        if analysis_type == "standard":
            beta = fit_results.get("beta")
            intercept = fit_results.get("intercept")
            beta_ci_lower = fit_results.get("beta_ci_lower")
            beta_ci_upper = fit_results.get("beta_ci_upper")

            # Plot the main fit line
            fit_line = np.exp(intercept - beta * log_freq)
            plt.loglog(
                np.exp(log_freq),
                fit_line,
                "r-",
                linewidth=2,
                label=f"Standard Fit (β ≈ {beta:.2f})",
            )

            # Plot the confidence interval if available
            if beta_ci_lower is not None and beta_ci_upper is not None:
                lower_bound = np.exp(intercept - beta_ci_upper * log_freq)
                upper_bound = np.exp(intercept - beta_ci_lower * log_freq)
                plt.fill_between(
                    np.exp(log_freq),
                    lower_bound,
                    upper_bound,
                    color="r",
                    alpha=0.2,
                    label="95% CI on β",
                )

        elif analysis_type == "segmented":
            n_breakpoints = fit_results.get("n_breakpoints", 0)
            log_power_fit = fit_results.get("fitted_log_power")
            log_bps = [np.log(bp) for bp in fit_results["breakpoints"]]
            colors = ["r", "m", "g"]

            # Plot the confidence interval for the entire fit if available
            fit_ci_lower = fit_results.get("fit_ci_lower")
            fit_ci_upper = fit_results.get("fit_ci_upper")
            if fit_ci_lower is not None and fit_ci_upper is not None:
                plt.fill_between(
                    np.exp(log_freq),
                    np.exp(fit_ci_lower),
                    np.exp(fit_ci_upper),
                    color="gray",
                    alpha=0.3,
                    label="95% CI on Fit",
                )

            # Plot each segment
            for i in range(n_breakpoints + 1):
                # Define the mask for this segment
                if i == 0:
                    mask = log_freq <= log_bps[0]
                    label = f"Low-Freq Fit (β1 ≈ {fit_results['betas'][0]:.2f})"
                elif i == n_breakpoints:
                    mask = log_freq > log_bps[i - 1]
                    label = f"High-Freq Fit (β{i+1} ≈ {fit_results['betas'][i]:.2f})"
                else:
                    mask = (log_freq > log_bps[i - 1]) & (log_freq <= log_bps[i])
                    label = f"Mid-Freq Fit (β{i+1} ≈ {fit_results['betas'][i]:.2f})"

                plt.loglog(
                    np.exp(log_freq[mask]),
                    np.exp(log_power_fit[mask]),
                    color=colors[i % len(colors)],
                    linestyle="-",
                    linewidth=2.5,
                    label=label,
                )

            # Plot breakpoint vertical lines
            linestyles = ["--", ":", "-."]
            for i, bp_freq in enumerate(fit_results["breakpoints"]):
                plt.axvline(
                    x=bp_freq,
                    color="k",
                    linestyle=linestyles[i % len(linestyles)],
                    alpha=0.8,
                    label=f"Breakpoint {i+1} ≈ {_format_period(bp_freq)}",
                )
    else:
        # If the fit was not successful, add a prominent annotation
        plt.text(
            0.5,
            0.5,
            "Spectral model fitting failed",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=14,
            color="red",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

    # Plot the FAP level and annotate significant peaks if available
    fap_level = fit_results.get("fap_level")
    if fap_level is not None:
        fap_threshold_val = fit_results.get("fap_threshold")
        label = "FAP Threshold"
        if isinstance(fap_threshold_val, (float, int)):
            label += f" ({fap_threshold_val*100:.0f}%)"
        plt.axhline(fap_level, ls="--", color="k", alpha=0.8, label=label)

    significant_peaks = fit_results.get("significant_peaks", [])
    for i, peak in enumerate(significant_peaks):
        peak_freq = peak["frequency"]
        peak_power = peak["power"]

        # Create annotation text based on which significance info is available
        if "fap" in peak:
            annotation_text = f'Period: {_format_period(peak_freq)}\n(FAP: {peak["fap"]:.2E})'
        elif "residual" in peak:
            annotation_text = f"Period: {_format_period(peak_freq)}\n(Residual: {peak['residual']:.2f})"
        else:
            annotation_text = f"Period: {_format_period(peak_freq)}"

        # Stagger annotations to reduce overlap
        vertical_offset = 1.5 if i % 2 == 0 else 2.5
        plt.annotate(
            annotation_text,
            xy=(peak_freq, peak_power),
            xytext=(peak_freq, peak_power * vertical_offset),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.9),
        )

    plt.title(f"Power Spectrum for {param_name}", fontsize=16)
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    plt.close()
