import matplotlib.pyplot as plt
import numpy as np

from .interpreter import _format_period


def plot_spectrum(
    frequency,
    power,
    fit_results,
    analysis_type="standard",
    output_path=None,
    param_name="Parameter",
):
    """
    Generates and saves a plot of the power spectrum and its fit.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        fit_results (dict): The dictionary of results from the workflow.
        analysis_type (str, optional): The type of analysis ('standard' or
            'segmented'). Defaults to 'standard'.
        output_path (str, optional): The path to save the plot image. If None,
            the plot is displayed. Defaults to None.
        param_name (str, optional): The name of the parameter being plotted.
            Defaults to "Parameter".
    """
    plt.figure(figsize=(10, 6))

    # Plot the raw power spectrum
    plt.loglog(frequency, power, "o", markersize=5, alpha=0.6, label="Raw Periodogram")

    log_freq = np.log(frequency[frequency > 0])

    if analysis_type == "standard":
        beta = fit_results.get("beta")
        intercept = fit_results.get("intercept")
        beta_ci_lower = fit_results.get("beta_ci_lower")
        beta_ci_upper = fit_results.get("beta_ci_upper")

        if beta is not None and intercept is not None:
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
                    label="95% CI",
                )

    elif analysis_type == "segmented":
        n_breakpoints = fit_results.get("n_breakpoints", 0)
        model = fit_results.get("model_object")
        log_freq_full = fit_results.get("log_freq")

        if model and log_freq_full is not None:
            # Predict power values across the full frequency range using the
            # fitted model
            log_power_fit = model.predict(log_freq_full)

            if n_breakpoints == 1:
                breakpoint_freq = fit_results.get("breakpoint")
                beta1 = fit_results.get("beta1")
                beta2 = fit_results.get("beta2")
                if all(v is not None for v in [breakpoint_freq, beta1, beta2]):
                    log_breakpoint = np.log(breakpoint_freq)
                    mask1 = log_freq_full <= log_breakpoint
                    mask2 = log_freq_full > log_breakpoint

                    plt.loglog(
                        np.exp(log_freq_full[mask1]),
                        np.exp(log_power_fit[mask1]),
                        color="r",
                        linestyle="-",
                        linewidth=2,
                        label=f"Low-Freq Fit (β1 ≈ {beta1:.2f})",
                    )
                    plt.loglog(
                        np.exp(log_freq_full[mask2]),
                        np.exp(log_power_fit[mask2]),
                        color="m",
                        linestyle="-",
                        linewidth=2,
                        label=f"High-Freq Fit (β2 ≈ {beta2:.2f})",
                    )
                    plt.axvline(
                        x=breakpoint_freq,
                        color="k",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Breakpoint ≈ {_format_period(breakpoint_freq)}",
                    )

            elif n_breakpoints == 2:
                bp1 = fit_results.get("breakpoint1")
                bp2 = fit_results.get("breakpoint2")
                beta1 = fit_results.get("beta1")
                beta2 = fit_results.get("beta2")
                beta3 = fit_results.get("beta3")

                if all(v is not None for v in [bp1, bp2, beta1, beta2, beta3]):
                    log_bp1 = np.log(bp1)
                    log_bp2 = np.log(bp2)

                    mask1 = log_freq_full <= log_bp1
                    mask2 = (log_freq_full > log_bp1) & (log_freq_full <= log_bp2)
                    mask3 = log_freq_full > log_bp2

                    plt.loglog(
                        np.exp(log_freq_full[mask1]),
                        np.exp(log_power_fit[mask1]),
                        color="r",
                        linestyle="-",
                        linewidth=2,
                        label=f"Seg 1 (β1 ≈ {beta1:.2f})",
                    )
                    plt.loglog(
                        np.exp(log_freq_full[mask2]),
                        np.exp(log_power_fit[mask2]),
                        color="m",
                        linestyle="-",
                        linewidth=2,
                        label=f"Seg 2 (β2 ≈ {beta2:.2f})",
                    )
                    plt.loglog(
                        np.exp(log_freq_full[mask3]),
                        np.exp(log_power_fit[mask3]),
                        color="g",
                        linestyle="-",
                        linewidth=2,
                        label=f"Seg 3 (β3 ≈ {beta3:.2f})",
                    )

                    plt.axvline(
                        x=bp1,
                        color="k",
                        linestyle="--",
                        alpha=0.7,
                        label=f"BP 1 ≈ {_format_period(bp1)}",
                    )
                    plt.axvline(
                        x=bp2,
                        color="c",
                        linestyle=":",
                        alpha=0.7,
                        label=f"BP 2 ≈ {_format_period(bp2)}",
                    )

    # Plot the FAP level and annotate significant peaks if available
    fap_level = fit_results.get("fap_level")
    if fap_level is not None:
        fap_threshold_val = fit_results.get("fap_threshold")
        label = "FAP Threshold"
        if isinstance(fap_threshold_val, (float, int)):
            label += f" ({fap_threshold_val:.2f})"
        plt.axhline(fap_level, ls="--", color="k", alpha=0.8, label=label)

    significant_peaks = fit_results.get("significant_peaks", [])
    for peak in significant_peaks:
        peak_freq = peak["frequency"]
        peak_power = peak["power"]

        # Create annotation text based on which significance info is available
        if "fap" in peak:
            annotation_text = (
                f'Period: {_format_period(peak_freq)}\n(FAP: {peak["fap"]:.2E})'
            )
        elif "residual" in peak:
            annotation_text = (
                f'Period: {_format_period(peak_freq)}\n'
                f'(Residual: {peak["residual"]:.2f})'
            )
        else:
            annotation_text = f"Period: {_format_period(peak_freq)}"

        plt.annotate(
            annotation_text,
            xy=(peak_freq, peak_power),
            xytext=(peak_freq, peak_power * 1.5),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.8),
        )

    # Add summary text box
    summary_text = fit_results.get("summary_text")
    if summary_text:
        # Position the text box in the bottom left corner
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax = plt.gca()
        plt.text(
            0.03,
            0.03,
            summary_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=props,
        )

    plt.title(f"Power Spectrum for {param_name}")
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
