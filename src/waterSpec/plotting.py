import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(frequency, power, fit_results, analysis_type='standard', output_path=None, param_name="Parameter"):
    """
    Generates and saves a plot of the power spectrum and its fit.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        fit_results (dict): The dictionary of fit results from the fitter module.
        analysis_type (str, optional): The type of analysis ('standard' or 'segmented').
                                       Defaults to 'standard'.
        output_path (str, optional): The path to save the plot image. If None, the plot is displayed.
                                     Defaults to None.
        param_name (str, optional): The name of the parameter being plotted. Defaults to "Parameter".
    """
    plt.figure(figsize=(10, 6))

    # Plot the raw power spectrum
    plt.loglog(frequency, power, 'o', markersize=5, alpha=0.6, label="Raw Periodogram")

    log_freq = np.log(frequency[frequency > 0])

    if analysis_type == 'standard':
        beta = fit_results.get('beta')
        intercept = fit_results.get('intercept')
        if beta is not None and intercept is not None:
            fit_line = np.exp(intercept - beta * log_freq)
            plt.loglog(np.exp(log_freq), fit_line, 'r-', linewidth=2, label=f'Standard Fit (β ≈ {beta:.2f})')

    elif analysis_type == 'segmented':
        model = fit_results.get('model_object')
        if model:
            model.plot_fit(fig=plt.gcf(), ax=plt.gca(), plot_data=False, plot_breakpoints=True, linewidth=2)
            plt.legend() # Re-add legend after piecewise-regression plot

    plt.title(f'Power Spectrum for {param_name}')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    plt.close()
