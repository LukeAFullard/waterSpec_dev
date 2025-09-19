import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def _find_significant_peaks(frequency, power, prominence_factor=0.5, max_peaks=5):
    """
    Finds significant peaks in a power spectrum.
    """
    # The prominence is the vertical distance between a peak and its lowest contour line.
    # We set a dynamic prominence threshold based on the power range.
    prominence_threshold = (np.max(power) - np.min(power)) * prominence_factor
    peaks, properties = find_peaks(power, prominence=prominence_threshold)

    # Sort peaks by prominence in descending order and take the top N
    sorted_indices = np.argsort(properties['prominences'])[::-1]
    top_peaks = peaks[sorted_indices][:max_peaks]

    return top_peaks

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

    # Find and annotate significant peaks
    significant_peaks = _find_significant_peaks(frequency, power)
    for peak_idx in significant_peaks:
        peak_freq = frequency[peak_idx]
        peak_power = power[peak_idx]
        plt.annotate(f'{peak_freq:.2f}',
                     xy=(peak_freq, peak_power),
                     xytext=(peak_freq, peak_power * 1.2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                     ha='center')

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
