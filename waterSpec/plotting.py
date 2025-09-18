import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(frequency, power, fit_results=None, output_path=None, show=True):
    """
    Plots the power spectrum on a log-log scale and optionally shows the fitted line.

    Args:
        frequency (np.ndarray): The frequency array.
        power (np.ndarray): The power array.
        fit_results (dict, optional): A dictionary with fitting results ('beta', 'intercept').
                                      If provided, the fitted line will be plotted. Defaults to None.
        output_path (str, optional): The path to save the plot image. If None, the plot
                                     is displayed interactively. Defaults to None.
        show (bool, optional): Whether to display the plot with plt.show().
                               Defaults to True.
    """
    plt.figure(figsize=(8, 6))

    # Plot the power spectrum
    plt.loglog(frequency, power, 'o', markersize=5, label='Power Spectrum')

    # Plot the fitted line if results are provided
    if fit_results:
        beta = fit_results.get('beta')
        intercept = fit_results.get('intercept')
        if beta is not None and intercept is not None:
            # Recreate the fitted line: log(power) = intercept - beta * log(frequency)
            # power = exp(intercept) * frequency ** -beta
            fit_line = np.exp(intercept) * (frequency ** -beta)
            plt.loglog(frequency, fit_line, 'r-', linewidth=2, label=f'Fit (β = {beta:.2f})')

    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    # Close the plot to free up memory, important for running many tests
    plt.close()
