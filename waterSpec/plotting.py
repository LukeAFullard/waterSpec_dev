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

    # Plot the raw data on a log-log scale
    plt.loglog(frequency, power, 'o', markersize=5, label='Power Spectrum')

    # Plot the fitted line if results are provided
    if fit_results:
        pw_model = fit_results.get('model_object')
        if pw_model:
            # Use the model's built-in plotting method for an accurate fit line
            # Note: this plots on a linear scale, so we are doing it differently.
            # We will transform the axes back to log-log.

            # Since plot_fit is complex, we will extract the fit lines manually from the model
            # This gives us more control over the plotting style.

            # Get the breakpoint in log space
            breakpoint_log_freq = pw_model.get_results()["estimates"]["breakpoint1"]["estimate"]

            # Get the model predictions in log-power space
            log_freq_sorted = np.sort(pw_model.x)
            log_power_pred = pw_model.predict(log_freq_sorted)

            # Transform back to linear space for plotting
            freq_fit = np.exp(log_freq_sorted)
            power_fit = np.exp(log_power_pred)

            plt.loglog(freq_fit, power_fit, 'r-', linewidth=2, label='Segmented Fit')

            # Add a vertical line for the breakpoint
            breakpoint_freq = np.exp(breakpoint_log_freq)
            plt.axvline(breakpoint_freq, color='k', linestyle='--', label=f'Breakpoint = {breakpoint_freq:.2e}')

        else:
            # Handle standard single-slope plot
            beta = fit_results.get('beta')
            intercept = fit_results.get('intercept')
            if beta is not None and intercept is not None:
                # Recreate the fitted line: log(power) = intercept - beta * log(frequency)
                # power = exp(intercept) * frequency ** -beta
                fit_line = np.exp(intercept) * (frequency ** -beta)
                plt.loglog(frequency, fit_line, 'r-', linewidth=2, label=f'Fit (β = {beta:.2f})')

    plt.xlabel('Frequency (log scale)')
    plt.ylabel('Power (log scale)')
    plt.title('Log-Log Power Spectrum')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)

    if show:
        plt.show()

    # Close the plot to free up memory, important for running many tests
    plt.close()
