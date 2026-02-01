
import numpy as np
from typing import Union

def power_law(f: Union[float, np.ndarray], beta: float, amp: float) -> Union[float, np.ndarray]:
    """
    Power law PSD: P(f) = amp * f^(-beta)
    """
    return amp * (f**(-beta))

def red_noise_psd(f: Union[float, np.ndarray], tau: float, variance: float) -> Union[float, np.ndarray]:
    """
    PSD of a Red Noise (Ornstein-Uhlenbeck) process.
    P(f) = (2 * variance * tau) / (1 + (2 * pi * f * tau)^2)

    Args:
        f: Frequency (Hz).
        tau: Decorrelation time scale (seconds).
        variance: Variance of the process (sigma^2).
    """
    return (2 * variance * tau) / (1 + (2 * np.pi * f * tau)**2)
