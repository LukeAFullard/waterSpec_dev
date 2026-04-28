def update_surrogates():
    with open('src/waterSpec/surrogates.py', 'r') as f:
        content = f.read()

    # Add generate_iaaft_surrogates
    iaaft_code = """
def generate_iaaft_surrogates(
    data: np.ndarray, n_surrogates: int = 100,
    n_iter: int = 100, seed=None
) -> np.ndarray:
    \"\"\"
    Generates surrogates using Iterative Amplitude Adjusted Fourier Transform (IAAFT).
    Preserves both the power spectrum and the amplitude distribution of the original data.
    \"\"\"
    rng = np.random.default_rng(seed)
    target_amplitudes = np.abs(np.fft.rfft(data))
    sorted_data = np.sort(data)
    surrogates = np.empty((n_surrogates, len(data)))

    for i in range(n_surrogates):
        current = rng.permutation(data)
        for _ in range(n_iter):
            # Match spectrum
            fft_cur = np.fft.rfft(current)
            fft_phased = target_amplitudes * np.exp(1j * np.angle(fft_cur))
            current = np.fft.irfft(fft_phased, n=len(data))
            # Match amplitude distribution by rank-ordering
            rank = np.argsort(np.argsort(current))
            current = sorted_data[rank]
        surrogates[i] = current

    return surrogates
"""
    content = content + iaaft_code

    with open('src/waterSpec/surrogates.py', 'w') as f:
        f.write(content)

update_surrogates()
