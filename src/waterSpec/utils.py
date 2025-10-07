import numpy as np


def make_rng(seed=None):
    """
    Creates a random number generator instance with consistent seed handling.

    This utility ensures that various seed types are handled correctly,
    preventing issues with correlated random sequences and improving
    reproducibility.

    Args:
        seed (None, int, np.random.SeedSequence, np.random.Generator):
            The seed to initialize the RNG.
            - If None, a new, unseeded generator is created.
            - If an int, it is used as the seed.
            - If a SeedSequence, it is used to spawn the generator.
            - If a Generator instance, it is returned directly.

    Returns:
        np.random.Generator: A NumPy random number generator instance.

    Raises:
        TypeError: If the seed is of an invalid type.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    elif isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    elif isinstance(seed, int):
        return np.random.default_rng(seed)
    elif seed is None:
        return np.random.default_rng()
    else:
        raise TypeError(
            f"Invalid seed type: {type(seed)}. Must be an int, "
            "np.random.SeedSequence, or np.random.Generator."
        )