import time
import numpy as np
from waterSpec.psresp import psresp_fit
import multiprocessing as mp

def dummy_psd(f, a, b):
    return a * f ** -b

t_obs = np.linspace(0, 100, 100)
x_obs = np.sin(2 * np.pi * t_obs / 10) + np.random.normal(0, 0.1, 100)
err_obs = np.ones(100) * 0.1

params_list = [(1.0, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 2.5), (1.0, 3.0),
               (2.0, 1.0), (2.0, 1.5), (2.0, 2.0), (2.0, 2.5), (2.0, 3.0)]

if __name__ == '__main__':
    mp.set_start_method('spawn')  # To emphasize process start overhead if any, but default fork is okay too
    start_time = time.time()
    res = psresp_fit(
        t_obs=t_obs,
        x_obs=x_obs,
        err_obs=err_obs,
        psd_func=dummy_psd,
        params_list=params_list,
        M=50, # Many params, fewer M to test overhead of executor creation
        n_jobs=4,
        binning=True
    )
    print(f"Time taken: {time.time() - start_time:.4f} seconds")
