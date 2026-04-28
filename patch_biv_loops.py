import numpy as np

def update_biv():
    with open('src/waterSpec/bivariate.py', 'r') as f:
        content = f.read()

    # Block 1
    search_block_1 = """            # Generate window boundaries
            t_starts_list = []
            curr_t = t_min
            while curr_t + tau <= t_max + 1e-9:
                t_starts_list.append(curr_t)
                if overlap:
                    curr_t += step_size
                else:
                    curr_t += tau
                    if curr_t >= t_max + 1e-9: break

            if t_starts_list:
                t_starts = np.array(t_starts_list)"""

    replace_block_1 = """            # Generate window boundaries
            n_windows_max = int(np.floor((t_max - t_min - tau) / step_size)) + 1
            if n_windows_max > 0:
                t_starts = t_min + np.arange(n_windows_max) * step_size
                tol = tau * 1e-9
                t_starts = t_starts[t_starts + tau <= t_max + tol]
            else:
                t_starts = np.array([])

            if len(t_starts) > 0:"""

    content = content.replace(search_block_1, replace_block_1)

    # Block 2
    search_block_2 = """        t_starts_list = []
        curr_t = time[0]
        while curr_t + tau <= time[-1] + 1e-9:
            t_starts_list.append(curr_t)
            if overlap:
                curr_t += step_size
            else:
                curr_t += tau
                if curr_t >= time[-1] + 1e-9: break

        if t_starts_list:
            t_starts = np.array(t_starts_list)"""

    replace_block_2 = """        n_windows_max = int(np.floor((time[-1] - time[0] - tau) / step_size)) + 1
        if n_windows_max > 0:
            t_starts = time[0] + np.arange(n_windows_max) * step_size
            tol = tau * 1e-9
            t_starts = t_starts[t_starts + tau <= time[-1] + tol]
        else:
            t_starts = np.array([])

        if len(t_starts) > 0:"""

    content = content.replace(search_block_2, replace_block_2)

    with open('src/waterSpec/bivariate.py', 'w') as f:
        f.write(content)

update_biv()
