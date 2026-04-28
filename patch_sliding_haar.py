def update_sliding_haar():
    with open('src/waterSpec/haar_analysis.py', 'r') as f:
        content = f.read()

    search_block = """    # Pre-calculate window boundaries to avoid iterative searchsorted calls
    t_starts = np.arange(time[0], time[-1] - window_size + step_size, step_size)
    # Ensure we don't exceed time[-1] due to floating point issues
    t_starts = t_starts[t_starts + window_size <= time[-1] + 1e-9]"""

    replace_block = """    # Pre-calculate window boundaries to avoid iterative searchsorted calls
    t_starts = np.arange(time[0], time[-1] - window_size + step_size, step_size)
    # Ensure we don't exceed time[-1] due to floating point issues
    tol = window_size * 1e-9
    t_starts = t_starts[t_starts + window_size <= time[-1] + tol]"""

    content = content.replace(search_block, replace_block)

    with open('src/waterSpec/haar_analysis.py', 'w') as f:
        f.write(content)

update_sliding_haar()
