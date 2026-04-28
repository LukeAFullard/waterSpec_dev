def update_data_loader():
    with open('src/waterSpec/data_loader.py', 'r') as f:
        content = f.read()

    search_block = """        # Make time relative *before* converting to float to preserve precision
        time_numeric_ns_relative = time_numeric_ns - time_numeric_ns[0]
        time_numeric_sec = time_numeric_ns_relative.astype(np.float64) / 1e9"""

    replace_block = """        # Cast to float64 *before* subtraction to prevent int64 overflow
        # for datasets spanning > ~292 years
        time_float = time_numeric_ns.astype(np.float64)
        time_numeric_ns_relative = time_float - time_float[0]
        time_numeric_sec = time_numeric_ns_relative / 1e9"""

    content = content.replace(search_block, replace_block)

    with open('src/waterSpec/data_loader.py', 'w') as f:
        f.write(content)

update_data_loader()
