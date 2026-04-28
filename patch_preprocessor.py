def update_preprocessor():
    with open('src/waterSpec/preprocessor.py', 'r') as f:
        content = f.read()

    search_block = """    normalized_data = np.copy(data)
    normalized_errors = np.copy(errors) if errors is not None else None

    valid_indices = ~np.isnan(normalized_data)
    valid_data = normalized_data[valid_indices]
    if len(valid_data) == 0:
        return normalized_data, normalized_errors"""

    replace_block = """    normalized_data = np.copy(data)
    normalized_errors = np.copy(errors) if errors is not None else None

    valid_indices = np.isfinite(normalized_data)
    valid_data = normalized_data[valid_indices]
    if len(valid_data) == 0:
        raise ValueError(f"Series '{name}' contains no finite values; cannot normalize.")"""

    content = content.replace(search_block, replace_block)

    with open('src/waterSpec/preprocessor.py', 'w') as f:
        f.write(content)

update_preprocessor()
