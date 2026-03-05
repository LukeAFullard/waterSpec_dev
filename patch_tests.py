import re

with open("tests/test_plotting.py", "r") as f:
    content = f.read()

# Fix the finally blocks that were left over from try/except removal
content = re.sub(r'    finally:\n        # Restore the original backend\n        plt\.switch_backend\(original_backend\)', r'    # Restore the original backend\n    plt.switch_backend(original_backend)', content)

with open("tests/test_plotting.py", "w") as f:
    f.write(content)
