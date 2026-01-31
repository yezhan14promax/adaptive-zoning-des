import os
import time


def _outputs_root(base_dir):
    base_dir = os.path.abspath(base_dir)
    candidates = [
        os.path.join(base_dir, "outputs", "outputs"),
        os.path.join(base_dir, "outputs"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def timestamped_output_root(base_dir, prefix="figure"):
    root = _outputs_root(base_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(root, f"{prefix}_{stamp}")


def normalize_output_root(base_dir, output_root=None, prefix="figure"):
    if output_root:
        return os.path.abspath(output_root)
    return timestamped_output_root(base_dir, prefix=prefix)
