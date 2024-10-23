import numpy as np

def compare_np_array_with_cache(a: np.ndarray, b: np.ndarray) -> bool:
    if a is None or b is None:
        return False

    if a.shape != b.shape:
        return False
    
    return np.equal(
        a, b
    ).all()