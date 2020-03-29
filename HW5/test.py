from scipy.stats import entropy
import numpy as np
a = np.array([1, 2, 6, 4, 2, 3, 2])
u, counts = np.unique(a, return_counts=True)
print(entropy([1/2,1/2], base=2))