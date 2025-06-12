from chapter1_network import Network as net1
from chapter2_network import Network as net2

import numpy as np
import pdb

# For development purposes and checking that the full-matrix approach works on the chapter 2 script
np.random.seed(42)  # For reproducibility
test1 = net1([2, 3, 1])

np.random.seed(42)  # For reproducibility
test2 = net2([2, 3, 1])

print("Checking that initialization is the same:")
# Compare each array in the lists
weights_equal = all(np.array_equal(w1, w2) for w1, w2 in zip(test1.weights, test2.weights))
biases_equal = all(np.array_equal(b1, b2) for b1, b2 in zip(test1.biases, test2.biases))

print(f"Weights equal: {weights_equal}")
print(f"Biases equal: {biases_equal}")

# If they're still False, let's debug:
if not weights_equal:
    print("\nDebugging weights:")
    for i, (w1, w2) in enumerate(zip(test1.weights, test2.weights)):
        print(f"Layer {i} weights equal: {np.array_equal(w1, w2)}")
        print(f"Layer {i} shapes: {w1.shape} vs {w2.shape}")